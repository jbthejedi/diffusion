import torch
import sys
import wandb
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchinfo import summary

from omegaconf import OmegaConf


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(num_features=out_channels),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(num_features=out_channels),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class VAEConv(nn.Module):
    def __init__(self, in_channels, latent_dims, p_dropout, image_size):
        super().__init__()
        """
        Assumptions for dimension calculations
        H_in, W_in = 224
        Conv:          {h/w}_out = [({h/w}_in + 2p - d(k - 1) - 1) / s] + 1
        ConvTranspose: {h/w}_out = ({h/w}_in - 1)s - 2p + k + output_padding
        """
        self.image_size = image_size

        self.conv0 = DoubleConv(3, 64, p_dropout)                   # -> (64, 224, 224)
        self.pool0 = nn.MaxPool2d(2)                                # -> (64, 112, 112)

        self.conv1 = DoubleConv(64, 128, p_dropout)                 # -> (128, 112, 112)
        self.pool1 = nn.MaxPool2d(2)                                # -> (128, 56, 56) 

        self.conv2 = DoubleConv(128, 256, p_dropout)                # -> (256, 56, 56)
        self.pool2 = nn.MaxPool2d(2)                                # -> (256, 28, 28)

        self.middle = DoubleConv(256, 256, p_dropout)               # -> (256, 14, 14)

        C, H, W = self._get_flattened_shape(in_channels)
        self.C, self.H, self.W = C, H, W
        print(f"flattened shape (C, H, W): ({C, H, W})")
        self.flattened_dim = C * H * W
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dims)     # -> (U, 64), U = 256*14*14 = 50176
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dims) # -> (U, 64)
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)

        self.fc_up = nn.Linear(latent_dims, C*W*H)                  # -> (64, U)
                                                            # Reshape -> (256, 14, 14)
        # # ConvTranspose: {h/w}_out = ({h/w}_in - 1)s - 2p + k + output_padding
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)          # -> (256, 28, 28)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)           # -> (64, 112, 112)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)           # -> (3, 224, 224)
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.head = nn.Conv2d(64, 3, 1, padding=0, stride=1)        # -> (3, 124, 124)


    def _get_flattened_shape(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.image_size, self.image_size)
            # d0 = self.layer0(dummy)
            d0 = self.conv0(dummy)
            x = self.pool0(d0)

            d1 = self.conv1(x)
            x = self.pool1(d1)

            d2 = self.conv2(x)
            x = self.pool2(d2)

            m = self.middle(x)
            _, C, H, W = m.shape

            return C, H, W


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # z = torch.tanh(self.fc_up(z))             # -> (, 256*14*14)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def encode(self, x):
        d0 = self.conv0(x)                              # -> (64, 224, 224)
        x = self.pool0(d0)                              # -> (64, 112, 112)

        d1 = self.conv1(x)                              # -> (128, 112, 112)
        x = self.pool1(d1)                              # -> (128, 56, 56)

        d2 = self.conv2(x)                              # -> (256, 56, 56)
        x = self.pool2(d2)                              # -> (256, 28, 28)

        m = self.middle(x)                              # -> (256, 28, 28)

        _, C, W, H = m.shape
        m = m.view(-1, C*W*H)                           # -> (1, U), U=256*28*28
        mu = self.fc_mu(m)
        logvar = self.fc_logvar(m)

        return mu, logvar
    

    def decode(self, z):
        z = torch.tanh(self.fc_up(z))                   # -> (, 256*14*14)
        z = z.view(-1, self.C, self.W, self.H)          # -> (, 256, 28, 28)
        x = torch.relu(self.bn2(self.up2(z)))       # -> (128, 56, 56)
        x = torch.relu(self.bn1(self.up1(x)))
        x = torch.relu(self.bn0(self.up0(x)))
        out = self.head(x)
        return out


def vae_loss_bce(x, x_hat, mu, logvar : torch.Tensor):
    x_hat = x_hat.clamp(min=-10, max=10)  # add this before BCEWithLogitsLoss
    recon_loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_weight = 1.0
    return (recon_loss + kl_weight * kl_div) / x.size(0)


def linear_anneal_kl(config, epoch):
    # Increase kl_weight linearly over epochs
    progress = min(epoch / config.n_epochs, config.kl_clip)
    return config.kl_start + (config.kl_end - config.kl_start) * progress


def vae_loss_mse(x, x_hat, mu, logvar, kl_weight):
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    λ = recon_loss + kl_weight * kl_div
    N = x.size(0)
    return λ / N


def show_reconstructions(config, model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(config.device)
            x_hat, _, _ = model(inputs)
            x_hat = torch.sigmoid(x_hat)  # output logits → probability
            break  # take only one batch

    # reshape to (B, 1, 28, 28)
    inputs = inputs.cpu()
    x_hat = x_hat.view(
        -1, config.num_channels, config.image_size, config.image_size).cpu()

    # concat input/output for side-by-side view
    comparison = torch.cat([inputs[:num_images], x_hat[:num_images]])
    grid = vutils.make_grid(comparison, nrow=num_images)
    plt.figure(figsize=(15, 4))
    plt.axis("off")
    plt.title("Top: Original — Bottom: Reconstruction")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


def show_samples(config, model, latent_dim=16, num_images=8):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(config.device)
        x_sample = model.decode(z)
        x_sample = torch.sigmoid(x_sample).view(
            -1, config.num_channels, config.image_size, config.image_size).cpu()

    grid = vutils.make_grid(x_sample, nrow=num_images)
    plt.figure(figsize=(15, 4))
    plt.axis("off")
    plt.title("Samples from Latent Space")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


def interpolate_latents(config, model, dataset, num_steps=10):
    model.eval()
    with torch.no_grad():
        # Select two random digits
        indices = np.random.choice(len(dataset), 2, replace=False)
        img1, _ = dataset[indices[0]]
        img2, _ = dataset[indices[1]]

        if config.model_type == "mlp":
            x1 = img1.view(1, -1).to(config.device)
            x2 = img2.view(1, -1).to(config.device)
        else:
            x1 = img1.unsqueeze(0).to(config.device) # (1, 3, 224, 224)
            x2 = img2.unsqueeze(0).to(config.device) # (1, 3, 224, 224)

        # Encode both to get means
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # Linearly interpolate in latent space
        interpolated = []
        for alpha in torch.linspace(0, 1, steps=num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            x_hat = torch.sigmoid(model.decode(z)).view(
                1, config.num_channels, config.image_size, config.image_size)
            interpolated.append(x_hat.cpu())

        # Stack and visualize
        interpolated = torch.cat(interpolated, dim=0)
        grid = torchvision.utils.make_grid(interpolated, nrow=num_steps)
        plt.figure(figsize=(num_steps, 2))
        plt.axis("off")
        plt.title("Latent Interpolation")
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.show()

def get_train_val_dl(dataset, config):
    train_len = int(len(dataset) * config.p_train_len)
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    return train_dl, val_dl

def train_test_model(config):
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )
    if config.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=config.data_root,
            download=(True if config.env == 'server' else False),
            # download=download,
            transform=T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ])
        )
    
    if config.dataset_name == 'minst':
        dataset = datasets.MNIST(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.ToTensor(),
            ]),
        )
    print("Dataset loaded")

    train_dl, val_dl = get_train_val_dl(dataset, config)
    print("Dataloaders created")

    print("Loading model")
    if config.model_type == 'mlp':
        input_dims = config.num_channels * config.image_size * config.image_size
        model = VAELinear(input_dims)
    if config.model_type == 'unet':
        model = VAEUnet256(
            in_channels=3,
            latent_dims=64,
            p_dropout=config.p_dropout,
            image_size=config.image_size
        )
    if config.model_type == 'conv':
        model = VAEConv(
            in_channels=3,
            latent_dims=64,
            p_dropout=config.p_dropout,
            image_size=config.image_size
        )
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Training Start")
    best_val_loss = float('inf')
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        model.train()
        total_loss = 0.0
        with tqdm(train_dl, desc="Training") as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(config.device)
                x_hat, mu, logvar = model(inputs)

                optimizer.zero_grad()

                if config.model_type == 'mlp':
                    inputs = torch.flatten(inputs, start_dim=1)
                    loss = vae_loss_bce(inputs, x_hat, mu, logvar)
                else:
                    recon = torch.sigmoid(x_hat)
                    kl_weight = linear_anneal_kl(config, epoch)
                    loss = vae_loss_mse(inputs, recon, mu, logvar, kl_weight)

                loss.backward()

                #### Gradient Clipping #####
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                ########## DEBUGGING ########## 
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            print(f"Bad grad in {name}")
                        grad_norm = p.grad.data.norm().item()
                        if grad_norm > 100:
                            print(f"Large grad norm in {name}: {grad_norm:.1e}")

                optimizer.step()
                total_loss += loss.item()
                
                activ = x_hat
                wandb.log({
                    "mu_max":   mu.max().item(),   "mu_min": mu.min().item(),
                    "logvar_max": logvar.max().item(), "logvar_min": logvar.min().item(),
                    "act_max": activ.max().item(),     "act_min": activ.min().item(),
                })

                pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
        train_epoch_loss = total_loss / len(train_dl)
        tqdm.write(f"Train Loss: {train_epoch_loss:.2f}")

        with tqdm(val_dl, desc="Validation") as pbar:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for inputs, _ in pbar:
                    inputs = inputs.to(config.device)
                    x_hat, mu, logvar = model(inputs)

                    if config.model_type == 'mlp':
                        inputs = torch.flatten(inputs, start_dim=1)
                        x_hat, mu, logvar = model(inputs)
                    else:
                        recon = torch.sigmoid(x_hat)
                        kl_weight = linear_anneal_kl(config, epoch)
                        loss = vae_loss_mse(inputs, recon, mu, logvar, kl_weight)

                    total_loss += loss.item()
                    pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
                val_epoch_loss = total_loss / len(val_dl)

        tqdm.write(f"Val Loss: {val_epoch_loss:.2f}")

        if config.save_model and (val_epoch_loss < best_val_loss):
            best_val_loss = val_epoch_loss
            tqdm.write("Writing best model...")
            torch.save(model.state_dict(), "best_model.pth")
            artifact = wandb.Artifact(name=f"{config.name}_best_model", type="model")
            artifact.add_file("best_model.pth")
            wandb.log_artifact(artifact)
            tqdm.write("Model written.")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "val/loss": val_epoch_loss,
            "kl_weight": kl_weight
        })
    
    if config.local_visualization:
        show_reconstructions(config, model, train_dl)
        # show_samples(config, model, latent_dim=16, num_images=10)
        interpolate_latents(config, model, dataset, num_steps=10)


def load_config(env="local"):
    base_config = OmegaConf.load("config/base.yaml")

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        env_config = OmegaConf.load(env_path)
        # Merges env_config into base_config (env overrides base)
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config

    return config


def load_and_test_model(config):
    api = wandb.Api()
    artifact_name = config.artifact_name
    try:
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()

        # Load model
        input_dims = config.num_channels * config.image_size * config.image_size
        if config.model_type == 'mlp':
            model = VAELinear(input_dims).to(config.device)
        if config.model_type == 'unet':
            model = VAEUnet256(
                in_channels=3,
                latent_dims=64,
                p_dropout=config.p_dropout,
                image_size=config.image_size
            ).to(config.device)
        else:
            model = VAEConv(
                in_channels=3,
                latent_dims=64,
                p_dropout=config.p_dropout,
                image_size=config.image_size
            )
        model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu"))
        model.eval()
        print("Model loaded successfully.")

        dataset = datasets.CIFAR10(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ])
        )
        _, val_dl = get_train_val_dl(dataset, config)
        print("Dataloaders created")

        # show_reconstructions(config, model, val_dl)
        interpolate_latents(config, model, dataset, num_steps=10)
    except wandb.CommError as e:
        print(f"Artifact not found: {artifact_name}")
        print(f"Error: {e}")
        exit(0)


def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")

    # for epoch in range(1, config.n_epochs):
    #     print(linear_anneal_kl(config, epoch))
    # exit()

    if config.summary:
        m = VAEConv(config.num_channels, 64, config.p_dropout, config.image_size)
        summary(
            m, 
            input_size=(1, config.num_channels, config.image_size, config.image_size),  # Add batch size here!
            col_names=["input_size", "output_size", "num_params"],
            # verbose=2
        )
        exit()

    if config.load_and_test_model:
        load_and_test_model(config)
    elif config.train_model:
        train_test_model(config)


if __name__ == '__main__':
    main()