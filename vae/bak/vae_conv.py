import torch
import sys
import lpips
import wandb
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from torchinfo import summary

from typing import Callable
from omegaconf import OmegaConf
from PIL import Image

torch.backends.cudnn.benchmark = True

class VAEConv(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3,
        latent_dims=128, p_dropout=0.1, image_size=224
    ):
        """
        64 -> 128 -> 256 -> 256
        mu: latent_dims
        logvar: latent_dims
        """
        super().__init__()
        self.image_size = image_size
        self.conv1 = DoubleConv(in_channels, 64, p_dropout)           # -> (,64,224,224)
        self.pool1 = nn.MaxPool2d(2)                       # -> (,64,112,122) 

        self.conv2 = DoubleConv(64, 128, p_dropout)                   # -> (,128,112,112)
        self.pool2 = nn.MaxPool2d(2)                       # -> (,128,56,56) 

        self.conv3 = DoubleConv(128, 256, p_dropout)                  # -> (,256,56,56)
        self.pool3 = nn.MaxPool2d(2)                       # -> (,256,28,28) 

        # Refine latent features
        self.conv4 = DoubleConv(256, 256, p_dropout)                  # -> (,256,28,28)
        C, H, W = self._get_flattened_dims(in_channels)
        self.flattened_dims = C*H*W
        self.chw = C, H, W
        self.fc_mu = nn.Linear(in_features=self.flattened_dims, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.flattened_dims, out_features=latent_dims)

        ###### ADDED ######
        # Xavier initialization sets layer weights so that the variance
        # of activations stays roughly the same across layers by scaling weights with
        # \frac{1}{sqrt{fan_in + fan_out}}, where
        # fan_in = Number of input units to the layer
        # → How many inputs each neuron receives.

        # fan_out = Number of output units from the layer
        # → How many outputs each neuron produces.
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)

        self.fc_up = nn.Linear(latent_dims, self.flattened_dims) # -> (latent_dims, C*H*W)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        #### ADDED ####
        self.head = nn.Conv2d(64, out_channels, 1, stride=1)
        # 256 -> 128 -> 64 -> 64
    
    
    def _get_flattened_dims(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.image_size, self.image_size)
            x : torch.Tensor = self.pool1(self.conv1(dummy))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.conv4(x)
            _, C, H, W = x.shape
            return C, H, W


    def encode(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        # x = x.view(-1, self.flattened_dims)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recons = self.decode(z)
        return recons, mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def decode(self, z):
        ###### ADDED ######
        x = torch.tanh(self.fc_up(z))
        C, H, W = self.chw
        x = x.view(-1, C, H, W)
        x = torch.relu(self.bn3(self.up3(x)))
        x = torch.relu(self.bn2(self.up2(x)))
        x = torch.relu(self.bn1(self.up1(x)))
        return torch.sigmoid(self.head(x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
        )

    def forward(self, x):
        return self.net(x)


class VAEMlp(nn.Module):
    def __init__(self, input_dims=784, hidden_dims=256, latent_dims=16):
        super().__init__()
        ## ENCODER ##
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.mu = nn.Linear(hidden_dims, latent_dims)
        self.logvar = nn.Linear(hidden_dims, latent_dims)
        ## DECODER ##
        self.fc2 = nn.Linear(latent_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, input_dims)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recons = self.decode(z)
        return recons, mu, z
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        

    def decode(self, z):
        x = torch.relu(self.fc2(z))
        return self.fc3(x)
        
    
    def encode(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar



def vae_loss(input, target, mu : torch.Tensor, logvar : torch.Tensor, loss : Callable, lpips_fn):
    # Analytic KL: -0.5 * (1 - log(sig^2) - mu^2 - e^(log(sig^2)))
    recon_loss = loss(input, target, reduction="sum") # F.binary_cross_entropy_with_logits()
    kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    lpips_loss = lpips_fn(target, input, normalize=True).view(-1).sum()

    return (recon_loss + lpips_loss + kl_div_loss) / input.size(0)


def show_reconstructions(config, model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(config.device)
            if config.model_type == 'mlp': inputs = torch.flatten(inputs, start_dim=1)
            x_hat, _, _ = model(inputs)
            # x_hat = torch.sigmoid(x_hat)  # output logits → probability
            break  # take only one batch

    # reshape to (B, 1, 28, 28)
    inputs = inputs.cpu()
    inputs = inputs.view(
        -1, config.num_channels, config.image_size, config.image_size).cpu()
    x_hat = x_hat.view(
        -1, config.num_channels, config.image_size, config.image_size).cpu()

    # concat input/output for side-by-side view
    print(f"inputs.shape {inputs.shape}")
    print(f"x_hat.shape {x_hat.shape}")
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
        # x_sample = torch.sigmoid(x_sample).view(
        #     -1, config.num_channels, config.image_size, config.image_size).cpu()
        x_sample = x_sample.view(
            -1, config.num_channels,
            config.image_size, config.image_size
        ).cpu()

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
            x_hat = model.decode(z).view(
                1, config.num_channels,
                config.image_size, config.image_size
            )
            interpolated.append(x_hat.cpu())

        # Stack and visualize
        interpolated = torch.cat(interpolated, dim=0)
        grid = torchvision.utils.make_grid(interpolated, nrow=num_steps)
        plt.figure(figsize=(num_steps, 2))
        plt.axis("off")
        plt.title("Latent Interpolation")
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.show()


def get_train_val_dl(dataset, config):
    train_len = int(len(dataset) * config.p_train_len)

    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        ### ADDED ####
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    return train_dl, val_dl


def load_and_test_model(config):
    api = wandb.Api()
    artifact_name = config.artifact_name
    try:
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        raw_sd = torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu")

        # strip out the "_orig_mod." prefix from every key
        fixed_sd = {}
        for k, v in raw_sd.items():
            # Python 3.9+: you can use k.removeprefix()
            new_k = k.removeprefix("_orig_mod.")
            fixed_sd[new_k] = v

        # Load model
        model = VAEConv(
            in_channels=config.num_channels,
            out_channels=3,
            latent_dims=config.latent_dims,
            p_dropout=config.p_dropout,
            image_size=config.image_size
        )
        # model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu"))
        model.load_state_dict(fixed_sd)
        model.eval()
        print("Model loaded successfully.")

        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(config.image_size),
            T.ToTensor(),
        ])
        dataset = CelebADataset(
            root_dir=os.path.join(config.data_root, "img_align_celeba"),
            transform=transform
        )
        _, val_dl = get_train_val_dl(dataset, config)
        print("Dataloaders created")

        show_reconstructions(config, model, val_dl)
        show_samples(config, model, latent_dim=config.latent_dims)
        interpolate_latents(config, model, dataset, num_steps=10)
    except wandb.CommError as e:
        print(f"Artifact not found: {artifact_name}")
        print(f"Error: {e}")
        exit(0)


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_names = sorted([f for f in os.listdir(root_dir) if f.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def train_test_model(config):
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )
    if config.dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.ToTensor(),
            ])
        )
    elif config.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ])
        )
    elif config.dataset_name == 'celeba':
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(config.image_size),
            T.ToTensor(),
        ])
        # dataset = datasets.ImageFolder(
        #     root=os.path.join(config.data_root, "img_align_celeba"),
        #     transform=transform
        # )
        dataset = CelebADataset(
            root_dir=os.path.join(config.data_root, "img_align_celeba"),
            transform=transform
        )
        # dataset = datasets.CelebA(
        #     root=config.data_root,
        #     split="train",
        #     target_type="none",
        #     download=True,
        #     transform=transform
        # )
    else:
        raise Exception('No dataset provided')

    train_dl, val_dl = get_train_val_dl(dataset, config)
    model = VAEConv(
        in_channels=config.num_channels,
        out_channels=3,
        latent_dims=config.latent_dims,
        p_dropout=config.p_dropout,
        image_size=config.image_size
    )
    if config.compile: model = torch.compile(model)
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Training Start")

    lpips_fn = lpips.LPIPS(net='vgg').to(config.device)

    ### ADDED ###
    best_val_loss = float('inf')
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs + 1}")
        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(pbar):
                inputs = inputs.to(config.device)
                # inputs = torch.flatten(inputs, start_dim=1)
                recons, mu, logvar = model(inputs)
                optimizer.zero_grad()
                # loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy_with_logits)

                #### ADDED #####
                loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy, lpips_fn)
                loss.backward()

                ###### ADDED ######
                #### Gradient Clipping #####
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(train_loss=f"{train_loss / (batch_idx + 1):.2f}")
        train_epoch_loss = train_loss / len(train_dl)
        tqdm.write(f"Train Loss {train_epoch_loss:.2f}")

        with tqdm(val_dl, desc="Validation") as pbar:
            ###### ADDED ######
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                for batch_idx, (inputs, _) in enumerate(pbar):
                    inputs = inputs.to(config.device)
                    # inputs = torch.flatten(inputs, start_dim=1)
                    recons, mu, logvar = model(inputs)
                    # loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy_with_logits)
                    loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy, lpips_fn)
                    val_loss += loss.item()
                    pbar.set_postfix(val_loss=f"{val_loss / (batch_idx + 1) :.2f}")
            val_epoch_loss = val_loss / len(val_dl)
        tqdm.write(f"val Loss {val_epoch_loss:.2f}")

        ###### ADDED ######
        if config.save_model and (val_epoch_loss < best_val_loss):
            best_val_loss = val_epoch_loss
            tqdm.write(f"New best val loss: {best_val_loss:.4f} — overwriting best-model.pth")
            # overwrite local file
            torch.save(model._orig_mod.state_dict(), "best-model.pth")
        ###### ADDED ######
        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "val/loss": val_epoch_loss,
        })
    if config.save_model:
        tqdm.write("Logging final best-model.pth to wandb as a single artifact…")
        artifact = wandb.Artifact(name=f"{config.name}-best-model", type="model")
        artifact.add_file("best-model.pth")
        wandb.log_artifact(artifact)
        tqdm.write("Done.")
    
    show_reconstructions(config, model, val_dl, num_images=8)
    show_samples(config, model, latent_dim=16, num_images=8)
    interpolate_latents(config, model, dataset, num_steps=10)


def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")

    if config.summary:
        if config.model_type == 'mlp':
            m = VAEMlp(input_dims=784, hidden_dims=256, latent_dims=16)
            input_shape = (1, 784)
        elif config.model_type == 'conv':
            m = VAEConv(config.num_channels, 64, config.latent_dims, config.p_dropout, config.image_size)
            input_shape = (1, config.num_channels, config.image_size, config.image_size)
        else:
            raise Exception("Model type for summary not provided or not valid")
        summary(
            m, 
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params"],
            # verbose=2
        )
        exit()

    if config.load_and_test_model:
        load_and_test_model(config)
    elif config.train_model:
        train_test_model(config)


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


if __name__ == '__main__':
    main()