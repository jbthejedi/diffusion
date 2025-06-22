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


class VAEMlp(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super().__init__()
        self.hidden1 = nn.Linear(input_dims, hidden_dims)
        self.mu = nn.Linear(hidden_dims, latent_dims)
        self.logvar = nn.Linear(hidden_dims, latent_dims)
        self.fc1 = nn.Linear(latent_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, input_dims)
    

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    

    def encode(self, x):
        x = self.hidden1(x)
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps 


    def decode(self, z):
        x = torch.relu(self.fc1(z))
        return self.fc2(x)


class VAEConv(nn.Module):
    def __init__(self):
        super().__init__()


def vae_loss(x, x_hat, mu : torch.Tensor, logvar : torch.Tensor, loss):
    recon_loss = loss(x_hat, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_div) / x.size(0)


def show_reconstructions(config, model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(config.device)
            if config.model_type == 'mlp': inputs = torch.flatten(inputs, start_dim=1)
            x_hat, _, _ = model(inputs)
            x_hat = torch.sigmoid(x_hat)  # output logits → probability
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
    if config.dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ])
        )
    else:
        raise Exception("No dataset specified")
        exit()

    train_dl, val_dl = get_train_val_dl(dataset, config)
    
    if config.model_type == 'mlp':
        model = VAEMlp(input_dims=784, hidden_dims=256, latent_dims=config.latent_dims)
    else:
        raise Exception("No model specified")
        exit()
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(1, config.n_epochs + 1):
        with tqdm(train_dl, desc="Training") as pbar:
            model.train()
            tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
            total_train_loss = 0.0
            for inputs, _ in pbar:
                inputs = inputs.to(config.device)
                if config.model_type == 'mlp': inputs = torch.flatten(inputs, start_dim=1)

                recons, mu, logvar = model(inputs)
                optimizer.zero_grad()

                f = F.binary_cross_entropy_with_logits if config.dataset_name == 'mnist' else F.mse_loss
                loss : torch.Tensor = vae_loss(inputs, recons, mu, logvar, f)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            train_epoch_loss = total_train_loss / len(train_dl)
        tqdm.write(f"Train Loss {train_epoch_loss:.6f}")

        with tqdm(val_dl, desc="Validation") as pbar:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for inputs, _ in pbar:
                    inputs = inputs.to(config.device)
                    if config.model_type == 'mlp': inputs = torch.flatten(inputs, start_dim=1)
                    recons, mu, logvar = model(inputs)

                    f = F.binary_cross_entropy_with_logits if config.dataset_name == 'mnist' else F.mse_loss
                    loss : torch.Tensor = vae_loss(inputs, recons, mu, logvar, f)
                    total_val_loss += loss.item()

                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                val_epoch_loss = total_val_loss / len(val_dl)
            tqdm.write(f"Val Loss {val_epoch_loss:.6f}")
    if config.local_visualization:
        show_reconstructions(config, model, val_dl, num_images=8)
        show_samples(config, model, latent_dim=config.latent_dims)
        


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
            model = VAEMlp(input_dims).to(config.device)
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

    if config.summary:
        if config.model_type == 'mlp':
            m = VAEMlp(input_dims=784, hidden_dims=256, latent_dims=64)
            input_shape = (1, 784)
        else:
            m = VAEConv(config.num_channels, 64, config.p_dropout, config.image_size)
            input_shape = (1, config.num_channels, config.image_size, config.image_size)
        summary(
            m, 
            input_size=input_shape,  # Add batch size here!
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