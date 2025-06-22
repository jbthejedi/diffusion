import random
import torch
import vae.models.vae as vae
import vae.models.autoencoderkl as aekl
import sys
import lpips
import wandb
import os
import warnings
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from tqdm import tqdm
from torchinfo import summary

from typing import Callable
from omegaconf import OmegaConf
from PIL import Image

patterns = [
    r".*The parameter 'pretrained' is deprecated.*",
    r".*pin_memory' argument is set as true but not supported.*",
    r".*Arguments other than a weight enum*",
    # add any other regex-style patterns here
]
for pat in patterns:
    warnings.filterwarnings(
        "ignore",
        message=pat,
        category=UserWarning,
    )

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def vae_loss(input, target, mu : torch.Tensor, logvar : torch.Tensor, loss : Callable, lpips_fn, config):
    # Analytic KL: -0.5 * (1 - log(sig^2) - mu^2 - e^(log(sig^2)))
    recon_loss = loss(input, target, reduction="sum") # L1
    kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    lpips_loss = lpips_fn(target, input, normalize=True).view(-1).sum()

    loss = (recon_loss + config.percep_weight * lpips_loss + config.kl_weight * kl_div_loss) / input.size(0)
    return loss


def l1_kl_percep_loss(recons, inputs, mu, logvar, config, LPIPS_FN):
    return vae_loss(recons, inputs, mu, logvar, F.l1_loss, LPIPS_FN, config)


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
        dummy = torch.zeros(1, config.num_channels, config.image_size, config.image_size).to(config.device)
        z, mu, logvar = model.enc(dummy)
        latent_shape = z.shape[1:]
        # z = torch.randn(num_images, latent_dim).to(config.device)
        # input = torch.empty(num_images, latent_dim, 8, 8)
        # z = torch.randn_like(input).to(config.device)
        z = torch.randn(num_images, *latent_shape).to(config.device)
        print(f"z.shape {z.shape}")
        # x_sample = model.decode(z)
        x_sample = model.dec(z)
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
        # mu1, _ = model.encode(x1)
        # mu2, _ = model.encode(x2)
        
        x_hat1, mu1, _ = model.enc(x1)
        x_hat2, mu2, _ = model.enc(x2)

        # Linearly interpolate in latent space
        interpolated = []
        for alpha in torch.linspace(0, 1, steps=num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            # x_hat = model.decode(z).view(
            x_hat = model.dec(z).view(
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
        raw_sd = torch.load(f"{artifact_dir}/best-model.pth", map_location="cpu")

        # Load model
        if config.model_type == 'conv':
            model = vae.VAEConv(
                in_channels=config.num_channels,
                out_channels=3,
                latent_dims=config.latent_dims,
                p_dropout=config.p_dropout,
                image_size=config.image_size
            )
        elif config.model_type == 'aekl':
            model = aekl.AutoencoderKLSmall(
                in_channels=config.num_channels,
                base_channels=(128,256,512,512),
                latent_channels=4,
                num_groups=32
                ) 
        else: raise Exception("No model specified")
        model.load_state_dict(torch.load(f"{artifact_dir}/best-model.pth", map_location="cpu"))
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

        # show_reconstructions(config, model, val_dl)
        show_samples(config, model, latent_dim=4)
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
        dataset = CelebADataset(
            root_dir=os.path.join(config.data_root, "img_align_celeba"),
            transform=transform
        )
    else:
        raise Exception('No dataset provided')

    if config.test_run:
        idxs = random.sample(range(len(dataset)), config.sample_size)
        dataset = Subset(dataset, idxs)

    train_dl, val_dl = get_train_val_dl(dataset, config)
    if config.model_type == 'conv':
        model = vae.VAEConv(
            in_channels=config.num_channels,
            out_channels=3,
            latent_dims=config.latent_dims,
            p_dropout=config.p_dropout,
            image_size=config.image_size
        )
    elif config.model_type == 'aekl':
       model = aekl.AutoencoderKLSmall(
           in_ch=config.num_channels,
           base_ch=(128,256,512),
           latent_ch=config.latent_channels,
           num_groups=config.num_groups,
        ) 
    if config.compile: model = torch.compile(model)
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Training Start")
    LPIPS_FN = lpips.LPIPS(net='vgg').to(device)

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
                loss = l1_kl_percep_loss(recons, inputs, mu, logvar, config, LPIPS_FN)

                #### ADDED #####
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
                    loss = l1_kl_percep_loss(recons, inputs, mu, logvar, config, LPIPS_FN)
                    val_loss += loss.item()
                    pbar.set_postfix(val_loss=f"{val_loss / (batch_idx + 1) :.2f}")
            val_epoch_loss = val_loss / len(val_dl)
        tqdm.write(f"val Loss {val_epoch_loss:.2f}")

        ###### ADDED ######
        if config.save_model and (val_epoch_loss < best_val_loss):
            best_val_loss = val_epoch_loss
            tqdm.write(f"New best val loss: {best_val_loss:.4f} — overwriting best-model.pth")
            # overwrite local file
            if config.compile:
                torch.save(model._orig_mod.state_dict(), "best-model.pth")
            else:
                torch.save(model.state_dict(), "best-model.pth")
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
    
    if config.local_visualization:
        show_reconstructions(config, model, val_dl, num_images=8)
        show_samples(config, model, latent_dim=16, num_images=8)
        interpolate_latents(config, model, dataset, num_steps=10)

def print_summary(config):
    if config.model_type == 'mlp':
        model = vae.VAEMlp(input_dims=784, hidden_dims=256, latent_dims=16)
        input_shape = (1, 784)
    elif config.model_type == 'conv':
        model = vae.VAEConv(config.num_channels, 64, config.latent_dims, config.p_dropout, config.image_size)
        input_shape = (1, config.num_channels, config.image_size, config.image_size)
    elif config.model_type == 'aekl':
        model = aekl.AutoencoderKLSmall(
            in_channels=config.num_channels,
            base_channels=(128,256,512,512),
            latent_channels=4,
            num_groups=32
        ) 
        input_shape = (1, 3, 128, 128)
    elif config.model_type == 'aekl_bak':
        # model = aeklbak.DownEncoderBlock2D(in_ch=3, out_ch=16, num_groups=4)
        # model = aeklbak.MidBlock(32)
        # model = aeklbak.Encoder()
        # input_shape = (1, config.num_channels, config.image_size, config.image_size)
        # input_shape = (1, 32, 64, 64)
        model = aekl.AutoencoderKLSmall(
            in_channels=3,
            base_channels=(128,256,512,512),
            latent_channels=4,
            num_groups=2
        )
        input_shape = (1, 3, 128, 128)
    else:
        raise Exception("Model type for summary not provided or not valid")
    summary(
        model, 
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params"],
        # verbose=2
    )

def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")
    if config.device == 'cuda':
        torch.set_float32_matmul_precision('high')

    if config.summary:
        print_summary(config)
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