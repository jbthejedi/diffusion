import torch
import random
import os
import wandb
import matplotlib.pyplot as plt

import torchvision.datasets as datasets

from tqdm import tqdm
from torchinfo import summary
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
from transformers import CLIPTokenizer
from omegaconf import OmegaConf

from diffusion.clip.models.clip import *
from diffusion.utils import utils as u

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from pathlib import Path

class Flickr30kDataset(Dataset):
    def __init__(self, images_root, captions_file, transform=None):
        self.images_root = images_root
        self.transform = transform

        self.captions = {}
        with Path(captions_file).open('r') as f:
            for line in f:
                line = line.strip()
                if line is None or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                img_id, caption = parts
                filename = img_id.split("#")[0].strip().strip('"').strip(',')
                self.captions.setdefault(filename, []).append(caption)
        self.filenames = sorted(self.captions.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.images_root, filename)
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        caption = random.choice(self.captions[filename])
        return img, caption


class CLIPCollator:
    def __init__(self, tokenizer_name='openai/clip-vit-base-patch32', max_length=77, device='cpu'):
        self.tokenizer : CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        tokenized = self.tokenizer(
            list(captions),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        attention_mask = tokenized.attention_mask
        input_ids = tokenized.input_ids
        return images, input_ids, attention_mask


def show_random_samples_sequential(dataset, num_samples=5):
    """
    Displays num_samples random (image, caption) pairs one at a time.
    Close each window to see the next.
    """
    idxs = random.sample(range(len(dataset)), num_samples)
    for idx in idxs:
        img, cap = dataset[idx]
        # convert [C,H,W] tensor to HxWxC numpy
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.title(cap, fontsize=12, wrap=True)
        plt.axis("off")
        
        # This call will block until you close the window
        plt.show()
        plt.close()


def test_data_loader(config):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # optional: add Norm
    ])

    dataset : Dataset = Flickr30kDataset(
        images_root=f"{config.data_root}/flickr30k/Images",
        captions_file=f"{config.data_root}/flickr30k/captions.txt",
        transform=tf,
    )

    show_random_samples_sequential(dataset, num_samples=10)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val   = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    def collate_fn(batch):
        # batch is list of (img_tensor, caption_str)
        imgs, caps = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        return imgs, caps

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )

    imgs, caps = next(iter(train_loader))
    print(f"Batch images: {imgs.shape}")
    print(f"Batch captions (first 3): {caps[:3]}")


def init_testing():
    model = VisionTransformer(img_size=224)
    input = torch.zeros(1, 3, 224, 224)
    out = model(input)
    print(f"out.shape {out.shape}")

    model = TextTransformer(vocab_size=3798, cw_size=77)
    input = torch.zeros(20, 77, dtype=torch.long)
    out = model(input)
    print(f"out.shape {out.shape}")

    # dummy inputs
    batch_size, seq_len = 4, 16
    dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    dummy_tokens = torch.randint(0, 49408, (batch_size, seq_len))

    model = CLIPModel()
    logits, i_emb, t_emb = model(dummy_imgs, dummy_tokens)
    loss = ContrastiveLoss()(logits)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")


def get_norm_mu_sigma(config):
    # Dataset mean: [0.44408291578292847, 0.4211477041244507, 0.38473448157310486]
    # Dataset std:  [0.2750486731529236, 0.26723721623420715, 0.2765313982963562]

    tf = T.Compose([T.Resize((224,224)), T.ToTensor()])
    ds = Flickr30kDataset(
        images_root=f"{config.data_root}/flickr30k/Images",
        captions_file=f"{config.data_root}/flickr30k/captions.txt",
        transform=tf
    )
    loader = DataLoader(ds, batch_size=64, num_workers=4)
    loader = DataLoader(ds, batch_size=64, num_workers=4)

    # accumulate sums
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_batches = 0

    for imgs, _ in loader:
        channel_sum     += imgs.mean(dim=[0,2,3])
        channel_sq_sum  += imgs.pow(2).mean(dim=[0,2,3])
        num_batches     += 1

    mean = channel_sum / num_batches
    var  = (channel_sq_sum / num_batches) - mean**2
    std  = var.sqrt()
    print("Dataset mean:", mean.tolist())
    print("Dataset std: ", std.tolist())


def get_dataset(config):
    # Dataset mean: [0.44408291578292847, 0.4211477041244507, 0.38473448157310486]
    # Dataset std:  [0.2750486731529236, 0.26723721623420715, 0.2765313982963562]
    mean = [0.444, 0.421, 0.384]
    std = [0.275, 0.267, 0.276]
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.4,0.4,0.4,0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    base_ds = Flickr30kDataset(
        images_root=f"{config.data_root}/flickr30k/Images",
        captions_file=f"{config.data_root}/flickr30k/captions.txt",
        transform=None
    )
    if config.test_run:
        idxs = random.sample(range(len(base_ds)), config.sample_size)
        base_ds = Subset(base_ds, idxs)
    n = len(base_ds)
    perm = torch.randperm(n).tolist()
    n_train = int(0.8 * n)

    train_inds = perm[:n_train]
    val_inds   = perm[n_train:]

    train_ds = Subset(
        Flickr30kDataset(
            f"{config.data_root}/flickr30k/Images",
            f"{config.data_root}/flickr30k/captions.txt",
            transform=train_tf
        ),
        train_inds
    )
    val_ds = Subset(
        Flickr30kDataset(
            f"{config.data_root}/flickr30k/Images",
            f"{config.data_root}/flickr30k/captions.txt",
            transform=val_tf
        ),
        val_inds
    )

    return train_ds, val_ds


def get_train_val_loader(train_ds, val_ds, config):
    collator = CLIPCollator(
        tokenizer_name="openai/clip-vit-base-patch32",
        max_length=config.cw_size,
        device=device
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=4,
        collate_fn=collator
    )
    return train_loader, val_loader


def train_test_model(config):
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )
    train_ds, val_ds = get_dataset(config)
    train_loader, val_loader = get_train_val_loader(train_ds, val_ds, config)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size
    print("CLIP vocab size:", vocab_size)    # typically 49,408

    model = CLIP(
        img_size=config.img_size,
        vocab_size=vocab_size,
        cw_size=config.cw_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        proj_dim=config.proj_dim,
        vision_depth=config.vision_depth,
        text_depth=config.text_depth,
        n_heads=config.num_heads,
    )
    if config.compile: model = torch.compile(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        with tqdm(train_loader, desc="Training") as pbar:
            model.train()
            train_losses = []
            for images, token_ids, attn_mask in pbar:
                optimizer.zero_grad()

                images, token_ids = images.to(device), token_ids.to(device)
                logits, _, _ = model(images, token_ids)
                loss = ContrastiveLoss()(logits)

                
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                pbar.set_postfix(loss=f"{loss:.4f}")
            train_epoch_loss = sum(train_losses)/len(train_losses)
            tqdm.write(f"Train Loss {train_epoch_loss:.4f}")

        with tqdm(val_loader, desc="Validation") as pbar:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for images, token_ids, _ in pbar:
                    images, token_ids = images.to(device), token_ids.to(device)
                    logits, _, _ = model(images, token_ids)
                    loss = ContrastiveLoss()(logits)
                    
                    eval_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")
                val_epoch_loss = sum(eval_losses)/len(eval_losses)
                tqdm.write(f"Val Loss {val_epoch_loss:.4f}")

        if config.save_model and (val_epoch_loss < best_val_loss):
            best_val_loss = val_epoch_loss
            tqdm.write(f"New best val loss: {best_val_loss:.4f} — overwriting best-model.pth")
            if config.compile:
                torch.save(model._orig_mod.state_dict(), "best-model.pth")
            else:
                torch.save(model.state_dict(), "best-model.pth")
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
            

def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = u.load_config(path="config/base.yaml", env=env)
    print("Configuration loaded")
    config.env = env
    print(f"Seed {config.seed} Device {device}")

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    # init_testing()
    # test_data_loader(config)
    # get_norm_mu_sigma(config)

    train_test_model(config)


if __name__ == "__main__":
    main()