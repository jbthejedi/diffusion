import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import transform.transformer_modules as tm

from torchinfo import summary
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

from utils import utils as u


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=32, in_ch=3, embed_dim=512,
        depth=6, num_heads=8, mlp_ratio=4.0, qkv_bias=True, dropout=0.1
    ):
        super().__init__()
        self.patch_embed = tm.PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[tm.TransformerBlock(embed_dim, dropout, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        return x[:, 0] # return cls_token


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        cw_size=77,        # cw_size
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.eos_token = 49407
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cw_size, embed_dim))
        self.transformer_blocks = nn.Sequential(
            *[tm.TransformerBlock(embed_dim, dropout, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens : torch.Tensor):
        # tokens.shape -> (B, L)
        B, L = tokens.shape
        x = self.token_embed(tokens) # (B, L, embed_dim)
        x = x + self.pos_embed[:, :L, :] # (B, L, embed_dim)
        x = self.transformer_blocks(x) # (B, 50, 512)
        x = self.norm(x)  # (B, 50, 512)
        eos_mask = tokens.eq(self.eos_token)
        eos_mask[:, -1] = True
        eos_pos = eos_mask.int().argmax(dim=1)
        return x[torch.arange(B), eos_pos]


class CLIPModel(nn.Module):
    def __init__(
        self, img_size=224, patch_size=32, vocab_size=49408, cw_size=77,
        embed_dim=512, proj_dim=512, vision_depth=6, text_depth=6, num_heads=8
    ):
        super().__init__()
        self.visual = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=vision_depth, num_heads=num_heads
        )
        self.textual = TextTransformer(
            vocab_size=vocab_size, cw_size=cw_size, embed_dim=embed_dim,
            depth=text_depth, num_heads=num_heads
        )
        self.visual_proj = nn.Linear(embed_dim, proj_dim)
        self.textual_proj = nn.Linear(embed_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def forward(self, image, text):
        i_features = self.visual(image)    # [B, embed_dim]
        t_features = self.textual(text)    # [B, embed_dim]
        # project
        i_proj = self.visual_proj(i_features)  # [B, proj_dim]
        t_proj = self.textual_proj(t_features) # [B, proj_dim]
        # normalize
        i_proj = F.normalize(i_proj, dim=-1)
        t_proj = F.normalize(t_proj, dim=-1)
        # scaled cosine similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * i_proj @ t_proj.t()
        return logits, i_proj, t_proj
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits):
        # logits: [B, B]
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_i2t = self.loss_fn(logits / self.temperature, labels)
        loss_t2i = self.loss_fn(logits.t() / self.temperature, labels)
        return (loss_i2t + loss_t2i) / 2


class Flickr30kDataset(Dataset):
    def __init__(self, images_root: str, captions_file: str, transform=None):
        self.images_root = images_root
        self.transform = transform

        # Load captions: split on any whitespace (tab, space, etc.)
        self.captions = {}
        with open(captions_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty or comment lines
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)  # split on first whitespace
                if len(parts) < 2:
                    # malformed line, skip
                    continue
                img_id, caption = parts
                # fn = img_id.split('#')[0]
                fn = img_id.split('#')[0].strip().strip('",') 
                self.captions.setdefault(fn, []).append(caption)
        self.filenames = sorted(self.captions.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img_path = os.path.join(self.images_root, fn)
        print(f"img_path: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        cap = random.choice(self.captions[fn])
        return img, cap


def collate_fn(batch):
    # batch is list of (img_tensor, caption_str)
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, caps


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


def test_data_loader():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # optional: add Norm
    ])

    dataset = Flickr30kDataset(
        images_root="/Users/justinbarry/projects/flickr30k_entities/flickr30k/Images",
        captions_file="/Users/justinbarry/projects/flickr30k_entities/flickr30k/captions.txt",
        transform=tf
    )

    show_random_samples_sequential(dataset, num_samples=10)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val   = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
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


def train_test_model(config):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # optional: add Norm
    ])

    dataset = Flickr30kDataset(
        images_root="/Users/justinbarry/projects/flickr30k_entities/flickr30k/Images",
        captions_file="/Users/justinbarry/projects/flickr30k_entities/flickr30k/captions.txt",
        transform=tf
    )

    show_random_samples_sequential(dataset, num_samples=10)

    n = len(dataset)
    n_train, n_val = int(n * 0.8), n - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )


def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = u.load_config(path="../config/base.yaml", env=env)
    print("Configuration loaded")
    config.env = env
    print(f"Seed {config.seed} Device {device}")

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    # init_testing()
    # test_data_loader()
    train_test_model(config)


if __name__ == "__main__":
    main()