import torch, torchvision, random, math, os, wandb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import Flickr30k
from omegaconf import OmegaConf
from diffusion.utils.utils import load_config
from transformers import CLIPTokenizer
from PIL import Image
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

"Emedding dim: (B, P, D), P = num_patches, D = embed size"

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_ch, patch_size, embed_dim, p_dropout):
        super().__init__()
        D = embed_dim
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        P = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, P+1, D))
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                                             # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)                             # (B, P, D), where P = H/ps * W/ps
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # (B, P+1, D)
        x = x + self.pos_embed                                       # (B, P+1, D)
        return self.dropout(x)

class AttentionHead(nn.Module):
    def __init__(self, head_size, embed_dim, p_dropout):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_dim, head_size)
        self.key = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x, mask=None):
        """
        mask: is a tensor of sahpe (B, T) where the index of eos_token is True
        """
        B, T, D = x.shape
        q = self.query(x)                                        # (B, T, D)
        k = self.key(x)                                          # (B, T, D)
        v = self.value(x)                                        # (B, T, D)
        attn = q @ k.transpose(1, 2) * (self.head_size ** -0.5)  # (B, T, T)
        if mask is not None:
            pad_mask = mask.to(torch.bool).unsqueeze(1)          # (B, 1, T)
            attn = attn.masked_fill(~pad_mask, float('-1e9'))    # (B, T, T)
        attn = F.softmax(attn)                                   # (B, T, T)
        attn = self.dropout(attn)                                # (B, T, T)
        return attn @ v                                          # (B, T, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, p_dropout):
        super().__init__()
        head_size = embed_dim // n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(head_size, embed_dim, p_dropout) for _ in range(n_heads)]
        )

    def forward(self, x):
        pass





def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(path="config/base.yaml", env=env)
    print("Configuration loaded")
    config.env = env
    print(f"Seed {config.seed} Device {device}")

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    # init_testing()
    # test_data_loader(config)
    # get_norm_mu_sigma(config)

    # train_test_model(config)

    test_architecture()


def test_architecture():
    input = torch.zeros(2, 3, 224, 224)
    model = PatchEmbedding(img_size=224, in_ch=3, patch_size=32, embed_dim=512, p_dropout=0.1)
    out = model(input)
    print(f"out.shape {out.shape}")

    # input = torch.zeros(3, 77, 512)
    # model = TransformerBlock(n_heads=4, embed_dim=512, mlp_ratio=4, p_dropout=0.1)
    # out = model(input)
    # print(f"out.shape {out.shape}")

    # input = torch.zeros(5, 3, 224, 224)
    # model = VisionTransformer(
    #     img_size=224,
    #     in_ch=3,
    #     patch_size=32,
    #     depth=6,
    #     n_heads=4,
    #     embed_dim=512,
    #     mlp_ratio=4,
    #     p_dropout=0.1
    # )
    # out = model(input)
    # print(f"out.shape {out.shape}")

if __name__ == "__main__":
    main()