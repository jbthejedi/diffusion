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
    
    def forward(self, x : torch.Tensor):
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
        mask: is a tensor of sahpe (B, T) where the tokens are 1 and the pad is 0.
        """
        B, T, D = x.shape
        q = self.query(x)                                        # (B, T, H) 
        k = self.key(x)                                          # (B, T, H)
        v = self.value(x)                                        # (B, T, H)
        attn = q @ k.transpose(1, 2) * (self.head_size ** -0.5)  # (B, T, T)
        if mask is not None:
            pad_mask = mask.to(torch.bool).unsqueeze(1)          # (B, 1, T)
            attn = attn.masked_fill(~pad_mask, float('-1e9'))    # (B, T, T)
        attn = F.softmax(attn, dim=-1)                           # (B, T, T)
        attn = self.dropout(attn)                                # (B, T, T)
        return attn @ v                                          # (B, T, H)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, p_dropout):
        super().__init__()
        head_size = embed_dim // n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(head_size, embed_dim, p_dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, mask=None):
        x = torch.cat([head(x, mask) for head in self.attention_heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, mlp_ratio, p_dropout):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, embed_dim, p_dropout)
        self.ffwd = FeedForward(embed_dim, mlp_ratio, p_dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        x = self.ln1(x)            # (B, T, D)
        x = x + self.mha(x, mask)  # (B, T, D)
        x = self.ln2(x)            # (B, T, D)
        return x + self.ffwd(x)    # (B, T, D)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, p_dropout):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(p_dropout),
        )
    
    def forward(self, x):
        return self.ffwd(x)

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size, in_ch, patch_size, n_heads,
        embed_dim, mlp_ratio, depth, p_dropout
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, in_ch, patch_size, embed_dim, p_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, embed_dim, mlp_ratio, p_dropout) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        return x[:, 0]

class TextTransformer(nn.Module):
    def __init__(
        self, eos_token, vocab_size, embed_dim, cw_size=77,
        depth=6, n_heads=8, mlp_ratio=4, p_dropout=1.0,
    ):
        super().__init__()
        self.eos_token = eos_token
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cw_size, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, embed_dim, mlp_ratio, p_dropout) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens : torch.Tensor, mask=None):
        B, L = tokens.shape

        x = self.tok_embed(tokens)
        x = x + self.pos_embed[:, :L, :]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x) # (B, L, D)

        eos_mask = tokens.eq(self.eos_token)
        eos_mask[:, -1] = True
        eos_pos = eos_mask.int().argmax(dim=1)

        return x[torch.arange(B), eos_pos]

class CLIP(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=32, in_ch=3,
        eos_token=49_407, vocab_size=49_408, cw_size=77,
        visual_depth=6,
        textual_depth=6,
        proj_dim=512,
        embed_dim=512, n_heads=8, mlp_ratio=4, p_dropout=1.0,
    ):
        super().__init__()
        self.visual = VisionTransformer(
            img_size, in_ch, patch_size, n_heads, embed_dim,
            mlp_ratio, visual_depth, p_dropout,
        )
        self.textual = TextTransformer(
            eos_token, vocab_size, embed_dim, cw_size,
            textual_depth, n_heads, mlp_ratio, p_dropout
        )
        self.i_proj = nn.Linear(embed_dim, proj_dim)
        self.t_proj = nn.Linear(embed_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def forward(self, img, cap, mask=None):
        # img.shape (B, C, H, W)
        # cap.shape (B, T)
        i_embed = self.i_proj(self.visual(img))  # (B, proj_dim)
        t_embed = self.t_proj(self.textual(cap)) # (B, proj_dim)

        i_embed = F.normalize(i_embed)
        t_embed = F.normalize(t_embed)

        logits = i_embed @ t_embed.t() * self.logit_scale
        return logits, i_embed, t_embed


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits):
        labels = torch.arange(logits.size(0), device=logits.device)
        i2t = self.loss_fn(logits / self.temperature, labels)
        t2i = self.loss_fn(logits.t() / self.temperature, labels)
        return (i2t + t2i) / 2

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

    input = torch.zeros(10, 77, 512)
    model = MultiHeadAttention(n_heads=8, embed_dim=512, p_dropout=0.1)
    out = model(input)
    print(f"out.shape {out.shape}")

    input = torch.zeros(3, 77, 512)
    model = TransformerBlock(n_heads=8, embed_dim=512, mlp_ratio=4, p_dropout=0.1)
    out = model(input)
    print(f"out.shape {out.shape}")

    input = torch.zeros(2, 3, 224, 224)
    model = VisionTransformer(
        img_size=224, in_ch=3, n_heads=8, patch_size=32, depth=6,
        embed_dim=512, mlp_ratio=4, p_dropout=0.1
    )
    out = model(input)
    print(f"out.shape {out.shape}")

    input = torch.zeros(3, 77, dtype=torch.long)
    model = TextTransformer(eos_token=49_407, vocab_size=49408, embed_dim=512, cw_size=77)
    out = model(input)
    print(f"out.shape {out.shape}")

    B = 10
    i_input = torch.zeros(B, 3, 224, 224)
    t_input = torch.zeros(B, 77, dtype=torch.long)
    model = CLIP(
        img_size=224, patch_size=32, in_ch=3, eos_token=49_407,
        vocab_size=49_408, cw_size=77, visual_depth=6, textual_depth=6,
        proj_dim=512, embed_dim=512, n_heads=8, mlp_ratio=4, p_dropout=1.0,
    )
    logits, i_embed, t_embed = model(i_input, t_input)
    print(f"logits.shape {logits.shape}")
    print(f"i_embed.shape {i_embed.shape}")
    print(f"t_embed.shape {t_embed.shape}")
    
    loss = ContrastiveLoss(temperature=0.07)(logits)
    print(f"loss {loss}")

if __name__ == "__main__":
    main()