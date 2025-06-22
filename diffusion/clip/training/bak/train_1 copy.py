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
    def __init__(self, img_size, in_ch, embed_dim, patch_size, p_dropout):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        P = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, P+1, embed_dim))
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                                              # (B, D, H/ps, W/ps), where D = embed_dim
        x = x.flatten(2).transpose(1, 2)                              # (B, P, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)   # (B, P+1, D)
        x = x + self.pos_embed
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
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = q @ k.transpose(1, 2) * (self.head_size ** -0.5)
        if mask is not None:
            pad_mask = mask.to(torch.bool).unsqueeze(dim=1)     # (B, 1, T)
            attn = attn.masked_fill(~pad_mask, float("-1e9"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)                               # (B, T, T)
        return attn @ v                                         # (B, T, H)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, p_dropout):
        super().__init__()
        head_size = (embed_dim // n_heads)
        self.heads = nn.ModuleList(
            [AttentionHead(head_size, embed_dim, p_dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x, mask=None):
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, p_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, mlp_ratio, p_dropout):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, embed_dim, p_dropout)
        self.ffwd = FeedForward(embed_dim, mlp_ratio, p_dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.ln1(x)
        x = x + self.mha(x, mask)
        x = self.ln2(x)
        return x + self.ffwd(x)


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size, in_ch, embed_dim, patch_size, p_dropout,
        n_heads, mlp_ratio, depth,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, in_ch, embed_dim, patch_size, p_dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_heads, embed_dim, mlp_ratio, p_dropout) for _ in range(depth)]
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)
        return x[:, 0]

class TextTransformer(nn.Module):
    def __init__(
        self, vocab_size=49_408, embed_dim=512,
        cw_size=77, n_heads=8, depth=6, p_dropout=0.1, mlp_ratio=4,
        eos_token=49_407,
    ):
        super().__init__()
        self.eos_token = eos_token
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cw_size, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_heads, embed_dim, mlp_ratio, p_dropout) for _ in range(depth)]
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, mask=None):
        B, L = tokens.shape
        x = self.token_embed(tokens)
        x = x + self.pos_embed[:, :L, :]
        for block in self.blocks:
            x = block(x, mask)
        x = self.layernorm(x)

        eos_mask = tokens.eq(self.eos_token)
        eos_mask[:, -1] = True
        eos_pos = eos_mask.int().argmax(dim=1)
        return x[torch.arange(B), eos_pos]


class CLIP(nn.Module):
    def __init__(
        self,
        img_size=224, in_ch=3, embed_dim=512, patch_size=32,
        n_heads=8, mlp_ratio=4, vision_depth=6,
        vocab_size=49_408, cw_size=77, text_depth=6,
        p_dropout=0.1, eos_token=49_407,
        proj_dim=512
    ):
        super().__init__()
        self.vision_transformer = VisionTransformer(
            img_size, in_ch, embed_dim, patch_size,
            p_dropout, n_heads, mlp_ratio, depth=vision_depth
        )
        self.text_transformer = TextTransformer(
            vocab_size, embed_dim, cw_size, n_heads,
            text_depth, p_dropout, mlp_ratio, eos_token
        )
        self.i_proj = nn.Linear(embed_dim, proj_dim)
        self.t_proj = nn.Linear(embed_dim, proj_dim)
        self.logit_scale = torch.ones([]) * math.log(1 / 0.07)

    def forward(self, img, cap, mask=None):
        e = self.vision_transformer(img)
        w = self.text_transformer(cap, mask)

        i_embed = F.normalize(self.i_proj(e), dim=-1)
        t_embed = F.normalize(self.t_proj(w), dim=-1)

        logits = i_embed @ t_embed.t() * self.logit_scale.exp()
        return logits, i_embed, t_embed
    
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits):
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = self.loss_fn(logits, labels)
        loss_t2i = self.loss_fn(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

class Flickr30kDataset(Dataset):
    def __init__(self, images_root : str, captions_file : str, transform=None):
        self.images_root = images_root
        self.transform = transform

        self.captions = {}
        with Path(captions_file).open('r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(None, 1)
                if parts < 2:
                    continue
                img_id, caption = parts
                fn = img_id.split('#')[0].strip().strip('",')
                self.captions.setdefault(fn, []).append(caption)
        self.filenames = sorted(self.captions.keys())
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Get filename
        Open and transform image
        Attach random caption and return image-caption pair
        """
        fn = self.filenames[idx]
        img_path = os.path.join(self.images_root, fn)
        Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        cap = random.choice(self.captions[fn])

class CLIPCollator:
    def __init__(
        self, tokenizer_name='openai/clip-vit-base-patch32', max_length=77, device='cpu'
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        """
        Tokenize captions
        """
        imgs, caps = zip(*batch)
        imgs = torch.stack(imgs, dim=0)

        tokenized = self.tokenizer(
            list(caps),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt' # return as PyTorch tensor
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        return imgs, input_ids, attention_mask

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
    model = AttentionHead(head_size=(input.size(2) // 8), embed_dim=512, p_dropout=0.1)
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
        vocab_size=49_408, cw_size=77, vision_depth=6, text_depth=6,
        proj_dim=512, embed_dim=512, n_heads=8, mlp_ratio=4, p_dropout=1.0,
    )
    logits, i_embed, t_embed = model(i_input, t_input)
    print(f"logits.shape {logits.shape}")
    print(f"i_embed.shape {i_embed.shape}")
    print(f"t_embed.shape {t_embed.shape}")
    
    loss = ContrastiveLoss()(logits)
    print(f"loss {loss}")

if __name__ == "__main__":
    main()