import torch, torchvision, random, math, os, wandb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import Flickr30k
from omegaconf import OmegaConf
from utils.utils import load_config
from transformers import CLIPTokenizer
from PIL import Image
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

"Emedding dim: (B, N, D), N = num_patches, D = embed size"
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_ch=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x : torch.Tensor = self.proj(x) # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2) # (B, N, embed_dim), n = num_patches
        cls_tokens : torch.Tensor = self.cls_token.expand(B, -1, -1) # (1, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
    
class AttentionHead(nn.Module):
    def __init__(
        self,
        head_size,
        n_embed,
        p_dropout
    ):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size)
        self.key   = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, mask=None):
        # x.shape = (B, T, H), where H = head_size
        q = self.query(x) # (B, T, H)
        k = self.key(x)   # (B, T, H)
        v = self.value(x) # (B, T, H)

        att = q @ k.transpose(1, 2) * (self.head_size ** -0.5) # (B, T, T)
        if mask is not None:
            # make it bool and broadcast to (B, 1, T)
            # True = real tokens.  We want to block where mask==0.
            pad_mask = mask.to(torch.bool).unsqueeze(1)  # (B, 1, T)
            # wherever pad_mask is False, we set score = -1e9
            att = att.masked_fill(~pad_mask, float('-1e9'))
        att = F.softmax(att, dim=-1) # (B, T, T)
        att = self.dropout(att)
        return att @ v # (B, T, H)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embed,
        p_dropout,
        n_heads
    ):
        super().__init__()
        head_size = n_embed // n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(head_size, n_embed, p_dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(p=p_dropout)
    
    def forward(self, x, mask=None):
        x = torch.cat([head(x, mask) for head in self.attention_heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, n_embed, p_dropout, mlp_ratio):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, mlp_ratio * n_embed),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embed, n_embed),
            nn.Dropout(p=p_dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embed, p_dropout, n_heads, mlp_ratio):
        super().__init__()
        self.mha = MultiHeadAttention(n_embed, p_dropout, n_heads)
        self.ffwd = FeedForward(n_embed, p_dropout, mlp_ratio)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=32, in_ch=3, embed_dim=512,
        depth=6, num_heads=8, p_dropout=0.1, mlp_ratio=4.0
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, p_dropout, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)        # (B, T, N)
        x = self.transformer_blocks(x) # (B, T, N)
        x = self.norm(x)               # (B, T, N)
        return x[:, 0]                 # (B, cls_token)
    
class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        cw_size=77,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        p_dropout=0.1
    ):
        super().__init__()
        self.eos_token = 4907               # From CLIP tokenizer
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cw_size, embed_dim))
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, p_dropout, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens : torch.Tensor, mask=None):
        # tokens.shape                          # (B, L)
        B, L = tokens.shape
        x = self.token_embed(tokens)            # (B, L, embed_dim)
        x = x + self.pos_embed[:, :L, :]        # (B, L, embed_dim)
        x = self.transformer_blocks(x, mask)    # (B, L, embed_dim)
        x = self.norm(x)                        # (B, L, embed_dim)
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
            img_size, patch_size, embed_dim=embed_dim, depth=vision_depth, num_heads=num_heads
        )
        self.textual = TextTransformer(
            vocab_size, cw_size, embed_dim, depth=text_depth, num_heads=num_heads
        )
        self.visual_proj = nn.Linear(embed_dim, proj_dim)
        self.textual_proj = nn.Linear(embed_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def forward(self, image, text, att_mask):
        image = self.visual_proj(self.visual(image))             # (B, embed_dim)
        text = self.textual_proj(self.textual(text, att_mask))   # (B, embed_dim)

        i_proj = F.normalize(image)                              # (B, proj_dim)
        t_proj = F.normalize(text)                               # (B, proj_dim)

        # (B, proj_dim) @ (proj_dim, B) -> (B, B) : similarity matrix
        logits = self.logit_scale.exp() * i_proj @ t_proj.t()
        return logits, i_proj, t_proj
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits):
        # logits.shape (B, B)
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_i2t = self.loss_fn(logits / self.temperature, labels)
        loss_t2i = self.loss_fn(logits.t() / self.temperature, labels)
        return (loss_i2t + loss_t2i) / 2

class Flickr30kDataset(Dataset):
    def __init__(self, images_root, captions_file, transform=None):
        self.images_root = images_root
        self.transform = transform

        # Load captions
        self.captions = {}
        with Path(captions_file).open('r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                img_id, caption = parts
                fn = img_id.split('#')[0].strip().strip('",')
                self.captions.setdefault(fn, []).append(caption)
        self.filenames = sorted(self.captions.keys())
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Get image-caption pair from the file system
        corresponding to id. Caption random
        selection of one of the appropriate captions for that image.
        """
        fn = self.filenames[idx]
        img_path = os.path.join(self.images_root, fn)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        cap = random.choice(self.captions[fn])
        return img, cap


class CLIPCollator:
    def __init__(
        self, tokenizer_name='openai/clip-vit-base-patch32', max_length=77,
        device="cpu"
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        """
        """
        imgs, caps = zip(*batch)
        imgs = torch.stack(imgs, dim=0)

        tokenized = self.tokenizer(
            list(caps),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        return imgs, input_ids, attention_mask

def get_dataset(config):
    mean = [0.444, 0.421, 0.384]
    std = [0.275, 0.267, 0.276]
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(.4,.4,.4,.1),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    base_ds = Flickr30kDataset(
        f"{config.data_root}/flickr30k/Images",
        f"{config.data_root}/flickr30k/captions.txt",
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
            f"{config.data_root}/flickr30k/captions.txt",
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
        train_ds, batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False,
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
    vocab_size = tokenizer.vocab_size #49_408

    model = CLIPModel(
        img_size=config.img_size,
        patch_size=config.patch_size,
        vocab_size=vocab_size,
        cw_size=config.cw_size,
        embed_dim=config.embed_dim,
        proj_dim=config.proj_dim,
        vision_depth=config.vision_depth,
        text_depth=config.text_depth,
        num_heads=config.num_heads,
    )
    if config.compile: model = torch.compile(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f'Epoch {epoch}/{config.n_epochs+1}')
        with tqdm(train_loader, desc="Training") as pbar:
            model.train()
            train_losses = []
            for images, input_ids, att_mas in pbar:
                images = images.to(device)
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)

                logits, _, _ = model(images, input_ids, att_mask)
                loss = ContrastiveLoss()(logits)
                optimizer.zero_grad()
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
                for images, input_ids, att_mask in pbar:
                    images = images.to(device)
                    input_ids = input_ids.to(device)
                    att_mask = att_mask.to(device)
                    logits, _, _ = model(images, input_ids, att_mask)
                    loss = ContrastiveLoss()(logits)

                    eval_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss:.4f}")
                val_epoch_loss = sum(eval_losses) / len(eval_losses)
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
    config = load_config(path="config/base.yaml", env=env)
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