import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------
# Patch-based Vision Transformer
# ----------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]
        x = self.proj(x)              # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)           # [B, N+1, embed_dim]
        x = x + self.pos_embed
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
        img_size=224, patch_size=32, in_chans=3, embed_dim=512,
        depth=6, num_heads=8, mlp_ratio=4.0, qkv_bias=True, dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)           # [B, N+1, embed_dim]
        x = self.encoder(x)               # [B, N+1, embed_dim]
        x = self.norm(x)
        return x[:, 0]  # return cls token

# ----------------------
# Text Transformer
# ----------------------
class TextTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len=77,
                 embed_dim=512,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        # tokens: [B, L]
        x = self.token_embed(tokens)      # [B, L, embed_dim]
        x = x + self.pos_embed[:, : tokens.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        # return embedding of the end-of-text token
        return x[torch.arange(x.size(0)), tokens.argmax(dim=-1)]

# ----------------------
# CLIP Model
# ----------------------
class CLIPModel(nn.Module):
    def __init__(self,
        img_size=224, patch_size=32, vocab_size=49408, max_len=77,
        embed_dim=512, proj_dim=512, vision_depth=6, text_depth=6, num_heads=8,
    ):
        super().__init__()
        # encoders
        self.visual = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=num_heads,
            )
        self.textual = TextTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            depth=text_depth,
            num_heads=num_heads,
        )
        # projection layers
        self.visual_proj = nn.Linear(embed_dim, proj_dim)
        self.textual_proj = nn.Linear(embed_dim, proj_dim)
        # learnable logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, image, text):
        # image: [B, 3, H, W], text: [B, L]
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

# ----------------------
# Contrastive Loss
# ----------------------
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

# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    # dummy inputs
    batch_size, seq_len = 4, 16
    dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    dummy_tokens = torch.randint(0, 49408, (batch_size, seq_len))

    model = CLIPModel()
    logits, i_emb, t_emb = model(dummy_imgs, dummy_tokens)
    loss = ContrastiveLoss()(logits)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
