import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusion.transformers.transformer_modules as tm


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=32, in_ch=3, embed_dim=512,
        depth=6, num_heads=8, mlp_ratio=4.0, qkv_bias=True, dropout=0.1,
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
        """
        img_size, patch_size,  vocab_size, cw_size
        embed_dim, proj_dim, vision_depth, text_depth, num_heads
        """
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
        logits = self.logit_scale.exp() * i_proj @ t_proj.t()
        return logits, i_proj, t_proj
    

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits):
        # logits: [B, B]
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_i2t = self.loss_fn(logits, labels)
        loss_t2i = self.loss_fn(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2