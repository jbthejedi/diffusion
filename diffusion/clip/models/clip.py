import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.transformers.transformer_modules import *

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, in_ch=3, embed_dim=512, patch_size=32, p_dropout=0.1,
        depth=6, n_heads=8, mlp_ratio=4,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, in_ch, embed_dim, patch_size, p_dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_heads, embed_dim, mlp_ratio, p_dropout) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, img):
        x = self.patch_embed(img)
        x = self.blocks(x)
        x = self.layer_norm(x)
        return x[:, 0]

class TextTransformer(nn.Module):
    def __init__(
        self, vocab_size=49_408, eos_tok=49_407, embed_dim=512,
        cw_size=77, n_heads=8, mlp_ratio=4, p_dropout=0.1, depth=6,
    ):
        super().__init__()
        self.eos_tok = eos_tok
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cw_size, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_heads, embed_dim, mlp_ratio, p_dropout) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens, mask=None):
        B, T = tokens.shape
        x = self.tok_embed(tokens)
        # print(f'x.shape {x.shape}')
        # print(f'pos_embed.shape {self.pos_embed.shape}')
        x = x + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x, mask)
        x = self.layer_norm(x)      # (B, T, D)

        # We want to return the last token in the sequence
        # So we need to create a mask that can identify
        # the last token in the sequence.
        eos_mask = tokens.eq(self.eos_tok)
        eos_mask[:, -1] = True
        eos_pos = eos_mask.int().argmax(dim=-1)

        return x[torch.arange(B), eos_pos]
    

class CLIP(nn.Module):
    def __init__(
        self,
        img_size=224, in_ch=3, embed_dim=512, patch_size=32, p_dropout=0.1,
        vision_depth=6, text_depth=6, n_heads=8, mlp_ratio=4, vocab_size=49_408, eos_tok=49_407,
        cw_size=77, proj_dim=512, temperature=0.07,
    ):
        super().__init__()
        self.t_vision = VisionTransformer(
            img_size, in_ch, embed_dim, patch_size, p_dropout, vision_depth, n_heads, mlp_ratio
        )
        self.t_text = TextTransformer(
            vocab_size, eos_tok, embed_dim, cw_size, n_heads, mlp_ratio, p_dropout, text_depth
        )
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.t_proj = nn.Linear(embed_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, img, cap, mask=None):
        i_embed = self.v_proj(self.t_vision(img))
        t_embed = self.t_proj(self.t_text(cap, mask))

        i_embed = F.normalize(i_embed) # (B, proj_dim)
        t_embed = F.normalize(t_embed) # (B, proj_dim)

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


def main():
    B, T, D = 10, 7, 512
    img_size = 224
    in_ch = 3
    n_heads = 8

    input = torch.zeros(B, in_ch, img_size, img_size)
    model = VisionTransformer()
    output = model(input)
    print(f'output.shape {output.shape}')

    input = torch.zeros(B, T, dtype=torch.long)
    model = TextTransformer()
    output = model(input)
    print(f'output.shape {output.shape}')

    img = torch.zeros(B, in_ch, img_size, img_size)
    cap = torch.zeros(B, T, dtype=torch.long)
    model = CLIP(proj_dim=256)
    logits, i_embed, t_embed = model(img, cap, mask=None)
    print(f'i_embed {i_embed.shape}')
    print(f't_embed {t_embed.shape}')
    print(f'logits.shape {logits.shape}')

    loss = ContrastiveLoss()(logits)
    print(f'loss {loss}')


if __name__ == '__main__':
    main()