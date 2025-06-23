import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary

import transformers.transformer_modules as tm
from transformers import CLIPTokenizer

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=32, in_ch=3, embed_dim=512,
        depth=6, num_heads=8, mlp_ratio=4.0, qkv_bias=True, dropout=0.1
    ):
        super().__init__()
        self.patch_embed = tm.PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[tm.TransformerBlock(
                n_embd=embed_dim,
                p_dropout=dropout,
                n_heads=num_heads) for _ in range(depth)]
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
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cw_size, embed_dim))
        self.transformer_blocks = nn.Sequential(
            *[tm.TransformerBlock(
                n_embd=embed_dim,
                p_dropout=dropout,
                n_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens : torch.Tensor):
        # tokens.shape -> (B, L)
        B, L = tokens.shape
        print(B, L)
        x = self.token_embed(tokens) # (B, L, embed_dim)
        x = x + self.pos_embed[:, :L, :] # (B, L, embed_dim)
        x = self.transformer_blocks(x) # (B, 50, 512)
        x = self.norm(x)  # (B, 50, 512)
        # eos_id = tokenizer.eos_token_id
        # eos_mask = tokens == eos_id
        # eos_pos  = eos_mask.float().argmax(dim=-1)
        # return x[torch.arange(B), eos_pos]
        return x[0, :]


class CLIPModel(nn.Module):
    def __init__(
        self
    ):
        pass
    
    def forward(self, x):
        return x


if __name__ == "__main__":
    # dummy inputs
    # batch_size, seq_len = 4, 16
    # dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    # dummy_tokens = torch.randint(0, 49408, (batch_size, seq_len))

    # model = VisionTransformer(img_size=224)
    # input = torch.zeros(1, 3, 224, 224)
    # out = model(input)
    # print(f"out.shape {out.shape}")

    # model = TextTransformer(vocab_size=3798, cw_size=77)
    # input = torch.zeros(20, 77, dtype=torch.long)
    # out = model(input)
    # print(f"out.shape {out.shape}")

    # 1) Load the CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 2) Inspect its special tokens
    print("bos_token:", tokenizer.bos_token,       "→", tokenizer.bos_token_id)
    print("eos_token:", tokenizer.eos_token,       "→", tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token,       "→", tokenizer.pad_token_id)
    print("unk_token:", tokenizer.unk_token,       "→", tokenizer.unk_token_id)

    # model = CLIPModel()
    # logits, i_emb, t_emb = model(dummy_imgs, dummy_tokens)
    # loss = ContrastiveLoss()(logits)
    # print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")