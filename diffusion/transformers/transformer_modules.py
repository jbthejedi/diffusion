import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_ch=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim)) # (B, N+1, embed_dim), N = num_patches
    
    def forward(self, x):
        B, C, H, W = x.shape
        x : torch.Tensor = self.proj(x) # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2) # (B, n, embed_dim)
        cls_tokens : torch.Tensor = self.cls_token.expand(B, -1, -1) # (1, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x

class CausalAttentionHead(nn.Module):
    def __init__(
        self,
        head_size,
        n_embd,
        cw_size,
        p_dropout,
    ):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

        tril = torch.tril(torch.ones(cw_size, cw_size))
        self.register_buffer("tril", tril)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        b, t, d = x.shape

        # What information am I looking for?
        q = self.query(x) # (b, t, h), where h=head_size

        # What information do I contain?
        k = self.key(x)  # (b, t, h)

        # What information do I want to
        # communicate during aggregation of token values?
        v = self.value(x)  # (b, t, h)

        att = q @ k.transpose(1, 2) * (d**-0.5) # (b, t, t)
        att = att.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att) # (b, t, t)
        return att @ v

class AttentionHead(nn.Module):
    def __init__(
        self,
        head_size,
        n_embd,
        p_dropout,
    ):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        b, t, d = x.shape
        q = self.query(x) # (b, t, h), where h=head_size
        k = self.key(x)  # (b, t, h)
        v = self.value(x)  # (b, t, h)

        att = q @ k.transpose(1, 2) * (self.head_size**-0.5) # (b, t, t)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att) # (b, t, t)
        return att @ v

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        p_dropout,
        n_heads,
    ):
        super().__init__()
        head_size = n_embd // n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(head_size, n_embd, p_dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd, p_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p=p_dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, p_dropout, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(n_embd, p_dropout, n_heads)
        self.ffwd = FeedForward(n_embd, p_dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x






