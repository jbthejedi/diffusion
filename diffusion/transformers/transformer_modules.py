import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, in_ch=3, embed_dim=512, patch_size=32, p_dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))              # (1, 1, D), where D = embed_dim
        P = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, P+1, embed_dim))
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x : torch.Tensor):
        B, C, H, W = x.shape
        x = self.patch_embed(x)                                     # (B, D, H/ps, W/ps), where ps = patch_size
        x = x.flatten(2).transpose(1, 2)                            # (B, P, D)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) # (B, P+1, D)
        x = x + self.pos_embed                                      # (B, P+1, D)
        return self.dropout(x)


class AttentionHead(nn.Module):
    def __init__(self, head_size, embed_dim=512, p_dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_dim, head_size)
        self.key = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x : torch.Tensor, mask=None):
        """
        mask.shape = (B, T)
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
        head_size = (embed_dim // n_heads)
        self.heads = nn.ModuleList([AttentionHead(head_size, embed_dim, p_dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x, mask=None):
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim=512, mlp_ratio=4, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(p_dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_heads=8, embed_dim=512, mlp_ratio=4, p_dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, embed_dim, p_dropout)
        self.ffwd = FeedForward(embed_dim, mlp_ratio, p_dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.ln1(x)
        x = x + self.mha(x, mask)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

def main():
    B, T, D = 10, 7, 512
    img_size = 224
    in_ch = 3
    n_heads = 8

    input = torch.zeros(B, in_ch, img_size, img_size)
    model = PatchEmbedding()
    output = model(input)
    print(f'output.shape {output.shape}')

    input = torch.zeros(B, T, D)
    model = AttentionHead(head_size=(D // n_heads), embed_dim=512, p_dropout=0.1)
    output = model(input)
    print(f'output.shape {output.shape}')

    input = torch.zeros(B, T, D)
    model = MultiHeadAttention(n_heads, embed_dim=512, p_dropout=0.1)
    output = model(input)
    print(f'output.shape {output.shape}')

    input = torch.zeros(B, T, D)
    model = FeedForward(embed_dim=512, mlp_ratio=4, p_dropout=0.1)
    output = model(input)
    print(f'output.shape {output.shape}')

    input = torch.zeros(B, T, D)
    model = TransformerBlock(n_heads=n_heads, embed_dim=512, mlp_ratio=4, p_dropout=0.1)
    output = model(input)
    print(f'output.shape {output.shape}')

if __name__ == '__main__':
    main()