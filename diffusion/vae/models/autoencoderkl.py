import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups):
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act2 = nn.SiLU(inplace=True)
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.skip(x)

class DownEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)
        # Downsampling
        self.convdown = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.convdown(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_groups):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)

        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, H*W).permute(0, 2, 1)                # (B, HW, C)
        k = self.k(x).reshape(B, C, H*W)                                 # (B, C, HW)
        v = self.v(x).reshape(B, C, H*W).permute(0, 2, 1)                # (B, HW, C)

        attn = (q @ k) * (C ** -0.5)                                     # (B, HW, HW)
        attn = torch.softmax(attn, dim=-1)
        h_attn = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W) # (B, C, H, W)
        return h_attn + self.proj(x)



class MidBlock(nn.Module):
    def __init__(self, in_ch, num_groups):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, in_ch, num_groups)
        self.att = AttentionBlock(in_ch, num_groups)
        self.res2 = ResnetBlock(in_ch, in_ch, num_groups)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.att(x)
        x = self.res2(x)
        return x


class LatentBottleneck(nn.Module):
    def __init__(self, in_ch, latent_ch):
        super().__init__()
        self.mu_conv = nn.Conv2d(in_ch, latent_ch, 1)
        self.logvar_conv = nn.Conv2d(in_ch, latent_ch, 1)
    
    def forward(self, x):
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    
class Encoder(nn.Module):
    def __init__(self, in_ch, latent_ch, base_ch, num_groups):
        super().__init__()
        self.downblocks = nn.ModuleList([
            DownEncoderBlock(in_ch, base_ch[0], num_groups),      # (,128, 256, 256)
            DownEncoderBlock(base_ch[0], base_ch[1], num_groups), # (,256, 128, 128)
            DownEncoderBlock(base_ch[1], base_ch[2], num_groups), # (,512, 64, 64)
        ])
        self.mid = MidBlock(base_ch[2], num_groups)
        self.lb = LatentBottleneck(base_ch[2], latent_ch)

    def forward(self, x):
        for block in self.downblocks:
            x = block(x)
        x = self.mid(x)
        z, mu, logvar = self.lb(x)
        return z, mu, logvar
    

class UpDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)
    
    def forward(self, x):
        x = self.up(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_ch, base_ch, out_ch, num_groups):
        super().__init__()
        self.upblocks = nn.ModuleList([
            UpDecoderBlock(latent_ch, base_ch[-1], num_groups),
            UpDecoderBlock(base_ch[-1], base_ch[-2], num_groups),
            UpDecoderBlock(base_ch[-2], base_ch[-3], num_groups),
        ])
        self.conv_head = nn.Conv2d(base_ch[-3], out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        for block in self.upblocks:
            x = block(x)
        return self.conv_head(x)

class AutoencoderKLSmall(nn.Module):
    def __init__(
        self,
        in_ch=3,
        latent_ch=4,
        base_ch=(128,256,512),
        num_groups=32,
    ):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_ch, base_ch, num_groups)
        self.decoder = Decoder(latent_ch, base_ch, in_ch, num_groups)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = torch.sigmoid(x_hat)
        return x_hat, mu, logvar


if __name__ == "__main__":
    model = AutoencoderKLSmall(
        in_ch=3,
        base_ch=(128, 256, 512),
        latent_ch=4,
        num_groups=32
    )
    input = torch.zeros(1, 3, 512, 512)
    x_hat, mu, logvar = model(input)

    print(f"input.shape {input.shape}")
    print(f"out.shape {x_hat.shape}")
    print(f"mu.shape {mu.shape}")
    print(f"logvar.shape {logvar.shape}")