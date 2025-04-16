import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=64, patch_size=16, emb_dim=128, img_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, emb_dim))

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0.1, ff_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_c=64, img_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, base_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_c, base_c, 3, padding=1), nn.ReLU()
        )
        self.patch_embed = PatchEmbedding(in_channels=base_c, patch_size=16, emb_dim=128, img_size=img_size)
        self.transformer = nn.Sequential(
            TransformerBlock(128), TransformerBlock(128)
        )
        self.img_size = img_size
        self.emb_dim = 128

    def forward(self, x):
        feat = self.cnn(x)
        tokens = self.patch_embed(feat)
        tokens = self.transformer(tokens)
        B, N, C = tokens.shape
        H = W = self.img_size // 16
        x = tokens.transpose(1, 2).view(B, C, H, W)
        return x  # [B, 128, 8, 8]


class Decoder(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        return x


class ViTUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=128):
        super().__init__()
        self.encoder = Encoder(in_channels, img_size=img_size)
        self.decoder = Decoder()
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)