import torch
import torch.nn as nn
from timm.models import vision_transformer

__all__ = ['VitEncoder']


class ImgEmbed(nn.Module):
    r"""
    Patch to Image Embedding.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, num_patches_w: int, num_patches_h: int) -> torch.Tensor:
        B, S, CKK = x.shape  # ckk = embed_dim
        x = self.proj(x.transpose(1, 2).reshape(
            B, CKK, num_patches_h, num_patches_w))  # b s (c k k) -> b (c k k) s -> b (c k k) sh sw -> b c h w
        return x


class VitEncoder(vision_transformer.VisionTransformer):
    r"""
    Inserts a watermark into an image using a Vision Transformer.
    """

    def __init__(self, num_bits: int, in_channels: int = 3, last_tanh: bool = True, **kwargs):
        super(VitEncoder, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.msg_linear = nn.Linear(self.embed_dim + num_bits, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=kwargs['patch_size'], in_channels=in_channels)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    # noinspection PyMethodOverriding
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:

        num_patches = int(self.patch_embed.num_patches ** 0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        m = m.unsqueeze(1)  # b 1 k
        m = m.repeat(1, x.shape[1], 1)  # b 1 k -> b l k
        for ii, blk in enumerate(self.blocks):
            x = torch.concat([x, m], dim=-1)  # b l (cpq+k)
            x = self.msg_linear(x)
            x = blk(x)

        x = x[:, 1:, :]  # without cls token
        x_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            x_w = self.tanh(x_w)

        return x_w
