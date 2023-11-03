from __future__ import annotations

from typing import Literal, TypeAlias, TypeVar

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .base import ERM_X, ERMModel

__all__ = ['ViT']

# helpers

T = TypeVar('T', int, float)
MaybePair: TypeAlias = T | tuple[T, T]

def pair(t: MaybePair[T]) -> tuple[T, T]:
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout_rate)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_rate)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout_rate: float = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout_rate),
                FeedForward(dim, mlp_dim, dropout_rate = dropout_rate)
            ]))

    def forward(self, x: torch.Tensor):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module, ERMModel):
    """
    Represents a ViT (Vision Transformer) [1]_ classifier for images.

    Parameters
    ----------
    num_classes : int
        The number of unique classes that can be outputted by the model.

    References
    ----------
    .. [1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
       Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
       Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021.
       An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
       In *ICLR 2021*. <https://doi.org/10.48550/arXiv.2010.11929>
    """

    def __init__(
        self,
        *,
        image_size: MaybePair[int],
        patch_size: MaybePair[int],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: Literal['cls', 'mean'] = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout_rate: float = 0.,
        emb_dropout_rate: float = 0.,
    ):
        super().__init__()

        self._image_size = image_size
        self._patch_size = patch_size
        self._num_classes = num_classes
        self._dim = dim
        self._depth = depth
        self._heads = heads
        self._mlp_dim = mlp_dim
        self._pool = pool
        self._channels = channels
        self._dim_head = dim_head
        self._dropout_rate = dropout_rate
        self._emb_dropout_rate = emb_dropout_rate

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout_rate)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    @property
    def image_size(self) -> MaybePair[int]:
        return self._image_size

    @property
    def patch_size(self) -> MaybePair[int]:
        return self._patch_size

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def heads(self) -> int:
        return self._heads

    @property
    def mlp_dim(self) -> int:
        return self._mlp_dim

    @property
    def pool(self) -> str:
        return self._pool

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dim_head(self) -> int:
        return self._dim_head

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def emb_dropout_rate(self) -> float:
        return self._emb_dropout_rate

    def get_num_classes(self) -> int:
        return self.num_classes

    def forward(self, img: ERM_X):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
