import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Union, Dict

from conformer import Conformer
from soundstorm_pytorch.attend import Attend

from audiolm_pytorch import SoundStream

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# conformer with sum reduction across quantized tokens at the beginning, along with heads

class ConformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        conformer: Union[Conformer, Dict[str, any]],
        num_tokens_reduce,
        num_tokens_per_head = None,
    ):
        super().__init__()
        self.conformer = conformer

        if isinstance(conformer, dict):
            self.conformer = Conformer(**self.conformer)

        dim = self.conformer.dim

        self.mask_tokens = nn.Parameter(torch.randn(num_tokens_reduce, dim))

        self.num_tokens_reduce = num_tokens_reduce
        self.num_tokens_per_head = default(num_tokens_per_head, num_tokens_reduce)

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * self.num_tokens_per_head),
            Rearrange('b n (h d) -> b (n h) d', h = self.num_tokens_per_head)
        )

    def add_mask_tokens(
        self,
        x,
        mask
    ):
        h = self.num_tokens_reduce

        x = torch.where(
            rearrange(mask, 'b (n h) -> b n h 1', h = h),
            rearrange(x, 'b (n h) d -> b n h d', h = h),
            self.mask_tokens,
        )

        return rearrange(x, 'b n h d -> b (n h) d')

    def forward(
        self,
        x,
        cond = None
    ):
        x = reduce(x, 'b (n h) d -> b n d', h = self.num_tokens_reduce)

        if exists(cond):
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            x = x + cond

        logits = self.conformer(x)
        out = self.heads(logits)
        return out

# main class

class SoundStorm(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
