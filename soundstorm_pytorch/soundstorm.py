import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
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
            self.conformer = Conformer(**conformer)
        else:
            self.conformer = conformer

        self.num_tokens_reduce = num_tokens_reduce
        self.num_tokens_per_head = default(num_tokens_per_head, num_tokens_reduce)

        dim = self.conformer.dim

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * self.num_tokens_per_head),
            Rearrange('b n (h d) -> b (n h) d', h = self.num_tokens_per_head)
        )

    def forward(
        self,
        x
    ):
        x = reduce(x, 'b (n h) d -> b n d', h = self.num_tokens_reduce)
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
