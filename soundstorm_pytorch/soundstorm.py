import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F

from einops import rearrange

from soundstorm_pytorch.attend import Attend

from audiolm_pytorch import SoundStream

# helpers

def exists(val):
    return val is not None

# main class

class SoundStorm(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
