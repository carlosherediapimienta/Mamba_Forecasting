import torch
from torch import nn, Tensor
from zeta import SSM

class CobraBlock(nn.Module):
    def __init__(self, dim: int, dt_rank: int, dim_inner: int, d_state: int, channels: int = 64):
        super().__init__()

        self.proj = nn.Linear(dim, dim)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, dilation=1, groups=1)
        self.swish = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)
        
    def forward(self, x: Tensor):
        skip = x  # Residual connection

        # Split up the paths
        x_one = self.proj(x) 
        x_two = self.proj(x)

        # Path1
        x_one = self.conv(x_one)    
        x_one = self.swish(x_one) 
        x_one = self.ssm(x_one)

        # Path2
        x_two = self.swish(x_two)

        # Nonlinear interaction
        out = x_one * x_two
        out = self.proj(out)  
        out += skip

        return out
    
class Cobra(nn.Module):
    def __init__(self, dim: int, dt_rank: int, dim_inner: int, d_state: int, channels: int = 64, num_tokens: int = 10000, depth: int = 12, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner 
        self.d_state = d_state 
        self.channels = channels
        self.num_tokens = num_tokens 
        self.depth = depth

        # Token Embedding
        self.embed = nn.Embedding(num_tokens, dim)

        # Layers
        self.layers = nn.ModuleList([
            CobraBlock(dim, dt_rank, dim_inner, d_state, channels, *args, **kwargs) for _ in range(depth)
        ])

        # Layer Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        x = self.embed(x)
        x = self.norm(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
