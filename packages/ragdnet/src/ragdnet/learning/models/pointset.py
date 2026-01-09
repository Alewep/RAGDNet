from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F

class PointSetEncoder(nn.Module, ABC):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError



    
class EdgeMLPEncoder(PointSetEncoder):
    def __init__(self, out_dim=128, hidden=128, eps: float = 1e-6):
        super().__init__(out_dim=out_dim)
        self.eps = float(eps)

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [L, 2] or [B, L, 2]
        """
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)  # [1, L, 2]

        x = x.float()

        # Cyclic deltas with wrap-around
        dx = torch.roll(x, shifts=-1, dims=1) - x  # [B, L, 2]

        h = self.mlp(dx)  # [B, L, H]
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        out = self.head(torch.cat([h_mean, h_max], dim=-1))

        return out.squeeze(0) if single else out



