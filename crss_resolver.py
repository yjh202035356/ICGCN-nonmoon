import torch
import torch.nn as nn


class PixelReliabilityResolver(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, p_sem, p_str):
        diff = torch.abs(p_sem - p_str)
        x = torch.cat([p_sem, p_str, diff], dim=1)
        reliability_logit = self.net(x)
        reliability = torch.sigmoid(reliability_logit)
        p_fused = reliability * p_sem + (1.0 - reliability) * p_str
        return {
            "reliability_logit": reliability_logit,
            "reliability": reliability,
            "p_fused": p_fused,
            "conflict": diff,
        }


def soft_reliability_target(p_sem, p_str, gt, tau=0.10):
    e_sem = torch.abs(p_sem - gt)
    e_str = torch.abs(p_str - gt)
    return torch.sigmoid((e_str - e_sem) / tau).detach()
