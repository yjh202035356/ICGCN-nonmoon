import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAwareReliabilityResolver(nn.Module):
    """
    CRSS Resolver v2.

    Resolver operates at SAM feature resolution (typically 64x64).

    Inputs:
      - downsampled P_sem
      - downsampled P_str
      - |P_sem - P_str|
      - projected SAM mid feature
      - projected SAM deep feature

    Output:
      - low-res reliability r_low
      - reliability upsampled to the segmentation resolution
      - fused probability
    """
    def __init__(self, mid_channels=1280, deep_channels=256,
                 proj_channels=16, hidden=64):
        super().__init__()

        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, proj_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, proj_channels),
            nn.GELU(),
        )

        self.deep_proj = nn.Sequential(
            nn.Conv2d(deep_channels, proj_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, proj_channels),
            nn.GELU(),
        )

        in_channels = 3 + proj_channels * 2

        self.resolver = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),

            nn.Conv2d(hidden, 1, kernel_size=1)
        )

    def forward(self, p_sem, p_str, mid_feat, deep_feat):
        target_hw = mid_feat.shape[-2:]

        p_sem_low = F.interpolate(
            p_sem, size=target_hw, mode="bilinear", align_corners=False
        )
        p_str_low = F.interpolate(
            p_str, size=target_hw, mode="bilinear", align_corners=False
        )
        conflict_low = torch.abs(p_sem_low - p_str_low)

        mid_p = self.mid_proj(mid_feat)
        deep_p = self.deep_proj(deep_feat)

        x = torch.cat(
            [p_sem_low, p_str_low, conflict_low, mid_p, deep_p],
            dim=1
        )

        reliability_logit_low = self.resolver(x)
        reliability_low = torch.sigmoid(reliability_logit_low)

        reliability = F.interpolate(
            reliability_low,
            size=p_sem.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        p_fused = reliability * p_sem + (1.0 - reliability) * p_str

        return {
            "reliability_logit_low": reliability_logit_low,
            "reliability_low": reliability_low,
            "reliability": reliability,
            "p_fused": p_fused,
            "conflict_low": conflict_low,
        }


def soft_reliability_target_low(
    p_sem, p_str, gt, target_hw, tau=0.10
):
    p_sem_low = F.interpolate(
        p_sem, size=target_hw, mode="bilinear", align_corners=False
    )
    p_str_low = F.interpolate(
        p_str, size=target_hw, mode="bilinear", align_corners=False
    )
    gt_low = F.interpolate(
        gt, size=target_hw, mode="nearest"
    )

    e_sem = torch.abs(p_sem_low - gt_low)
    e_str = torch.abs(p_str_low - gt_low)

    target = torch.sigmoid((e_str - e_sem) / tau)
    return target.detach(), gt_low
