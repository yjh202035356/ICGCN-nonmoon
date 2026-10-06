import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenSAMFeatureExtractor(nn.Module):
    def __init__(self, sam, mid_block=15):
        super().__init__()
        self.sam = sam
        self.mid_block = mid_block
        self._mid = None

        for p in self.sam.parameters():
            p.requires_grad = False
        self.sam.eval()

        n_blocks = len(self.sam.image_encoder.blocks)
        if not 0 <= mid_block < n_blocks:
            raise ValueError(f"mid_block must be in [0,{n_blocks-1}]")

        self.sam.image_encoder.blocks[mid_block].register_forward_hook(self._save_mid)
        self.mid_channels = int(self.sam.image_encoder.blocks[0].norm1.normalized_shape[0])
        self.deep_channels = int(self.sam.image_encoder.neck[0].out_channels)

    def _save_mid(self, module, inputs, output):
        self._mid = output.detach()

    @torch.no_grad()
    def forward(self, image_0_255):
        x = self.sam.preprocess(image_0_255)
        self._mid = None
        deep = self.sam.image_encoder(x)
        if self._mid is None:
            raise RuntimeError("Mid-level hook did not fire.")
        mid = self._mid.permute(0,3,1,2).contiguous()
        return mid, deep.detach()

class SmallSegHead(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
    def forward(self, x):
        return self.net(x)

class DualCueHeads(nn.Module):
    def __init__(self, mid_channels, deep_channels, hidden=128):
        super().__init__()
        self.semantic = SmallSegHead(deep_channels, hidden)
        self.structural = SmallSegHead(mid_channels, hidden)

    def forward(self, mid_feat, deep_feat):
        return self.semantic(deep_feat), self.structural(mid_feat)

def soft_dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0*inter + eps)/(denom + eps)).mean()

def seg_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)

def soft_boundary_map(x):
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    return (dil - ero).clamp(0,1)

def structural_boundary_loss(logits, target):
    pred_b = soft_boundary_map(torch.sigmoid(logits))
    gt_b = soft_boundary_map(target)
    return F.binary_cross_entropy(pred_b.clamp(1e-5, 1-1e-5), gt_b)
