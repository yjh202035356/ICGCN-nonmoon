import argparse
import torch
import torch.nn.functional as F

from crss_dataset import KvasirTeacherDataset
from psf_sam_controlled_common import (
    build_psf_sam,
    count_parameters,
    make_records,
    outputs_to_logits,
)


def soft_dice_loss(prob_fg, gt, eps=1e-6):
    inter = (prob_fg * gt).sum(dim=(1,2,3))
    denom = prob_fg.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/Kvasir/train")
    ap.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--repo_root", default="baselines/PSF-SAM")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = KvasirTeacherDataset(args.data_root, image_size=1024, augment=False)
    sample = ds[0]

    model = build_psf_sam(args.sam_checkpoint, args.repo_root, args.device)
    model.train()

    image = sample["image"].unsqueeze(0).to(args.device)
    gt = sample["mask"].unsqueeze(0).to(args.device)

    scaler = torch.cuda.amp.GradScaler()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(make_records(image), multimask_output=True)
        logits = outputs_to_logits(outputs)

        target = gt[:, 0].long()
        ce = F.cross_entropy(logits, target)

        prob_fg = torch.softmax(logits, dim=1)[:, 1:2]
        dice = soft_dice_loss(prob_fg, gt)
        loss = ce + dice

    scaler.scale(loss).backward()
    total, trainable = count_parameters(model)

    grad_tensors = 0
    grad_params = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grad_tensors += 1
            grad_params += p.numel()

    print("sample:", sample["name"])
    print("image:", tuple(image.shape), float(image.min()), float(image.max()))
    print("gt:", tuple(gt.shape), float(gt.min()), float(gt.max()))
    print("logits:", tuple(logits.shape))
    print("prob_fg:", tuple(prob_fg.shape), float(prob_fg.min()), float(prob_fg.max()))
    print("loss:", float(loss.detach()))
    print("total parameters:", total)
    print("trainable parameters:", trainable)
    print("trainable ratio:", trainable / total)
    print("trainable tensors with grad:", grad_tensors)
    print("trainable parameters with grad:", grad_params)

    assert logits.shape[1] == 2
    assert torch.isfinite(loss)
    assert grad_tensors > 0

    print("PSF-SAM CONTROLLED SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
