import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crss_dataset import KvasirTeacherDataset
from psf_sam_controlled_common import (
    build_psf_sam,
    make_records,
    outputs_to_logits,
)


def binary_metrics(prob_fg, gt, threshold=0.5, eps=1e-6):
    pred = prob_fg >= threshold
    truth = gt >= 0.5

    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--repo_root", default="baselines/PSF-SAM")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds = KvasirTeacherDataset(args.data_root, image_size=1024, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_psf_sam(args.sam_checkpoint, args.repo_root, args.device)

    ckpt = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    per_image_dice = []
    per_image_iou = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="psf-eval"):
            image = batch["image"].to(args.device)
            gt = batch["mask"].to(args.device)

            outputs = model(make_records(image), multimask_output=True)
            logits = outputs_to_logits(outputs)
            prob_fg = torch.softmax(logits, dim=1)[:, 1:2]

            for i in range(image.shape[0]):
                d, j = binary_metrics(
                    prob_fg[i:i+1],
                    gt[i:i+1],
                )
                per_image_dice.append(d)
                per_image_iou.append(j)

    result = {
        "n_images": len(ds),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "foreground_dice": float(np.mean(per_image_dice)),
        "foreground_iou": float(np.mean(per_image_iou)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
