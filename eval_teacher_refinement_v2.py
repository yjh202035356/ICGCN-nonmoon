import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from crss_dataset import KvasirTeacherDataset
from crss_model import FrozenSAMFeatureExtractor
from crss_teacher_refine_v2 import SAMAffinityTeacherRefiner


def soft_dice(prob, gt, eps=1e-6):
    dims = tuple(range(1, prob.ndim))

    inter = (prob * gt).sum(dim=dims)
    denom = prob.sum(dim=dims) + gt.sum(dim=dims)

    return (
        (2.0 * inter + eps)
        / (denom + eps)
    ).mean().item()


def binary_dice(prob, gt, threshold=0.5, eps=1e-6):
    pred = (prob >= threshold).float()

    dims = tuple(range(1, pred.ndim))

    inter = (pred * gt).sum(dim=dims)
    denom = pred.sum(dim=dims) + gt.sum(dim=dims)

    return (
        (2.0 * inter + eps)
        / (denom + eps)
    ).mean().item()


def binary_iou(prob, gt, threshold=0.5, eps=1e-6):
    pred = (prob >= threshold).float()

    dims = tuple(range(1, pred.ndim))

    inter = (pred * gt).sum(dim=dims)

    union = (
        pred.sum(dim=dims)
        + gt.sum(dim=dims)
        - inter
    )

    return (
        (inter + eps)
        / (union + eps)
    ).mean().item()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)

    ap.add_argument(
        "--model_type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
    )

    ap.add_argument("--mid_block", type=int, default=15)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--eval_size", type=int, default=256)

    ap.add_argument(
        "--affinity_temperature",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--teacher_weight",
        type=float,
        default=0.5,
    )

    ap.add_argument("--device", default="cuda")

    ap.add_argument(
        "--output",
        default=(
            "experiments/kvasir/seed123/"
            "teacher_refine/teacher_quality.json"
        ),
    )

    args = ap.parse_args()

    ds = KvasirTeacherDataset(
        args.data_root,
        args.image_size,
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    sam = sam_model_registry[
        args.model_type
    ](
        checkpoint=args.sam_checkpoint
    ).to(args.device)

    ext = FrozenSAMFeatureExtractor(
        sam,
        args.mid_block,
    ).to(args.device)

    refiner = SAMAffinityTeacherRefiner(
        temperature=args.affinity_temperature,
        teacher_weight=args.teacher_weight,
    ).to(args.device)

    original_soft_dice = []
    refined_soft_dice = []

    original_binary_dice = []
    refined_binary_dice = []

    original_iou = []
    refined_iou = []

    original_mae = []
    refined_mae = []

    deltas = []

    improved_soft_dice = 0
    improved_mae = 0

    with torch.no_grad():

        for b in tqdm(
            loader,
            desc="teacher-quality",
        ):
            image = b["image"].to(args.device)
            gt = b["mask"].to(args.device)
            teacher = b["teacher"].to(args.device)

            _, deep = ext(image)

            refined, _ = refiner(
                deep,
                teacher,
                output_size=(
                    args.eval_size,
                    args.eval_size,
                ),
            )

            original = F.interpolate(
                teacher,
                (
                    args.eval_size,
                    args.eval_size,
                ),
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1)

            gt_s = F.interpolate(
                gt,
                (
                    args.eval_size,
                    args.eval_size,
                ),
                mode="nearest",
            )

            o_sd = soft_dice(
                original,
                gt_s,
            )

            r_sd = soft_dice(
                refined,
                gt_s,
            )

            o_bd = binary_dice(
                original,
                gt_s,
            )

            r_bd = binary_dice(
                refined,
                gt_s,
            )

            o_iou = binary_iou(
                original,
                gt_s,
            )

            r_iou = binary_iou(
                refined,
                gt_s,
            )

            o_mae = (
                original - gt_s
            ).abs().mean().item()

            r_mae = (
                refined - gt_s
            ).abs().mean().item()

            delta = (
                refined - original
            ).abs().mean().item()

            original_soft_dice.append(o_sd)
            refined_soft_dice.append(r_sd)

            original_binary_dice.append(o_bd)
            refined_binary_dice.append(r_bd)

            original_iou.append(o_iou)
            refined_iou.append(r_iou)

            original_mae.append(o_mae)
            refined_mae.append(r_mae)

            deltas.append(delta)

            if r_sd > o_sd:
                improved_soft_dice += 1

            if r_mae < o_mae:
                improved_mae += 1

    n = len(ds)

    result = {
        "n_images": n,

        "original_soft_dice":
            float(np.mean(original_soft_dice)),

        "refined_soft_dice":
            float(np.mean(refined_soft_dice)),

        "soft_dice_gain":
            float(
                np.mean(refined_soft_dice)
                - np.mean(original_soft_dice)
            ),

        "original_binary_dice":
            float(np.mean(original_binary_dice)),

        "refined_binary_dice":
            float(np.mean(refined_binary_dice)),

        "binary_dice_gain":
            float(
                np.mean(refined_binary_dice)
                - np.mean(original_binary_dice)
            ),

        "original_iou":
            float(np.mean(original_iou)),

        "refined_iou":
            float(np.mean(refined_iou)),

        "original_mae":
            float(np.mean(original_mae)),

        "refined_mae":
            float(np.mean(refined_mae)),

        "mae_reduction":
            float(
                np.mean(original_mae)
                - np.mean(refined_mae)
            ),

        "mean_refine_delta":
            float(np.mean(deltas)),

        "soft_dice_improved_fraction":
            improved_soft_dice / n,

        "mae_improved_fraction":
            improved_mae / n,
    }

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    from pathlib import Path

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("saved:", out)


if __name__ == "__main__":
    main()