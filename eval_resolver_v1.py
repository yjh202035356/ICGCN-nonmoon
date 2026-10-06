import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry

from crss_dataset import KvasirTeacherDataset
from crss_model import FrozenSAMFeatureExtractor, DualCueHeads
from crss_resolver import PixelReliabilityResolver, soft_reliability_target


def dice_score(prob, gt, threshold=0.5, eps=1e-6):
    pred = prob >= threshold
    gt_b = gt >= 0.5
    inter = (pred & gt_b).sum().item()
    denom = pred.sum().item() + gt_b.sum().item()
    return (2.0 * inter + eps) / (denom + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--heads_checkpoint", required=True)
    ap.add_argument("--resolver_checkpoint", required=True)

    ap.add_argument("--model_type", default="vit_h",
                    choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--mid_block", type=int, default=15)
    ap.add_argument("--out_size", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--output",
        default="workdir/crss_resolver_v1/eval_metrics.json"
    )
    args = ap.parse_args()

    ds = KvasirTeacherDataset(
        args.data_root, image_size=1024, augment=False
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    ).to(args.device)
    extractor = FrozenSAMFeatureExtractor(
        sam, mid_block=args.mid_block
    ).to(args.device)

    heads = DualCueHeads(
        extractor.mid_channels,
        extractor.deep_channels
    ).to(args.device)
    hck = torch.load(
        args.heads_checkpoint, map_location="cpu"
    )
    heads.load_state_dict(hck["heads"])
    heads.eval()

    resolver = PixelReliabilityResolver(
        hidden=32
    ).to(args.device)
    rck = torch.load(
        args.resolver_checkpoint, map_location="cpu"
    )
    resolver.load_state_dict(rck["resolver"])
    resolver.eval()

    sem_dices = []
    str_dices = []
    avg_dices = []
    fused_dices = []
    oracle_dices = []
    rel_maes = []

    # Resolver decision quality only on meaningful conflicts.
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="resolver-eval"):
            image = batch["image"].to(args.device)
            gt = batch["mask"].to(args.device)

            mid, deep = extractor(image)
            sem_logits, str_logits = heads(mid, deep)

            sem_logits = F.interpolate(
                sem_logits,
                (args.out_size, args.out_size),
                mode="bilinear",
                align_corners=False
            )
            str_logits = F.interpolate(
                str_logits,
                (args.out_size, args.out_size),
                mode="bilinear",
                align_corners=False
            )
            gt_small = F.interpolate(
                gt,
                (args.out_size, args.out_size),
                mode="nearest"
            )

            p_sem = torch.sigmoid(sem_logits)
            p_str = torch.sigmoid(str_logits)

            out = resolver(p_sem, p_str)
            p_fused = out["p_fused"]
            p_avg = 0.5 * (p_sem + p_str)

            e_sem = torch.abs(p_sem - gt_small)
            e_str = torch.abs(p_str - gt_small)
            p_oracle = torch.where(
                e_sem < e_str, p_sem, p_str
            )

            r_star = soft_reliability_target(
                p_sem, p_str, gt_small, tau=args.tau
            )
            rel_maes.append(
                torch.abs(
                    out["reliability"] - r_star
                ).mean().item()
            )

            sem_dices.append(dice_score(p_sem, gt_small))
            str_dices.append(dice_score(p_str, gt_small))
            avg_dices.append(dice_score(p_avg, gt_small))
            fused_dices.append(dice_score(p_fused, gt_small))
            oracle_dices.append(dice_score(p_oracle, gt_small))

            # Evaluate selection only where predictions differ enough
            # AND one branch is meaningfully better than the other.
            conflict = torch.abs(p_sem - p_str)
            advantage = torch.abs(e_sem - e_str)
            valid = (conflict > 0.10) & (advantage > 0.05)

            gt_choose_sem = e_sem < e_str
            pred_choose_sem = out["reliability"] >= 0.5

            correct += int(
                ((gt_choose_sem == pred_choose_sem) & valid)
                .sum().item()
            )
            total += int(valid.sum().item())

    sem = float(np.mean(sem_dices))
    st = float(np.mean(str_dices))
    avg = float(np.mean(avg_dices))
    fused = float(np.mean(fused_dices))
    oracle = float(np.mean(oracle_dices))

    result = {
        "n_images": len(ds),
        "semantic_dice": sem,
        "structural_dice": st,
        "naive_average_dice": avg,
        "resolver_fused_dice": fused,
        "oracle_dice": oracle,
        "resolver_gain_vs_best_head": fused - max(sem, st),
        "resolver_gain_vs_naive": fused - avg,
        "oracle_gain_vs_resolver": oracle - fused,
        "oracle_gap_recovery_fraction": (
            (fused - avg) / (oracle - avg)
            if oracle > avg else None
        ),
        "reliability_mae": float(np.mean(rel_maes)),
        "resolver_choice_accuracy_on_conflict": (
            correct / total if total > 0 else None
        ),
        "resolver_choice_pixels": total,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
