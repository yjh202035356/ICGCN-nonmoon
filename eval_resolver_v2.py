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
from crss_resolver_v2 import FeatureAwareReliabilityResolver


def dice_score(prob, gt, threshold=0.5, eps=1e-6):
    pred = prob >= threshold
    gt_b = gt >= 0.5
    inter = (pred & gt_b).sum().item()
    denom = pred.sum().item() + gt_b.sum().item()
    return (2.0 * inter + eps) / (denom + eps)


def boundary_band(gt, width=3):
    k = 2 * width + 1
    dil = F.max_pool2d(gt, kernel_size=k, stride=1, padding=width)
    ero = -F.max_pool2d(-gt, kernel_size=k, stride=1, padding=width)
    return (dil - ero) > 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--heads_checkpoint", required=True)
    ap.add_argument("--resolver_checkpoint", required=True)
    ap.add_argument("--model_type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--mid_block", type=int, default=15)
    ap.add_argument("--out_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds = KvasirTeacherDataset(args.data_root, image_size=1024, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    ).to(args.device)
    ext = FrozenSAMFeatureExtractor(sam, mid_block=args.mid_block).to(args.device)

    heads = DualCueHeads(ext.mid_channels, ext.deep_channels).to(args.device)
    hck = torch.load(args.heads_checkpoint, map_location="cpu")
    heads.load_state_dict(hck["heads"])
    heads.eval()

    rck = torch.load(args.resolver_checkpoint, map_location="cpu")
    resolver = FeatureAwareReliabilityResolver(
        mid_channels=ext.mid_channels,
        deep_channels=ext.deep_channels,
        proj_channels=16,
        hidden=64,
    ).to(args.device)
    resolver.load_state_dict(rck["resolver"])
    resolver.eval()

    sems, strs, avgs, fuseds, oracles = [], [], [], [], []
    fg_correct = fg_total = 0
    bg_correct = bg_total = 0
    bnd_correct = bnd_total = 0
    all_correct = all_total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="resolver-v2-eval"):
            image = batch["image"].to(args.device)
            gt = batch["mask"].to(args.device)

            mid, deep = ext(image)
            sem_l, str_l = heads(mid, deep)

            sem_l = F.interpolate(sem_l, (args.out_size, args.out_size),
                                  mode="bilinear", align_corners=False)
            str_l = F.interpolate(str_l, (args.out_size, args.out_size),
                                  mode="bilinear", align_corners=False)
            gt_s = F.interpolate(gt, (args.out_size, args.out_size), mode="nearest")

            p_sem = torch.sigmoid(sem_l)
            p_str = torch.sigmoid(str_l)
            p_avg = 0.5 * (p_sem + p_str)

            out = resolver(p_sem, p_str, mid, deep)
            p_fused = out["p_fused"]

            e_sem = torch.abs(p_sem - gt_s)
            e_str = torch.abs(p_str - gt_s)
            p_oracle = torch.where(e_sem < e_str, p_sem, p_str)

            sems.append(dice_score(p_sem, gt_s))
            strs.append(dice_score(p_str, gt_s))
            avgs.append(dice_score(p_avg, gt_s))
            fuseds.append(dice_score(p_fused, gt_s))
            oracles.append(dice_score(p_oracle, gt_s))

            conflict = torch.abs(p_sem - p_str)
            advantage = torch.abs(e_sem - e_str)
            valid = (conflict > 0.10) & (advantage > 0.05)

            oracle_sem = e_sem < e_str
            pred_sem = out["reliability"] >= 0.5
            correct = oracle_sem == pred_sem

            fg = gt_s >= 0.5
            bg = ~fg
            bnd = boundary_band(gt_s, width=3)

            all_correct += int((correct & valid).sum().item())
            all_total += int(valid.sum().item())

            v = valid & fg
            fg_correct += int((correct & v).sum().item())
            fg_total += int(v.sum().item())

            v = valid & bg
            bg_correct += int((correct & v).sum().item())
            bg_total += int(v.sum().item())

            v = valid & bnd
            bnd_correct += int((correct & v).sum().item())
            bnd_total += int(v.sum().item())

    sem = float(np.mean(sems))
    st = float(np.mean(strs))
    avg = float(np.mean(avgs))
    fused = float(np.mean(fuseds))
    oracle = float(np.mean(oracles))

    result = {
        "n_images": len(ds),
        "checkpoint_epoch": int(rck.get("epoch", -1)),
        "semantic_dice": sem,
        "structural_dice": st,
        "naive_average_dice": avg,
        "resolver_v2_fused_dice": fused,
        "oracle_dice": oracle,
        "resolver_gain_vs_best_head": fused - max(sem, st),
        "resolver_gain_vs_naive": fused - avg,
        "oracle_gain_vs_resolver": oracle - fused,
        "oracle_gap_recovery_fraction_from_naive": (
            (fused - avg) / (oracle - avg) if oracle > avg else None
        ),
        "overall_choice_accuracy": all_correct / all_total if all_total else None,
        "foreground_choice_accuracy": fg_correct / fg_total if fg_total else None,
        "background_choice_accuracy": bg_correct / bg_total if bg_total else None,
        "boundary_choice_accuracy": bnd_correct / bnd_total if bnd_total else None,
        "choice_pixels": all_total,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
