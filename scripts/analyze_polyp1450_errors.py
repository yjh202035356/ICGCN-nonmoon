#!/usr/bin/env python3
"""Retrospective Polyp1450 error/Oracle audit using saved raw probabilities.

This is an analysis on the fixed TEST set, NOT a validation procedure.
Do not use these results to choose model, K, or checkpoint hyperparameters.
Requires only NumPy and Pillow; no GPU, PyTorch, or inference required.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DATASETS = {
    "Kvasir": 100,
    "CVC-ClinicDB": 62,
    "CVC-ColonDB": 380,
    "ETIS-LaribPolypDB": 196,
    "CVC-300": 60,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KEYS = ("semantic", "structural", "naive_average", "resolver_v2")


def indexed(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate stem {path.stem} in {folder}")
            result[path.stem] = path
    return result


def dice_score(pred, gt):
    total = int(np.count_nonzero(pred)) + int(np.count_nonzero(gt))
    if total == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(pred & gt)) / total


def iou_score(pred, gt):
    union = int(np.count_nonzero(pred | gt))
    return 1.0 if union == 0 else int(np.count_nonzero(pred & gt)) / union


def pct(n, d):
    return 100.0 * n / d if d else None


def difference_rate(base, count):
    return float(count / base) if base else None


def mask_top10(conflict):
    """Top 10% of original-resolution pixels, for descriptive audit only.

    This is NOT the same spatial selection grid used by the separate
    Kvasir conflict-interaction model.
    """
    n = conflict.size
    k = max(1, int(np.ceil(n * 0.10)))
    index = np.argpartition(conflict.reshape(-1), n - k)[n - k:]
    mask = np.zeros(n, dtype=bool)
    mask[index] = True
    return mask.reshape(conflict.shape)


def summarize(group):
    cases = group["cases"]
    if not cases:
        return {}
    n_pix = group["pixels"]
    sem_err = group["semantic_wrong_pixels"]
    str_err = group["structural_wrong_pixels"]
    res_err = group["resolver_wrong_pixels"]
    recoverable = group["semantic_wrong_structural_correct"]
    recover_by_resolver = group["recoverable_corrected_by_resolver"]
    high = group["high_conflict_pixels"]
    low = n_pix - high
    high_err = group["high_conflict_semantic_wrong"]
    low_err = sem_err - high_err
    out = {
        "n_images": cases,
        "n_pixels": n_pix,
        "mean_case_dice_at_raw_p_0_5": {
            k: group[f"sum_dice_{k}"] / cases
            for k in ("semantic", "structural", "naive_average", "resolver_v2", "oracle")
        },
        "mean_case_iou_at_raw_p_0_5": {
            k: group[f"sum_iou_{k}"] / cases
            for k in ("semantic", "structural", "naive_average", "resolver_v2", "oracle")
        },
        "pixel_error_rates_pct": {
            "semantic": pct(sem_err, n_pix),
            "structural": pct(str_err, n_pix),
            "resolver_v2": pct(res_err, n_pix),
        },
        "semantic_vs_structural_pixels": {
            "both_correct": group["both_correct"],
            "both_wrong": group["both_wrong"],
            "semantic_correct_structural_wrong": group["semantic_correct_structural_wrong"],
            "semantic_wrong_structural_correct": recoverable,
        },
        "resolver_pixel_changes_vs_semantic": {
            "recovered_wrong_to_correct": group["resolver_recovered"],
            "damaged_correct_to_wrong": group["resolver_damaged"],
            "net_pixel_gain": group["resolver_recovered"] - group["resolver_damaged"],
            "fraction_of_structural_recoverable_corrected_by_resolver": difference_rate(recoverable, recover_by_resolver),
            "recoverable_corrected_by_resolver": recover_by_resolver,
        },
        "descriptive_original_resolution_top10_conflict": {
            "selected_fraction_pct": pct(high, n_pix),
            "semantic_error_rate_in_selected_pct": pct(high_err, high),
            "semantic_error_rate_outside_selected_pct": pct(low_err, low),
            "error_enrichment_ratio": (high_err / high) / (low_err / low) if high and low and low_err else None,
            "structural_correct_semantic_wrong_in_selected": group["high_conflict_recoverable"],
            "resolver_recovered_in_selected": group["high_conflict_resolver_recovered"],
            "resolver_damaged_in_selected": group["high_conflict_resolver_damaged"],
        },
    }
    out["oracle_mean_case_dice_gain_vs_semantic"] = (
        out["mean_case_dice_at_raw_p_0_5"]["oracle"]
        - out["mean_case_dice_at_raw_p_0_5"]["semantic"]
    )
    return out


def update_counts(agg, masks, metrics):
    gt, sem, st, res, high = masks
    sw, tw, rw = (sem != gt), (st != gt), (res != gt)
    recovered = sw & ~rw
    damaged = ~sw & rw
    structural_recoverable = sw & ~tw
    agg["cases"] += 1
    agg["pixels"] += gt.size
    agg["semantic_wrong_pixels"] += int(sw.sum())
    agg["structural_wrong_pixels"] += int(tw.sum())
    agg["resolver_wrong_pixels"] += int(rw.sum())
    agg["both_correct"] += int((~sw & ~tw).sum())
    agg["both_wrong"] += int((sw & tw).sum())
    agg["semantic_correct_structural_wrong"] += int((~sw & tw).sum())
    agg["semantic_wrong_structural_correct"] += int(structural_recoverable.sum())
    agg["resolver_recovered"] += int(recovered.sum())
    agg["resolver_damaged"] += int(damaged.sum())
    agg["recoverable_corrected_by_resolver"] += int((structural_recoverable & ~rw).sum())
    agg["high_conflict_pixels"] += int(high.sum())
    agg["high_conflict_semantic_wrong"] += int((high & sw).sum())
    agg["high_conflict_recoverable"] += int((high & structural_recoverable).sum())
    agg["high_conflict_resolver_recovered"] += int((high & recovered).sum())
    agg["high_conflict_resolver_damaged"] += int((high & damaged).sum())
    for key, (d, iou) in metrics.items():
        agg[f"sum_dice_{key}"] += d
        agg[f"sum_iou_{key}"] += iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="experiments/polyp1450/seed123/predictions/raw_probabilities")
    parser.add_argument("--test-root", default="dataset/TestDataset")
    parser.add_argument("--out-dir", default="results/polyp1450_error_audit")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test first N cases per dataset; 0 means full 798")
    args = parser.parse_args()
    raw_root, test_root, out_dir = Path(args.raw_root), Path(args.test_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(float)
    per_dataset = {}
    case_rows = []
    failures = []
    for ds, expected in DATASETS.items():
        gt_idx = indexed(test_root / ds / "masks")
        npz_idx = {p.stem: p for p in (raw_root / ds).glob("*.npz")}
        if len(gt_idx) != expected or len(npz_idx) != expected or gt_idx.keys() != npz_idx.keys():
            raise ValueError(f"{ds}: expected {expected} matched GT/NPZ; actual GT={len(gt_idx)}, NPZ={len(npz_idx)}; "
                             f"unmatched GT={sorted(gt_idx.keys()-npz_idx.keys())[:4]}, "
                             f"unmatched NPZ={sorted(npz_idx.keys()-gt_idx.keys())[:4]}")
        agg = defaultdict(float)
        stems = sorted(gt_idx)
        if args.limit:
            stems = stems[:args.limit]
        for j, stem in enumerate(stems, 1):
            with Image.open(gt_idx[stem]) as im:
                gt = (np.asarray(im.convert("L")) > 127)
            with np.load(npz_idx[stem], allow_pickle=False) as raw:
                missing = set(KEYS) - set(raw.files)
                if missing:
                    raise ValueError(f"{ds}/{stem}: missing raw keys: {missing}")
                arr = {k: np.asarray(raw[k], dtype=np.float32) for k in KEYS}
            for key, p in arr.items():
                if p.shape != gt.shape or not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1:
                    raise ValueError(f"{ds}/{stem}/{key}: invalid shape, finite value or probability range")
            s, t, a, r = (arr[k] for k in KEYS)
            oracle = np.where(np.abs(s - gt) <= np.abs(t - gt), s, t)
            preds = {k: arr[k] >= 0.5 for k in KEYS}
            preds["oracle"] = oracle >= 0.5
            scores = {k: (dice_score(pred, gt), iou_score(pred, gt)) for k, pred in preds.items()}
            conflict_high = mask_top10(np.abs(s - t))
            sem, st, res = (preds[k] for k in ("semantic", "structural", "resolver_v2"))
            sw, tw, rw = (sem != gt), (st != gt), (res != gt)
            update_counts(agg, (gt, sem, st, res, conflict_high), scores)
            update_counts(totals, (gt, sem, st, res, conflict_high), scores)
            case_rows.append({
                "dataset": ds, "stem": stem,
                "dice_semantic_raw_05": scores["semantic"][0],
                "dice_structural_raw_05": scores["structural"][0],
                "dice_naive_average_raw_05": scores["naive_average"][0],
                "dice_resolver_raw_05": scores["resolver_v2"][0],
                "dice_oracle_raw_05": scores["oracle"][0],
                "delta_resolver_vs_semantic_raw_05": scores["resolver_v2"][0] - scores["semantic"][0],
                "delta_oracle_vs_semantic_raw_05": scores["oracle"][0] - scores["semantic"][0],
                "sem_wrong_struct_correct_pixels": int((sw & ~tw).sum()),
                "resolver_recovered_pixels": int((sw & ~rw).sum()),
                "resolver_damaged_pixels": int((~sw & rw).sum()),
                "top10_sem_wrong_rate_pct": pct(int((conflict_high & sw).sum()), int(conflict_high.sum())),
            })
            if j % 100 == 0 or j == len(stems):
                print(f"[{ds}] {j}/{len(stems)}", flush=True)
        per_dataset[ds] = summarize(agg)
    summary = {
        "analysis_scope": "retrospective fixed TEST set; do not tune model or K using these results",
        "protocol": "original-resolution raw sigmoid probabilities, per-image Dice/IoU @ 0.5; not mean-threshold Dice",
        "top10_note": "descriptive per-image top-10% ORIGINAL-resolution conflict, not the low-resolution selection of the separate Kvasir CRSS interaction model",
        "dataset": per_dataset,
        "pooled_pixels_and_case_mean": summarize(totals),
        "macro_dataset_mean_dice": {key: float(np.mean([per_dataset[d]["mean_case_dice_at_raw_p_0_5"][key] for d in DATASETS]))
                                    for key in ("semantic", "structural", "naive_average", "resolver_v2", "oracle")},
    }
    with (out_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Macro dataset mean raw-probability Dice@0.5 ===")
    for key, score in summary["macro_dataset_mean_dice"].items():
        print(f"  {key:14s}: {score:.6f}")
    total = summary["pooled_pixels_and_case_mean"]
    print("\n=== Pooled-pixel analysis ===")
    print(json.dumps({
        "semantic_vs_structural_pixels": total["semantic_vs_structural_pixels"],
        "resolver_pixel_changes_vs_semantic": total["resolver_pixel_changes_vs_semantic"],
        "descriptive_original_resolution_top10_conflict": total["descriptive_original_resolution_top10_conflict"],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_dir / 'summary.json'} and {out_dir / 'case_metrics.csv'}")


if __name__ == "__main__":
    main()
