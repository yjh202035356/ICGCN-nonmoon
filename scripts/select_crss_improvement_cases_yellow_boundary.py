#!/usr/bin/env python3
"""Select real Kvasir-SEG cases where CRSS-SAM exceeds BOTH baselines.

Keep the existing ranking/filtering/audit; render Figure-1 style:
   Rows:    Simple-ViLMedSAM | PSF-SAM | CRSS-SAM (Ours)
   Columns: Input | Ground Truth | Output Mask | Output (yellow prediction boundary only)

All three rows show the SAME input image and GT. No Dice annotations in PNG.
No GPU is required: use actual stored prediction masks/probability maps.

Example:
python scripts/select_crss_improvement_cases.py --min-baseline-dice 0.80 \\
    --min-crss-dice 0.89 --min-gain 0.02 --top 3 --render
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PRED_EXT = IMAGE_EXT | {".npy", ".npz"}
MODEL_NAMES = ("Simple-ViLMedSAM", "PSF-SAM", "CSRS-SAM (Ours)")


def index_files(folder: str, extensions: set[str], label: str) -> dict[str, Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"{label}: folder not found: {root}")
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    result: dict[str, Path] = {}
    for path in files:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate case '{path.stem}' in {label}: "
                             f"{result[path.stem]} and {path}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"{label}: no supported images/masks in {root}")
    return result


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_gt(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != image_hw:
        raise ValueError(f"Input and GT sizes disagree for {path}: {image_hw} vs {mask.shape}")
    return mask > (0 if mask.max() <= 1 else 127)


def read_prediction(path: Path, key: str | None, kind: str) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if key is None:
                if len(arrays.files) != 1:
                    raise ValueError(f"{path} has keys {arrays.files}; supply --MODEL-key")
                key = arrays.files[0]
            if key not in arrays.files:
                raise ValueError(f"{path}: invalid key {key!r}; available: {arrays.files}")
            arr = np.asarray(arrays[key])
    elif path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"))
    arr = np.squeeze(arr)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError(f"{path}: expected finite 2D prediction; found {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    if kind == "binary":
        return arr > 0
    if kind == "logits":
        return arr >= 0  # sigmoid(logit) >= 0.5
    if kind == "probability":
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"{path}: values outside [0,1] in probability mode")
        return arr >= 0.5
    # Auto matches the existing 256x256 PNG-mask interpretation.
    if path.suffix.lower() in IMAGE_EXT:
        return (arr > 0) if arr.max() <= 1 else (arr >= 128)
    if arr.min() >= 0 and arr.max() <= 1:
        return arr >= 0.5
    if set(np.unique(arr)).issubset({0., 255.}):
        return arr >= 128
    raise ValueError(f"{path}: cannot infer mask semantics; use --MODEL-kind logits")


def pad_to_256(arr: np.ndarray, is_mask: bool) -> np.ndarray:
    """Resize longest side to 256, then pad at bottom and right (Kvasir protocol)."""
    h, w = arr.shape[:2]
    nh, nw = round(h * 256 / max(h, w)), round(w * 256 / max(h, w))
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), resampling))
    target = np.zeros((256, 256) + arr.shape[2:], dtype=np.uint8)
    target[:nh, :nw] = resized
    return target


def align_sources(rgb: np.ndarray, gt: np.ndarray,
                  preds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], str]:
    original_hw = rgb.shape[:2]
    pred_shapes = [p.shape for p in preds]
    if all(s == original_hw for s in pred_shapes):
        return rgb, gt, preds, "original-size"
    if all(s == original_hw or s == (256, 256) for s in pred_shapes):
        image_canvas = pad_to_256(rgb, is_mask=False)
        gt_canvas = pad_to_256(gt.astype(np.uint8) * 255, is_mask=True) > 127
        pred_canvas = [
            pad_to_256(p.astype(np.uint8) * 255, is_mask=True) > 127
            if p.shape == original_hw else p
            for p in preds
        ]
        return image_canvas, gt_canvas, pred_canvas, "256x256 padded"
    raise ValueError(f"Prediction sizes {pred_shapes} do not match image {original_hw} "
                     "or 256x256 padded masks")


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = int(pred.sum()) + int(gt.sum())
    return float(2 * np.logical_and(pred, gt).sum() / denom) if denom else 1.0


def mask_edge(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    eroded = image
    for _ in range(max(1, thickness)):
        eroded = eroded.filter(ImageFilter.MinFilter(size=3))
    return mask & ~(np.asarray(eroded) > 127)


def overlay(rgb: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Draw ONLY the predicted mask's boundary in yellow on the input image."""
    result = rgb.copy()
    pred_edge = mask_edge(pred)
    result[pred_edge] = (255, 255, 0)  # yellow: prediction boundary only
    return result


def render_figure1_style(stem: str, rgb: np.ndarray, gt: np.ndarray,
                         preds: list[np.ndarray], out_file: Path, dpi: int) -> None:
    """Render 3 rows (methods) x 4 columns (Input / GT / Mask / Overlay)."""
    fig, axes = plt.subplots(3, 4, figsize=(14.2, 9.2))
    fig.patch.set_facecolor("white")
    for j, title in enumerate(("Input", "Ground Truth", "Output Mask", "Output")):
        axes[0, j].set_title(title, fontsize=16, fontweight="bold", pad=17)

    for i, (model, pred) in enumerate(zip(MODEL_NAMES, preds)):
        panels = (rgb, gt, pred, overlay(rgb, pred))
        for j, panel in enumerate(panels):
            ax = axes[i, j]
            if j in (1, 2):
                ax.imshow(panel.astype(np.uint8), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
            else:
                ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].text(-0.14, 0.5, model, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=14, fontweight="bold")

    fig.subplots_adjust(left=0.115, right=0.99, top=0.915, bottom=0.075,
                        wspace=0.075, hspace=0.18)
    fig.text(0.55, 0.025,
             "Output: yellow = predicted mask boundary only",
             ha="center", fontsize=11, color="#555555")
    # No Dice text inside the poster figure.
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out_file}")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", default="data/Kvasir/test/images")
    p.add_argument("--gts", default="data/Kvasir/test/labels")
    p.add_argument("--simple", default="qualitative_outputs/simple_vilmedsam")
    p.add_argument("--psf", default="qualitative_outputs/psf_sam")
    p.add_argument("--crss", default="qualitative_outputs/crss")
    p.add_argument("--out", default="qualitative_outputs/crss_improvement_cases")
    p.add_argument("--min-crss-dice", type=float, default=0.89)
    p.add_argument("--min-baseline-dice", type=float, default=0.80)
    p.add_argument("--min-gain", type=float, default=0.02,
                   help="CRSS Dice minus the STRONGER of the two baselines")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--render", action="store_true", help="Save 3x4 Figure-1-style PNGs")
    p.add_argument("--dpi", type=int, default=240)
    for model in ("simple", "psf", "crss"):
        p.add_argument(f"--{model}-key")
        p.add_argument(f"--{model}-kind", default="auto",
                       choices=("auto", "probability", "logits", "binary"))
    return p.parse_args()


def main() -> None:
    args = cli()
    if not (0 <= args.min_crss_dice <= 1 and 0 <= args.min_baseline_dice <= 1
            and -1 <= args.min_gain <= 1):
        raise ValueError("Dice threshold or gain outside valid range")
    if args.top < 1 or args.dpi < 1:
        raise ValueError("--top and --dpi must be positive")

    folders = (args.images, args.gts, args.simple, args.psf, args.crss)
    labels = ("Input", "Ground Truth") + MODEL_NAMES
    indices = [index_files(path, IMAGE_EXT if i < 2 else PRED_EXT, name)
               for i, (path, name) in enumerate(zip(folders, labels))]
    shared = sorted(set.intersection(*(set(idx) for idx in indices)))
    if not shared:
        raise RuntimeError("No common case IDs across input, GT, and three models")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for stem in shared:
        img = read_rgb(indices[0][stem])
        gt_native = read_gt(indices[1][stem], img.shape[:2])
        pred_native = [read_prediction(indices[i][stem], getattr(args, f"{model}_key"),
                                       getattr(args, f"{model}_kind"))
                       for i, model in ((2, "simple"), (3, "psf"), (4, "crss"))]
        _, gt, (simple, psf, crss), canvas = align_sources(img, gt_native, pred_native)
        d_simple, d_psf, d_crss = [dice(p, gt) for p in (simple, psf, crss)]
        best = simple if d_simple >= d_psf else psf
        repaired = int(((best != gt) & (crss == gt)).sum())
        damaged = int(((best == gt) & (crss != gt)).sum())
        recovered_fn = int((gt & ~best & crss).sum())
        reduced_fp = int((~gt & best & ~crss).sum())
        rows.append({
            "case": stem,
            "simple_dice": round(d_simple, 7),
            "psf_dice": round(d_psf, 7),
            "crss_dice": round(d_crss, 7),
            "gain_over_simple": round(d_crss - d_simple, 7),
            "gain_over_psf": round(d_crss - d_psf, 7),
            "gain_over_best_baseline": round(d_crss - max(d_simple, d_psf), 7),
            "repaired_pixels_vs_best": repaired,
            "damaged_pixels_vs_best": damaged,
            "recovered_fn_pixels_vs_best": recovered_fn,
            "removed_fp_pixels_vs_best": reduced_fp,
            "canvas": canvas,
            "meets_thresholds": (d_crss >= args.min_crss_dice
                                 and min(d_simple, d_psf) >= args.min_baseline_dice
                                 and d_crss - max(d_simple, d_psf) >= args.min_gain),
        })

    rows.sort(key=lambda r: (-r["gain_over_best_baseline"], -r["crss_dice"], r["case"]))
    csv_path = output / "all_shared_case_comparisons.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    wins = sum(r["gain_over_best_baseline"] > 1e-8 for r in rows)
    ties = sum(abs(r["gain_over_best_baseline"]) <= 1e-8 for r in rows)
    qualified = [r for r in rows if r["meets_thresholds"]]
    print(f"Shared cases: {len(rows)} | CRSS better than BOTH: {wins} | "
          f"tied-best: {ties} | not better: {len(rows) - wins - ties}")
    print(f"Cases meeting thresholds: {len(qualified)} "
          f"(both baselines >= {args.min_baseline_dice:.2f}, "
          f"CRSS >= {args.min_crss_dice:.2f}, "
          f"gain over stronger baseline >= {args.min_gain:.2f})")
    print(f"Saved CSV: {csv_path}")
    print("Top 10 by advantage over the stronger baseline:")
    for row in rows[:10]:
        print(f" {row['case']}: Simple {row['simple_dice']:.4f}, "
              f"PSF {row['psf_dice']:.4f}, CRSS {row['crss_dice']:.4f}, "
              f"gain {row['gain_over_best_baseline']:+.4f}"
              f"{' [passes]' if row['meets_thresholds'] else ''}")

    if not qualified:
        print("No cases meet all filters. No figure generated; consider lowering thresholds.")
        return
    if not args.render:
        print("To generate 3x4 PNG figures, pass --render")
        return

    for row in qualified[:args.top]:
        stem = row["case"]
        image = read_rgb(indices[0][stem])
        gt_native = read_gt(indices[1][stem], image.shape[:2])
        pred_native = [read_prediction(indices[i][stem], getattr(args, f"{model}_key"),
                                       getattr(args, f"{model}_kind"))
                       for i, model in ((2, "simple"), (3, "psf"), (4, "crss"))]
        rgb, gt, preds, _ = align_sources(image, gt_native, pred_native)
        render_figure1_style(stem, rgb, gt, preds,
                             output / f"{stem}_three_models.png", args.dpi)


if __name__ == "__main__":
    main()
