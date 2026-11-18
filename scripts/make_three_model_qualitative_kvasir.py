#!/usr/bin/env python3
"""Publication-quality 3-model x 4-column qualitative segmentation comparison.

For each *same* held-out Kvasir test case, plot Simple-ViLMedSAM, PSF-SAM and CRSS-SAM.
Columns: Input | Ground Truth | Output Mask | Output (GT & prediction contour overlay).
No GPU/inference required: works with existing per-case masks/probability maps.
Correctly aligns the square 256x256 padded Kvasir evaluation canvas.

Examples:
 python make_three_model_qualitative.py \
   --images dataset/TestDataset/Kvasir/images \
   --gts dataset/TestDataset/Kvasir/masks \
   --simple /actual/path/simple_predictions \
   --psf /actual/path/psf_predictions \
   --crss /actual/path/crss_predictions \
   --case SAMPLE_STEM \
   --out results/qualitative_three_models

If per-case .npz files contain multiple arrays, specify --simple-key, --psf-key,
--crss-key. For raw logits (not probabilities) specify --*-kind logits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
PRED_EXT = IMAGE_EXT | {'.npy', '.npz'}


def index_files(directory: str, extensions: set[str], label: str) -> dict[str, Path]:
    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(f'{label} path does not exist: {root}')
    candidates = [root] if root.is_file() else sorted(root.rglob('*'))
    result: dict[str, Path] = {}
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.stem
        if key in result:
            raise ValueError(f"Duplicate case name '{key}' in {label}:\n"
                             f"  {result[key]}\n  {path}\n"
                             "Pass a dataset-specific directory for each input to avoid ambiguity.")
        result[key] = path
    if not result:
        raise ValueError(f'No usable files found in {label}: {root}')
    return result


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def read_gt(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L'))
    if arr.shape != target_hw:
        raise ValueError(f'GT/image spatial shape mismatch: {path} has {arr.shape}, image {target_hw}; '
                         'check you selected corresponding files.')
    return (arr > 0) if arr.max() <= 1 else (arr > 127)


def read_prediction(path: Path, key: str | None, kind: str) -> np.ndarray:
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z:
            if key is None:
                if len(z.files) != 1:
                    raise ValueError(f'{path} contains {z.files}; pass the matching --MODEL-key')
                key = z.files[0]
            if key not in z.files:
                raise ValueError(f'{path} has keys {z.files}, not {key!r}')
            arr = np.asarray(z[key])
    elif path.suffix.lower() == '.npy':
        arr = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im.convert('L'))
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f'{path}: expected 2D map after squeeze, got shape {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError(f'{path}: NaN or Inf in prediction')
    arr = arr.astype(np.float64, copy=False)
    if kind == 'logits':
        # Numerically stable sigmoid: clip avoids overflow and preserves threshold.
        arr = 1 / (1 + np.exp(-np.clip(arr, -60, 60)))
        return arr >= 0.5
    if kind == 'binary':
        return arr > 0
    if kind == 'probability':
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError(f'{path}: probability mode requires values in [0,1]; '
                             f'found [{arr.min():.3g}, {arr.max():.3g}]')
        return arr >= 0.5
    # Auto: image grayscale masks use >=128; floating arrays must be probabilities.
    if path.suffix.lower() in IMAGE_EXT:
        return (arr > 0) if arr.max() <= 1 else (arr >= 128)
    if arr.min() < 0 or arr.max() > 1:
        if set(np.unique(arr)).issubset({0., 255.}):
            return arr >= 128
        raise ValueError(f'{path}: array values outside [0,1]. '
                         'Use --MODEL-kind logits if these are logits.')
    return arr >= 0.5


def padded_canvas(arr: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    """Match the longest-side resize and bottom/right pad at display resolution."""
    h, w = arr.shape[:2]
    nh, nw = int(round(h * size / max(h, w))), int(round(w * size / max(h, w)))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), resample))
    shape = (size, size) if arr.ndim == 2 else (size, size, arr.shape[-1])
    out = np.zeros(shape, dtype=np.uint8)
    out[:nh, :nw] = resized
    return out


def align_sources(rgb: np.ndarray, gt: np.ndarray,
                  predictions: list[np.ndarray], labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], str]:
    """Use one identical canvas; never stretch a padded prediction to a native image."""
    hw = rgb.shape[:2]
    sizes = [p.shape for p in predictions]
    if all(x == hw for x in sizes):
        return rgb, gt, predictions, 'original-size'
    if all(x in (hw, (256, 256)) for x in sizes):
        rgb_canvas = padded_canvas(rgb, 256, nearest=False)
        gt_canvas = padded_canvas(gt.astype(np.uint8) * 255, 256, nearest=True) > 127
        p_canvas = [
            padded_canvas(p.astype(np.uint8) * 255, 256, nearest=True) > 127 if p.shape == hw else p
            for p in predictions
        ]
        return rgb_canvas, gt_canvas, p_canvas, '256x256 padded'
    raise ValueError(f'Input {hw}, predictions {sizes} ({labels}); '
                     'expected either original-size masks or 256x256 padded masks.')


def edge(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    eroded = im
    for _ in range(max(1, thickness)):
        eroded = eroded.filter(ImageFilter.MinFilter(size=3))
    return mask & ~(np.asarray(eroded) > 127)


def overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, thickness: int = 2) -> np.ndarray:
    image = rgb.copy()
    gt_edge = edge(gt, thickness)
    pred_edge = edge(pred, thickness)
    image[gt_edge] = (49, 226, 83)       # green: ground truth contour
    image[pred_edge] = (255, 65, 72)    # red: prediction contour
    image[gt_edge & pred_edge] = (248, 228, 66)  # yellow: matching contours
    return image


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    den = int(pred.sum()) + int(gt.sum())
    return float(2 * np.logical_and(pred, gt).sum() / den) if den else 1.0


def render_case(stem: str, image_path: Path, gt_path: Path,
                methods: list[tuple[str, Path, str | None, str]],
                out_file: Path, dpi: int, thickness: int, show_dice: bool) -> None:
    rgb_native = read_rgb(image_path)
    gt_native = read_gt(gt_path, rgb_native.shape[:2])
    names = [m[0] for m in methods]
    native_preds = [read_prediction(path, key, kind) for _, path, key, kind in methods]
    rgb, gt, preds, canvas = align_sources(rgb_native, gt_native, native_preds, names)
    fig, axes = plt.subplots(3, 4, figsize=(14.2, 9.2), constrained_layout=False)
    fig.patch.set_facecolor('white')
    for j, title in enumerate(['Input', 'Ground Truth', 'Output Mask', 'Output (Overlay)']):
        axes[0, j].set_title(title, fontsize=16, fontweight='bold', pad=17)
    for i, (name, pred) in enumerate(zip(names, preds)):
        displays = (rgb, gt, pred, overlay(rgb, gt, pred, thickness))
        for j, img in enumerate(displays):
            ax = axes[i, j]
            ax.imshow(img, cmap='gray' if j in (1, 2) else None,
                      vmin=0 if j in (1, 2) else None,
                      vmax=1 if j in (1, 2) else None,
                      interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].text(-0.14, 0.5, name, rotation=90, va='center', ha='center',
                        fontsize=14, fontweight='bold', transform=axes[i, 0].transAxes)
        if show_dice:
            axes[i, 3].set_xlabel(f'Dice (figure canvas) = {dice(pred, gt):.4f}', fontsize=12, labelpad=6)
    fig.subplots_adjust(left=.115, right=.99, top=.915, bottom=.075,
                        wspace=.075, hspace=.18)
    fig.text(.55, .025, 'Overlay: green = GT boundary  |  red = prediction boundary  |  yellow = overlap',
             ha='center', fontsize=11, color='#555555')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    print(f'Saved: {out_file}  [canvas: {canvas}]')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', required=True, help='Directory containing original test RGB images')
    p.add_argument('--gts', required=True, help='Directory containing corresponding GT masks')
    p.add_argument('--simple', required=True, help='Simple-ViLMedSAM output mask/probability directory')
    p.add_argument('--psf', required=True, help='PSF-SAM output mask/probability directory')
    p.add_argument('--crss', required=True, help='CRSS-SAM output mask/probability directory')
    for model in ('simple', 'psf', 'crss'):
        p.add_argument(f'--{model}-key', default=None, help='Per-case .npz key')
        p.add_argument(f'--{model}-kind', default='auto', choices=['auto', 'probability', 'logits', 'binary'],
                       help='Interpretation of stored model predictions')
    p.add_argument('--case', action='append', default=[], help='Image filename stem; repeat to produce multiple images')
    p.add_argument('--count', type=int, default=1, help='When --case omitted, number of shared cases to render')
    p.add_argument('--out', default='results/qualitative_3models', help='Destination folder')
    p.add_argument('--dpi', type=int, default=240)
    p.add_argument('--contour-thickness', type=int, default=2)
    p.add_argument('--no-dice', action='store_true')
    args = p.parse_args()

    sources = [('Input', args.images, IMAGE_EXT), ('GT', args.gts, IMAGE_EXT),
               ('Simple-ViLMedSAM', args.simple, PRED_EXT), ('PSF-SAM', args.psf, PRED_EXT),
               ('CRSS-SAM (Ours)', args.crss, PRED_EXT)]
    indices = [index_files(root, allowed, name) for name, root, allowed in sources]
    intersection = set.intersection(*(set(idx) for idx in indices))
    if not intersection:
        print('No case IDs shared among all five directories.', file=sys.stderr)
        for (name, _, _), idx in zip(sources, indices):
            print(f'  {name}: {len(idx)} files. Examples: {sorted(idx)[:5]}', file=sys.stderr)
        raise SystemExit(2)
    if args.case:
        missing = sorted(set(args.case) - intersection)
        if missing:
            raise ValueError(f'Cases not found in all five inputs: {missing}; first shared: {sorted(intersection)[:10]}')
        cases = args.case
    else:
        cases = sorted(intersection)[:args.count]
        print('No --case specified. Using the first shared case(s) in lexical order:', ', '.join(cases))
    out = Path(args.out)
    for stem in cases:
        methods = [
            ('Simple-ViLMedSAM', indices[2][stem], args.simple_key, args.simple_kind),
            ('PSF-SAM', indices[3][stem], args.psf_key, args.psf_kind),
            ('CRSS-SAM (Ours)', indices[4][stem], args.crss_key, args.crss_kind),
        ]
        render_case(stem, indices[0][stem], indices[1][stem], methods,
                    out / f'{stem}_three_models.png', args.dpi,
                    args.contour_thickness, not args.no_dice)
    print(f'Common test cases across all five inputs: {len(intersection)}')

if __name__ == '__main__':
    main()
