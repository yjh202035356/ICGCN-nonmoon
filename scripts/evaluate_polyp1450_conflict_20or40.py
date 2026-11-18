"""Export and evaluate frozen-head Polyp1450 Conflict Top-10% predictions.

Uses the established original-image restoration method, and compares against
already-exported raw probabilities for Semantic/Structural/Naive/Resolver.
Never trains, never selects hyperparameters or a checkpoint using test labels.

From CRSS-SAM root:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_conflict --max-images 2 \
    --output experiments/polyp1450/seed123/conflict_top10_20ep/eval_smoke
  CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_conflict \
    --output experiments/polyp1450/seed123/conflict_top10_20ep/eval_798
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from segment_anything import sam_model_registry
from crss_dataset import resize_longest_and_pad, SAM_PIXEL_MEAN_RGB
from crss_conflict_interaction import ConflictAwareBidirectionalInteraction
from export_polyp_benchmark import restore_logits
from polyp_checkpoint import build_backbone, file_sha256
from scripts.prepare_polyp1450 import TEST_COUNTS, paired


def metrics(probability, truth):
    binary = probability >= 0.5
    intersection = int(np.logical_and(binary, truth).sum())
    predicted = int(binary.sum())
    target = int(truth.sum())
    union = int(np.logical_or(binary, truth).sum())
    return (2.0 * intersection / (predicted + target) if (predicted + target) else 1.0,
            intersection / union if union else 1.0)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to save')
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root', default='data/Polyp1450/dataset/TestDataset')
    parser.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    parser.add_argument('--conflict-checkpoint', default='experiments/polyp1450/seed123/conflict_top10_20ep/latest.pth')
    parser.add_argument('--baseline-raw-root', default='experiments/polyp1450/seed123/predictions/raw_probabilities')
    parser.add_argument('--output', default='experiments/polyp1450/seed123/conflict_top10_20ep/eval_798')
    parser.add_argument('--max-images', type=int, default=0, help='Take this many per dataset for smoke test (not benchmark)')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images cannot be negative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: explicitly set up GPU 0; no silent CPU fallback')
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index not in (None, 0):
        raise ValueError('Only CUDA-visible device 0 is allowed; set CUDA_VISIBLE_DEVICES=0')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('For GPU isolation, set CUDA_VISIBLE_DEVICES=0 (one visible GPU)')
    print('Device:', torch.cuda.get_device_name(device), flush=True)

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output must be empty; refusing to overwrite: {out}')
    ck = torch.load(args.conflict_checkpoint, map_location='cpu')
    config = ck.get('args', {})
    if (ck.get('smoke') or ck.get('epoch') != config.get('epochs') or
            config.get('epochs') not in (20, 40) or config.get('max_train_batches') != 0 or
            ck.get('selection') != 'fixed_final_epoch' or
            ck.get('interaction_type') != 'original_conflict_topk_v1' or
            abs(float(config.get('topk_ratio', -1)) - 0.10) > 1e-9):
        raise RuntimeError('Expected completed 20- or 40-epoch, Top-10%, full-schedule v1 conflict checkpoint')
    if ck.get('backbone_sha256') != file_sha256(args.heads_checkpoint):
        raise RuntimeError('Frozen-head checkpoint SHA256 mismatch; experiment is not comparable')
    hck = torch.load(args.heads_checkpoint, map_location='cpu')
    hargs = hck.get('args', {})
    if not hargs.get('fixed_schedule') or hck.get('epoch') != hargs.get('epochs'):
        raise RuntimeError('Heads checkpoint is not a completed fixed-schedule run')
    sam = sam_model_registry[hargs.get('model_type', 'vit_h')](checkpoint=args.sam_checkpoint).to(device)
    extractor, heads, _ = build_backbone(sam, args.heads_checkpoint, device)
    extractor.eval()
    heads.eval()
    interaction = ConflictAwareBidirectionalInteraction(
        semantic_channels=extractor.deep_channels,
        structural_channels=extractor.mid_channels,
        embed_dim=int(config.get('embed_dim', 128)),
        num_heads=int(config.get('num_heads', 4)),
        topk_ratio=float(config['topk_ratio']),
    ).to(device)
    interaction.load_state_dict(ck['interaction'], strict=True)
    interaction.eval()
    out.mkdir(parents=True, exist_ok=True)  # existing empty directory is safe
    methods = ('semantic', 'structural', 'naive_average', 'resolver_v2', 'conflict_top10')
    rows = []
    sample_count = 0
    max_baseline_difference = 0.0

    for dataset, expected in TEST_COUNTS.items():
        images, masks = paired(Path(args.test_root) / dataset, expected)
        stems = sorted(images)[:args.max_images] if args.max_images else sorted(images)
        raw_dir = out / 'raw_probabilities' / dataset
        png_dir = out / 'binary_masks' / dataset
        raw_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)
        for stem in tqdm(stems, desc=f'Conflict Top-10% {dataset}'):
            image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Unreadable image: {images[stem]}')
            height, width = image.shape[:2]
            with Image.open(masks[stem]) as gt_img:
                truth = np.asarray(gt_img.convert('L')) > 127
            if truth.shape != (height, width):
                raise RuntimeError(f'GT/image dimensions do not match: {dataset}/{stem}')
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = resize_longest_and_pad(rgb, 1024, image_pad_value=SAM_PIXEL_MEAN_RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            mid, deep = extractor(tensor)
            semantic_logits, structural_logits = heads(mid, deep)
            refined = interaction(semantic_feat=deep, structural_feat=mid,
                                  semantic_logits=semantic_logits,
                                  structural_logits=structural_logits)['final_logits']
            native_size = 256
            refined = F.interpolate(refined, (native_size, native_size), mode='bilinear', align_corners=False)
            final_prob = restore_logits(refined, height, width).astype(np.float32)
            if final_prob.shape != truth.shape or not np.isfinite(final_prob).all() or final_prob.min() < 0 or final_prob.max() > 1:
                raise RuntimeError(f'Invalid refined probabilities: {dataset}/{stem}')
            baseline_file = Path(args.baseline_raw_root) / dataset / f'{stem}.npz'
            with np.load(baseline_file) as archive:
                baseline = {name: archive[name].copy() for name in methods if name != 'conflict_top10'}
            if any(array.shape != truth.shape for array in baseline.values()):
                raise RuntimeError(f'Original-size baseline shape mismatch: {dataset}/{stem}')
            # Check the frozen branch actually reproduces the saved baseline
            sem_256 = F.interpolate(semantic_logits, (native_size, native_size), mode='bilinear', align_corners=False)
            recreated_sem = restore_logits(sem_256, height, width).astype(np.float32)
            mismatch = float(np.max(np.abs(recreated_sem - baseline['semantic'])))
            max_baseline_difference = max(max_baseline_difference, mismatch)
            if mismatch > 0.005:
                raise RuntimeError(f'Restored semantic baseline differs from archived prediction ({dataset}/{stem}, max diff={mismatch:.6f}); stop and inspect protocol')
            probabilities = {**baseline, 'conflict_top10': final_prob}
            row = {'dataset': dataset, 'stem': stem}
            for name in methods:
                dice, iou = metrics(probabilities[name], truth)
                row[f'{name}_dice'] = dice
                row[f'{name}_iou'] = iou
            row['gain_vs_semantic_dice'] = row['conflict_top10_dice'] - row['semantic_dice']
            row['gain_vs_resolver_dice'] = row['conflict_top10_dice'] - row['resolver_v2_dice']
            rows.append(row)
            np.savez_compressed(raw_dir / f'{stem}.npz', conflict_top10=final_prob)
            Image.fromarray(((final_prob >= 0.5).astype(np.uint8) * 255)).save(png_dir / f'{stem}.png')
            sample_count += 1
        print(f'Completed {dataset}: {len(stems)} images', flush=True)

    write_csv(out / 'case_metrics.csv', rows)
    summary_rows = []
    for dataset in TEST_COUNTS:
        dataset_rows = [row for row in rows if row['dataset'] == dataset]
        summary = {'dataset': dataset, 'n': len(dataset_rows)}
        for name in methods:
            summary[f'{name}_dice'] = float(np.mean([r[f'{name}_dice'] for r in dataset_rows]))
            summary[f'{name}_iou'] = float(np.mean([r[f'{name}_iou'] for r in dataset_rows]))
        summary['gain_vs_semantic_dice'] = summary['conflict_top10_dice'] - summary['semantic_dice']
        summary['gain_vs_resolver_dice'] = summary['conflict_top10_dice'] - summary['resolver_v2_dice']
        summary_rows.append(summary)
    macro = {'dataset': 'MACRO_5', 'n': sample_count}
    for name in methods:
        macro[f'{name}_dice'] = float(np.mean([r[f'{name}_dice'] for r in summary_rows]))
        macro[f'{name}_iou'] = float(np.mean([r[f'{name}_iou'] for r in summary_rows]))
    macro['gain_vs_semantic_dice'] = macro['conflict_top10_dice'] - macro['semantic_dice']
    macro['gain_vs_resolver_dice'] = macro['conflict_top10_dice'] - macro['resolver_v2_dice']
    summary_rows.append(macro)
    write_csv(out / 'summary.csv', summary_rows)
    (out / 'summary.json').write_text(json.dumps({
        'protocol': 'Raw original-resolution probabilities thresholded at 0.5; per-image Dice and IoU, dataset macro mean.',
        'method': 'original_conflict_topk_v1',
        'topk_ratio': float(config['topk_ratio']),
        'checkpoint': str(args.conflict_checkpoint),
        'heads_checkpoint_sha256': ck['backbone_sha256'],
        'checkpoint_selection': 'fixed_final_epoch_without_test_validation',
        'smoke': bool(args.max_images),
        'max_baseline_semantic_probability_difference': max_baseline_difference,
        'results': summary_rows,
    }, indent=2), encoding='utf-8')
    print('\n=== Raw probability Dice@0.5 / IoU@0.5 ===')
    for r in summary_rows:
        print(f"{r['dataset']:20s} n={r['n']:3d} | semantic {r['semantic_dice']:.4f} | resolver {r['resolver_v2_dice']:.4f} | conflict {r['conflict_top10_dice']:.4f} | delta-sem {r['gain_vs_semantic_dice']:+.4f}")
    print(f'\nBaseline reproduction max abs difference: {max_baseline_difference:.6g}')
    print('Smoke only - NOT benchmark results' if args.max_images else 'Full benchmark complete: 798 images')
    print('Saved:', out / 'summary.csv', out / 'summary.json')


if __name__ == '__main__':
    main()
