"""Evaluate frozen-probe structural Block 14/15/16/17 + CRSS Conflict Top-10%.

Compare raw probabilities (original resolution, threshold=0.5) against the
archived semantic, Resolver v2 and original Block-15 Conflict predictions.
The original Block-15 comparison was trained with a different structural
head regimen (heads60); this is a descriptive, NOT a controlled pure block
ablation. For a depth-only ablation train the same probe+interaction at B15.

From CRSS-SAM root, copy file to scripts/, then:
 CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_probe_conflict \
   --structural-block 14 --max-images 2 \
   --output experiments/polyp1450/seed123/conflict_probe_b14_20ep/eval_smoke
 CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_probe_conflict \
   --structural-block 14 \
   --output experiments/polyp1450/seed123/conflict_probe_b14_20ep/eval_798
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry  # imported by loader; explicit dependency check
from crss_dataset import resize_longest_and_pad, SAM_PIXEL_MEAN_RGB
from crss_conflict_interaction import ConflictAwareBidirectionalInteraction
from export_polyp_benchmark import restore_logits
from scripts.prepare_polyp1450 import TEST_COUNTS, paired
from scripts.train_polyp1450_probe_conflict import (
    BLOCKS, BlockFeatureTap, load_frozen_probe_models)


def evaluate_mask(binary, truth):
    inter = int((binary & truth).sum())
    total = int(binary.sum()) + int(truth.sum())
    union = int((binary | truth).sum())
    return (2 * inter / total if total else 1.0,
            inter / union if union else 1.0)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--structural-block', type=int, choices=BLOCKS, required=True)
    ap.add_argument('--test-root', default='data/Polyp1450/dataset/TestDataset')
    ap.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    ap.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    ap.add_argument('--probe-checkpoint', default='experiments/polyp1450/seed123/block_probe_14_17/latest.pth')
    ap.add_argument('--conflict-checkpoint', default=None)
    ap.add_argument('--baseline-raw-root', default='experiments/polyp1450/seed123/predictions/raw_probabilities')
    ap.add_argument('--original-conflict-raw-root', default='experiments/polyp1450/seed123/conflict_top10_20ep/eval_798/raw_probabilities')
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-images', type=int, default=0)
    args = ap.parse_args()
    if args.max_images < 0:
        ap.error('--max-images must be >=0')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        ap.error('Use CUDA_VISIBLE_DEVICES=0, exposing GPU 0 only')
    device = torch.device('cuda:0')
    print('CUDA-visible GPU:', torch.cuda.get_device_name(0), flush=True)

    if args.conflict_checkpoint is None:
        args.conflict_checkpoint = (
            f'experiments/polyp1450/seed123/conflict_probe_b{args.structural_block}_20ep/latest.pth')
    if not Path(args.conflict_checkpoint).is_file():
        raise FileNotFoundError(f'Missing conflict checkpoint: {args.conflict_checkpoint}')
    ck = torch.load(args.conflict_checkpoint, map_location='cpu', weights_only=False)
    config = ck.get('args', {})
    if (ck.get('smoke') or int(ck.get('epoch', -1)) != 20 or int(config.get('epochs', -1)) != 20
            or int(config.get('max_train_batches', -1)) != 0
            or ck.get('selection') != 'fixed_final_epoch'
            or ck.get('interaction_type') != 'original_conflict_topk_v1_frozen_probe'
            or ck.get('structural_block') != args.structural_block
            or abs(float(config.get('topk_ratio', -1)) - 0.10) > 1e-9):
        raise RuntimeError('Expected complete 20-epoch fixed 10% probe-refinement checkpoint')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output must be empty to avoid overwriting: {out}')

    ext, sem_head, probe_head, provenance = load_frozen_probe_models(
        args.sam_checkpoint, args.heads_checkpoint, args.probe_checkpoint,
        args.structural_block, device)
    if ck.get('provenance') != provenance:
        raise RuntimeError('Stored probe/heads SHA or provenance differs from the trained interaction')
    tap = BlockFeatureTap(ext, args.structural_block)
    interaction = ConflictAwareBidirectionalInteraction(
        semantic_channels=ext.deep_channels, structural_channels=ext.mid_channels,
        embed_dim=int(config.get('embed_dim', 128)),
        num_heads=int(config.get('num_heads', 4)),
        topk_ratio=float(config['topk_ratio'])).to(device)
    interaction.load_state_dict(ck['interaction'], strict=True)
    interaction.eval()

    out.mkdir(parents=True, exist_ok=True)
    cases = []
    summary = []
    max_sem_diff = 0.0
    methods = ('semantic', 'resolver_v2', 'conflict_b15_original',
               'probe_structural', 'conflict_probe')
    try:
        for ds, expected in TEST_COUNTS.items():
            images, masks = paired(Path(args.test_root) / ds, expected)
            stems = sorted(images)[:args.max_images] if args.max_images else sorted(images)
            group = []
            for stem in tqdm(stems, desc=f'Conflict probe B{args.structural_block} {ds}'):
                im = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
                if im is None:
                    raise RuntimeError(f'Could not read image {ds}/{stem}')
                h, w = im.shape[:2]
                with Image.open(masks[stem]) as gtim:
                    gt = np.asarray(gtim.convert('L')) > 127
                if gt.shape != (h, w):
                    raise RuntimeError(f'GT shape mismatch {ds}/{stem}')
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                input_arr = resize_longest_and_pad(rgb, 1024,
                                                   image_pad_value=SAM_PIXEL_MEAN_RGB)
                tensor = torch.from_numpy(input_arr).permute(2, 0, 1).unsqueeze(0).to(device)
                tap.reset()
                mid15, deep = ext(tensor)
                mid_probe = tap(mid15)
                sem_logits = sem_head(deep)
                str_logits = probe_head(mid_probe)
                out_logits = interaction(
                    semantic_feat=deep, structural_feat=mid_probe,
                    semantic_logits=sem_logits, structural_logits=str_logits)['final_logits']
                sem_p = restore_logits(F.interpolate(sem_logits, (256, 256),
                                  mode='bilinear', align_corners=False), h, w).astype(np.float32)
                str_p = restore_logits(F.interpolate(str_logits, (256, 256),
                                  mode='bilinear', align_corners=False), h, w).astype(np.float32)
                final_p = restore_logits(F.interpolate(out_logits, (256, 256),
                                    mode='bilinear', align_corners=False), h, w).astype(np.float32)
                with np.load(Path(args.baseline_raw_root) / ds / f'{stem}.npz') as arr:
                    raw_sem = arr['semantic'].copy()
                    raw_res = arr['resolver_v2'].copy()
                with np.load(Path(args.original_conflict_raw_root) / ds / f'{stem}.npz') as arr:
                    raw_orig_conflict = arr['conflict_top10'].copy()
                all_arrays = {
                    'semantic': raw_sem, 'resolver_v2': raw_res,
                    'conflict_b15_original': raw_orig_conflict,
                    'probe_structural': str_p, 'conflict_probe': final_p,
                }
                if any(v.shape != gt.shape or not np.isfinite(v).all()
                       or v.min() < 0 or v.max() > 1 for v in all_arrays.values()):
                    raise RuntimeError(f'Invalid original-size arrays at {ds}/{stem}')
                sem_diff = float(np.max(np.abs(raw_sem - sem_p)))
                max_sem_diff = max(max_sem_diff, sem_diff)
                if sem_diff > 0.005:
                    raise RuntimeError(f'Semantic backbone reproduction mismatch: {ds}/{stem} diff={sem_diff:.6f}')
                row = {'dataset': ds, 'stem': stem}
                for method, prob in all_arrays.items():
                    dice, iou = evaluate_mask(prob >= 0.5, gt)
                    row[f'{method}_dice'] = dice
                    row[f'{method}_iou'] = iou
                row['delta_probe_vs_semantic_dice'] = row['conflict_probe_dice'] - row['semantic_dice']
                row['delta_probe_vs_original_conflict_dice'] = (row['conflict_probe_dice']
                                                              - row['conflict_b15_original_dice'])
                group.append(row)
                cases.append(row)
                save_dir = out / 'raw_probabilities' / ds
                save_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_dir / f'{stem}.npz', probe_structural=str_p,
                                    conflict_probe=final_p)
            rec = {'dataset': ds, 'n': len(group)}
            for name in methods:
                for key in ('dice', 'iou'):
                    field = f'{name}_{key}'
                    rec[field] = float(np.mean([x[field] for x in group]))
            rec['delta_probe_vs_semantic_dice'] = rec['conflict_probe_dice'] - rec['semantic_dice']
            rec['delta_probe_vs_original_conflict_dice'] = (rec['conflict_probe_dice']
                                                            - rec['conflict_b15_original_dice'])
            summary.append(rec)
            print(f'{ds:20s} n={len(group):3d} | semantic {rec["semantic_dice"]:.4f} | '
                  f'original Conflict-B15 {rec["conflict_b15_original_dice"]:.4f} | '
                  f'probe B{args.structural_block} + Conflict {rec["conflict_probe_dice"]:.4f}', flush=True)
        macro = {'dataset': 'MACRO_5', 'n': len(cases)}
        for name in methods:
            for key in ('dice', 'iou'):
                field = f'{name}_{key}'
                macro[field] = float(np.mean([s[field] for s in summary]))
        macro['delta_probe_vs_semantic_dice'] = macro['conflict_probe_dice'] - macro['semantic_dice']
        macro['delta_probe_vs_original_conflict_dice'] = (macro['conflict_probe_dice']
                                                         - macro['conflict_b15_original_dice'])
        summary.append(macro)
        write_csv(out / 'case_metrics.csv', cases)
        write_csv(out / 'summary.csv', summary)
        (out / 'summary.json').write_text(json.dumps({
            'protocol': 'Original-size float sigmoid >=0.5; case Dice/IoU, equal five-dataset macro',
            'scope': ('Exploratory; original Conflict-B15 used original heads60 Structural Head, '
                      'probe Conflict uses a separately trained frozen 20-epoch probe head. '
                      'Not a pure block-depth causal ablation; test already inspected.'),
            'structural_block': args.structural_block,
            'checkpoint': args.conflict_checkpoint,
            'model_provenance': provenance,
            'smoke': bool(args.max_images),
            'max_semantic_baseline_difference': max_sem_diff,
            'results': summary,
        }, indent=2), encoding='utf-8')
        print('\n=== Final raw probability Dice@0.5 ===', flush=True)
        print(f'Semantic macro             : {macro["semantic_dice"]:.4f}', flush=True)
        print(f'Resolver macro             : {macro["resolver_v2_dice"]:.4f}', flush=True)
        print(f'Original Conflict-B15 macro: {macro["conflict_b15_original_dice"]:.4f}', flush=True)
        print(f'Probe B{args.structural_block} Conflict macro: {macro["conflict_probe_dice"]:.4f}', flush=True)
        print('Semantic reproduction max difference:', max_sem_diff, flush=True)
        print('SMOKE only, do NOT report as 798-case benchmark' if args.max_images else
              'Complete 798-case exploratory diagnostic', flush=True)
        print('Saved:', out / 'summary.csv', out / 'summary.json', flush=True)
    finally:
        tap.close()


if __name__ == '__main__':
    main()
