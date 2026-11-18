"""Evaluate frozen Conflict-B15 + learned reliability gate on the Polyp1450 test set.

Original-size p>=.5 per-case Dice and IoU, then equal average over 5 datasets.
Uses existing archived original Conflict and Semantic probability maps for strict
reproduction checks.  Exploratory comparison: test cases were analyzed before
this gate was proposed; do not claim a fresh independent holdout.

Run from the CRSS-SAM root after putting this file in scripts/:
 CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_reliability_gate \
   --max-images 2 --output experiments/polyp1450/seed123/reliability_gate_20ep/eval_smoke
 CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_reliability_gate \
   --output experiments/polyp1450/seed123/reliability_gate_20ep/eval_798
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from crss_dataset import resize_longest_and_pad, SAM_PIXEL_MEAN_RGB
from export_polyp_benchmark import restore_logits
from polyp_checkpoint import file_sha256
from scripts.evaluate_polyp1450_conflict import metrics
from scripts.prepare_polyp1450 import TEST_COUNTS, paired
from scripts.train_polyp1450_reliability_gate import ReliabilityGate, load_frozen_models, verify_cuda


def write_csv(path, rows):
    if not rows:
        raise RuntimeError('No results to save')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root', default='data/Polyp1450/dataset/TestDataset')
    parser.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    parser.add_argument('--conflict-checkpoint', default='experiments/polyp1450/seed123/conflict_top10_20ep/latest.pth')
    parser.add_argument('--gate-checkpoint', default='experiments/polyp1450/seed123/reliability_gate_20ep/latest.pth')
    parser.add_argument('--baseline-raw-root', default='experiments/polyp1450/seed123/predictions/raw_probabilities')
    parser.add_argument('--original-conflict-raw-root', default='experiments/polyp1450/seed123/conflict_top10_20ep/eval_798/raw_probabilities')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-images', type=int, default=0, help='Smoke if >0')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('max-images must be >= 0')
    device = torch.device(args.device)
    verify_cuda(device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} must be empty; refusing to overwrite')
    gate_ck = torch.load(args.gate_checkpoint, map_location='cpu', weights_only=False)
    cfg = gate_ck.get('args', {})
    if (gate_ck.get('smoke') or gate_ck.get('epoch') != 20 or
            int(cfg.get('epochs', -1)) != 20 or int(cfg.get('max_train_batches', -1)) != 0 or
            gate_ck.get('selection') != 'fixed_final_epoch' or
            gate_ck.get('method') != 'frozen_conflict_v1_plus_attenuation_gate'):
        raise RuntimeError('Expected completed fixed 20-epoch reliability-gate checkpoint')
    if gate_ck.get('backbone_sha256') != file_sha256(args.heads_checkpoint):
        raise RuntimeError('Gate/base-head checkpoint mismatch')
    if gate_ck.get('conflict_sha256') != file_sha256(args.conflict_checkpoint):
        raise RuntimeError('Gate/original-conflict checkpoint mismatch')
    ext, heads, interaction, _, _ = load_frozen_models(args, device)
    gate = ReliabilityGate().to(device)
    gate.load_state_dict(gate_ck['gate'], strict=True)
    gate.eval()
    output.mkdir(parents=True, exist_ok=True)

    cases = []
    max_semantic_diff = 0.0
    max_original_conflict_diff = 0.0
    pixel_counts = {
        'original_recovered_vs_sem': 0, 'original_damaged_vs_sem': 0,
        'gate_recovered_vs_sem': 0, 'gate_damaged_vs_sem': 0,
    }
    for dataset, expected in TEST_COUNTS.items():
        images, masks = paired(Path(args.test_root) / dataset, expected)
        stems = sorted(images)[:args.max_images] if args.max_images else sorted(images)
        for stem in tqdm(stems, desc=f'Reliability gate {dataset}'):
            bgr = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f'Could not read {images[stem]}')
            h, w = bgr.shape[:2]
            with Image.open(masks[stem]) as im:
                gt = np.asarray(im.convert('L')) > 127
            if gt.shape != (h, w):
                raise RuntimeError(f'GT mismatch: {dataset}/{stem}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            padded = resize_longest_and_pad(rgb, 1024, image_pad_value=SAM_PIXEL_MEAN_RGB)
            tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).to(device)
            mid, deep = ext(tensor)
            sem_logits, struct_logits = heads(mid, deep)
            interaction_out = interaction(
                semantic_feat=deep, structural_feat=mid,
                semantic_logits=sem_logits, structural_logits=struct_logits,
            )
            gated_logits, gate_map, topk_mask = gate(sem_logits, struct_logits, interaction_out)
            old_logits = interaction_out['final_logits']
            old_prob = restore_logits(F.interpolate(old_logits, (256, 256), mode='bilinear', align_corners=False), h, w).astype(np.float32)
            new_prob = restore_logits(F.interpolate(gated_logits, (256, 256), mode='bilinear', align_corners=False), h, w).astype(np.float32)
            semantic_recreated = restore_logits(F.interpolate(sem_logits, (256, 256), mode='bilinear', align_corners=False), h, w).astype(np.float32)
            if any(a.shape != (h, w) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1
                   for a in (old_prob, new_prob, semantic_recreated)):
                raise RuntimeError(f'Invalid raw probabilities: {dataset}/{stem}')

            baseline_path = Path(args.baseline_raw_root) / dataset / f'{stem}.npz'
            old_path = Path(args.original_conflict_raw_root) / dataset / f'{stem}.npz'
            with np.load(baseline_path) as archive:
                semantic_original = archive['semantic'].copy()
                resolver_original = archive['resolver_v2'].copy()
            with np.load(old_path) as archive:
                original_conflict_original = archive['conflict_top10'].copy()
            max_semantic_diff = max(max_semantic_diff, float(np.max(np.abs(semantic_recreated - semantic_original))))
            max_original_conflict_diff = max(max_original_conflict_diff, float(np.max(np.abs(old_prob - original_conflict_original))))
            if max_semantic_diff > .005 or max_original_conflict_diff > .005:
                raise RuntimeError(f'Frozen baseline reproduction mismatch, stop: {dataset}/{stem} semantic={max_semantic_diff}, conflict={max_original_conflict_diff}')
            methods = {
                'semantic': semantic_original,
                'resolver': resolver_original,
                'original_conflict': original_conflict_original,
                'reliability_gate': new_prob,
            }
            row = {'dataset': dataset, 'stem': stem}
            for name, prob in methods.items():
                dice, iou = metrics(prob, gt)
                row[f'{name}_dice'] = dice
                row[f'{name}_iou'] = iou
            row['gate_selected_mean'] = float((gate_map * topk_mask).sum().item() / topk_mask.sum().clamp_min(1).item())
            cases.append(row)
            sem_ok = (semantic_original >= .5) == gt
            old_ok = (original_conflict_original >= .5) == gt
            gated_ok = (new_prob >= .5) == gt
            pixel_counts['original_recovered_vs_sem'] += int((~sem_ok & old_ok).sum())
            pixel_counts['original_damaged_vs_sem'] += int((sem_ok & ~old_ok).sum())
            pixel_counts['gate_recovered_vs_sem'] += int((~sem_ok & gated_ok).sum())
            pixel_counts['gate_damaged_vs_sem'] += int((sem_ok & ~gated_ok).sum())
        print('Completed', dataset, len(stems), flush=True)
    write_csv(output / 'case_metrics.csv', cases)

    names = ('semantic', 'resolver', 'original_conflict', 'reliability_gate')
    summaries = []
    for ds in TEST_COUNTS:
        records = [row for row in cases if row['dataset'] == ds]
        s = {'dataset': ds, 'n': len(records)}
        for name in names:
            s[f'{name}_dice'] = float(np.mean([r[f'{name}_dice'] for r in records]))
            s[f'{name}_iou'] = float(np.mean([r[f'{name}_iou'] for r in records]))
        s['gate_selected_mean'] = float(np.mean([r['gate_selected_mean'] for r in records]))
        summaries.append(s)
    macro = {'dataset': 'MACRO_5', 'n': len(cases)}
    for name in names:
        macro[f'{name}_dice'] = float(np.mean([row[f'{name}_dice'] for row in summaries]))
        macro[f'{name}_iou'] = float(np.mean([row[f'{name}_iou'] for row in summaries]))
    macro['gate_selected_mean'] = float(np.mean([r['gate_selected_mean'] for r in cases]))
    summaries.append(macro)
    write_csv(output / 'summary.csv', summaries)
    report = {
        'status': 'smoke_complete' if args.max_images else 'complete',
        'protocol': 'Original-size sigmoid p>=0.5; per-case Dice/IoU and equal 5-dataset macro mean',
        'scope': 'Exploratory; test set previously inspected for method decisions; not independent final validation',
        'method': 'original_conflict_b15_frozen_plus_learned_attenuation_gate',
        'gate_checkpoint': args.gate_checkpoint,
        'baseline_semantic_reproduction_max_abs_difference': max_semantic_diff,
        'baseline_conflict_reproduction_max_abs_difference': max_original_conflict_diff,
        'pooled_pixel_counts': pixel_counts,
        'results': summaries,
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n=== Final Raw Probability Dice@0.5 ===')
    for s in summaries:
        print(f"{s['dataset']:20s} | semantic {s['semantic_dice']:.4f} | original {s['original_conflict_dice']:.4f} | gate {s['reliability_gate_dice']:.4f} | mean gate {s['gate_selected_mean']:.3f}")
    print('Reproduction max abs diff: semantic', max_semantic_diff, ', conflict', max_original_conflict_diff)
    print('Saved', output / 'summary.csv', output / 'summary.json')


if __name__ == '__main__':
    main()
