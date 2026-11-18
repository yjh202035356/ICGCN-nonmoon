"""Evaluate frozen-backbone structural probes at SAM blocks 14/15/16/17.

Exploratory diagnostic on the pre-existing Polyp1450 798-image test split.
This script DOES NOT train, tune, or select a checkpoint.

Run from the CRSS-SAM repository root, with this file in scripts/:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_block_probe --max-images 2 \
      --output experiments/polyp1450/seed123/block_probe_14_17/eval_smoke
    CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_polyp1450_block_probe \
      --output experiments/polyp1450/seed123/block_probe_14_17/eval_798

Caution: frozen SAM+LoRA backbone was fitted on all 1,450 training images.
Block probes were trained on the same 1,450 images. This test-set analysis
is descriptive; it should not be used to tune or choose a block for an
independent unbiased performance claim. Use a proper train/validation split
with retraining to select architecture, then one locked final test evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from crss_dataset import resize_longest_and_pad, SAM_PIXEL_MEAN_RGB
from crss_model_lora import SmallSegHead
from export_polyp_benchmark import restore_logits
from polyp_checkpoint import build_backbone, file_sha256
from scripts.prepare_polyp1450 import TEST_COUNTS, paired
from segment_anything import sam_model_registry

BLOCKS = (14, 15, 16, 17)


def scores(pred, gt):
    inter = int((pred & gt).sum())
    n = int(pred.sum()) + int(gt.sum())
    union = int((pred | gt).sum())
    return (2 * inter / n if n else 1.0, inter / union if union else 1.0)


def write_csv(filepath, rows):
    with filepath.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--test-root', default='data/Polyp1450/dataset/TestDataset')
    ap.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    ap.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    ap.add_argument('--probe-checkpoint', default='experiments/polyp1450/seed123/block_probe_14_17/latest.pth')
    ap.add_argument('--baseline-raw-root', default='experiments/polyp1450/seed123/predictions/raw_probabilities')
    ap.add_argument('--output', default='experiments/polyp1450/seed123/block_probe_14_17/eval_798')
    ap.add_argument('--max-images', type=int, default=0, help='Smoke test: number per dataset')
    args = ap.parse_args()
    if args.max_images < 0:
        ap.error('--max-images must be nonnegative')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require exactly one CUDA device visible; use CUDA_VISIBLE_DEVICES=0')
    device = torch.device('cuda:0')
    print('CUDA-visible GPU:', torch.cuda.get_device_name(0), flush=True)

    outdir = Path(args.output)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing output: {outdir}')
    if not Path(args.sam_checkpoint).is_file() or not Path(args.heads_checkpoint).is_file():
        raise FileNotFoundError('Missing SAM ViT-H or heads60 checkpoint')

    ck = torch.load(args.probe_checkpoint, map_location='cpu', weights_only=False)
    config = ck['config']
    if (ck.get('epoch') != 20 or int(config['epochs']) != 20 or
        int(config.get('max_train_batches', -1)) != 0 or ck.get('smoke_only') or
        ck.get('blocks') != list(BLOCKS)):
        raise RuntimeError('Probe checkpoint is not a finished full 20-epoch 4-block run')
    if ck.get('backbone_sha256') != file_sha256(args.heads_checkpoint):
        raise RuntimeError('Probe checkpoint was trained using another heads60 backbone')

    original_ck = torch.load(args.heads_checkpoint, map_location='cpu', weights_only=False)
    if (original_ck.get('epoch') != original_ck.get('args', {}).get('epochs') or
        not original_ck.get('args', {}).get('fixed_schedule')):
        raise RuntimeError('Original heads60 checkpoint is not the completed fixed-schedule checkpoint')
    sam = sam_model_registry['vit_h'](checkpoint=args.sam_checkpoint).to(device)
    ext, original_heads, _ = build_backbone(sam, args.heads_checkpoint, device)
    ext.eval(); original_heads.eval()
    if ext.mid_block != 15 or ext.lora_start_block != 16:
        raise RuntimeError('Expected original mid_block 15 and LoRA starting from block 16')

    probes = nn.ModuleDict({str(b): SmallSegHead(ext.mid_channels, hidden=128) for b in BLOCKS}).to(device)
    probes.load_state_dict(ck['probe_heads'], strict=True)
    probes.eval()
    features = {}
    handles = []
    for b in BLOCKS:
        if b == 15:
            continue
        def hook(module, inputs, output, which=b):
            features[which] = output.detach().permute(0, 3, 1, 2).contiguous()
        handles.append(ext.sam.image_encoder.blocks[b].register_forward_hook(hook))

    outdir.mkdir(parents=True, exist_ok=True)
    cases = []
    summaries = []
    pooled = {str(b): {'semantic_wrong_probe_correct': 0,
                       'semantic_correct_probe_wrong': 0,
                       'both_correct': 0, 'both_wrong': 0}
              for b in BLOCKS}
    max_baseline_diff = 0.0
    try:
        for dataset, expected in TEST_COUNTS.items():
            images, masks = paired(Path(args.test_root) / dataset, expected)
            stems = sorted(images)[:args.max_images] if args.max_images else sorted(images)
            group = []
            for stem in tqdm(stems, desc=f'4-block probe {dataset}'):
                image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Cannot read {images[stem]}')
                h, w = image.shape[:2]
                with Image.open(masks[stem]) as gt_im:
                    gt = (np.asarray(gt_im.convert('L')) > 127)
                if gt.shape != (h, w):
                    raise RuntimeError(f'GT size mismatch: {dataset}/{stem}')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_image = resize_longest_and_pad(image, 1024,
                                                      image_pad_value=SAM_PIXEL_MEAN_RGB)
                tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).to(device)
                features.clear()
                mid15, deep = ext(tensor)
                features[15] = mid15
                sem_logits, _ = original_heads(mid15, deep)
                sem256 = F.interpolate(sem_logits, (256, 256), mode='bilinear', align_corners=False)
                recreated_sem = restore_logits(sem256, h, w)
                with np.load(Path(args.baseline_raw_root) / dataset / f'{stem}.npz') as npz:
                    stored_sem = npz['semantic']
                if stored_sem.shape != gt.shape:
                    raise RuntimeError(f'Stored semantic shape mismatch: {dataset}/{stem}')
                difference = float(np.max(np.abs(stored_sem - recreated_sem)))
                max_baseline_diff = max(max_baseline_diff, difference)
                if difference > 0.005:
                    raise RuntimeError(f'Backbone semantic mismatch at {dataset}/{stem}: {difference}')
                sem = stored_sem >= 0.5
                row = {'dataset': dataset, 'stem': stem}
                row['semantic_dice'], row['semantic_iou'] = scores(sem, gt)
                for b in BLOCKS:
                    if b not in features:
                        raise RuntimeError(f'Block {b} hook did not run at {dataset}/{stem}')
                    logits = probes[str(b)](features[b])
                    logits256 = F.interpolate(logits, (256, 256), mode='bilinear', align_corners=False)
                    prob = restore_logits(logits256, h, w)
                    if prob.shape != gt.shape or not np.isfinite(prob).all():
                        raise RuntimeError(f'Invalid probe probabilities: Block {b} {dataset}/{stem}')
                    pred = prob >= 0.5
                    oracle = np.where(sem == gt, sem, pred)  # binary oracle; one correct branch suffices
                    row[f'b{b}_dice'], row[f'b{b}_iou'] = scores(pred, gt)
                    row[f'b{b}_oracle_dice'], row[f'b{b}_oracle_iou'] = scores(oracle, gt)
                    row[f'b{b}_delta_sem'] = row[f'b{b}_dice'] - row['semantic_dice']
                    four = pooled[str(b)]
                    four['semantic_wrong_probe_correct'] += int(((sem != gt) & (pred == gt)).sum())
                    four['semantic_correct_probe_wrong'] += int(((sem == gt) & (pred != gt)).sum())
                    four['both_correct'] += int(((sem == gt) & (pred == gt)).sum())
                    four['both_wrong'] += int(((sem != gt) & (pred != gt)).sum())
                group.append(row); cases.append(row)
            average = {'dataset': dataset, 'n': len(group)}
            for k in group[0]:
                if k not in ('dataset', 'stem'):
                    average[k] = float(np.mean([r[k] for r in group]))
            summaries.append(average)
            print(f'{dataset}: ' + ' '.join(
                f'B{b}={average[f"b{b}_dice"]:.4f} (oracle {average[f"b{b}_oracle_dice"]:.4f})'
                for b in BLOCKS), flush=True)

        macro = {'dataset': 'MACRO_5', 'n': len(cases)}
        for k in summaries[0]:
            if k not in ('dataset', 'n'):
                macro[k] = float(np.mean([r[k] for r in summaries]))
        summaries.append(macro)
        write_csv(outdir / 'case_metrics.csv', cases)
        write_csv(outdir / 'summary.csv', summaries)
        (outdir / 'summary.json').write_text(json.dumps({
            'status': 'smoke_only' if args.max_images else 'complete',
            'protocol': 'Raw probabilities at original image size, fixed 0.5 threshold; per-case Dice/IoU, equal-dataset macro mean',
            'scope': 'Exploratory trained-on-all-1450 frozen backbone probe; test used only for descriptive analysis, not block selection',
            'epoch': ck['epoch'], 'blocks': list(BLOCKS),
            'heads_checkpoint_sha256': ck['backbone_sha256'],
            'max_semantic_baseline_difference': max_baseline_diff,
            'pooled_pixel_counts': pooled,
            'results': summaries,
        }, indent=2), encoding='utf-8')
        print('\n=== 4-block probe results (raw p>=0.5) ===')
        print(f'Baseline semantic reproduction max abs diff: {max_baseline_diff:.6g}')
        print(f'Semantic macro Dice: {macro["semantic_dice"]:.4f}')
        for b in BLOCKS:
            stats = pooled[str(b)]
            print(f'Block {b:2}: Dice={macro[f"b{b}_dice"]:.4f}, Oracle={macro[f"b{b}_oracle_dice"]:.4f}, '
                  f'Sem wrong/Block correct={stats["semantic_wrong_probe_correct"]:,}, '
                  f'Sem correct/Block wrong={stats["semantic_correct_probe_wrong"]:,}')
        print('Smoke only, do not report as a benchmark' if args.max_images else 'Full diagnostic complete: 798 images')
        print('Saved:', outdir / 'summary.csv', outdir / 'summary.json')
    finally:
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    main()
