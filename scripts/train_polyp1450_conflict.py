"""Train an ICGCN-style Conflict Top-10% correction head on Polyp1450.

The already trained SAM + LoRA + dual heads remain frozen.  No test dataset,
validation-based checkpoint selection or BiomedCLIP inference is involved.

Run from the CRSS-SAM repository root (where polyp_checkpoint.py exists).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from crss_dataset import KvasirTeacherDataset
from crss_model_lora import seg_loss
from crss_conflict_interaction import ConflictAwareBidirectionalInteraction
from polyp_checkpoint import build_backbone, file_sha256


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', default='data/Polyp1450/train')
    parser.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    parser.add_argument('--work-dir', default='experiments/polyp1450/seed123/conflict_top10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--loss-size', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--topk-ratio', type=float, default=0.10)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--expected-train-count', type=int, default=1450)
    parser.add_argument('--max-train-batches', type=int, default=0, help='Smoke test only: >0')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if args.batch_size != 1:
        parser.error('Use batch-size=1 for compatibility with the original ICGCN training setup')
    if args.epochs < 1 or args.accumulation_steps < 1 or args.max_train_batches < 0:
        parser.error('epochs, accumulation-steps must be positive; max-train-batches must be nonnegative')
    if abs(args.topk_ratio - 0.10) > 1e-9:
        parser.error('This experiment pre-fixes Top-K at 10%; do not tune K on the 798 test cases')
    if args.image_size != 1024 or args.loss_size != 256:
        parser.error('Use the established 1024 input / 256 loss resolution')

    seed_everything(args.seed)
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA unavailable; do not silently fall back to CPU')
    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f'Using CUDA-visible device {device.index or 0}: {torch.cuda.get_device_name(device)}')

    work = Path(args.work_dir)
    if (work / 'latest.pth').exists():
        raise FileExistsError(f'{work}/latest.pth already exists. Refusing to overwrite an existing experiment.')
    work.mkdir(parents=True, exist_ok=True)

    dataset = KvasirTeacherDataset(args.train_root, args.image_size, augment=True, require_teacher=False)
    if len(dataset) != args.expected_train_count:
        raise RuntimeError(f'Expected {args.expected_train_count} training samples; found {len(dataset)}')
    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, generator=loader_gen)

    head_ck = torch.load(args.heads_checkpoint, map_location='cpu')
    config = head_ck.get('args', {})
    if config.get('fixed_schedule') is not True or head_ck.get('epoch') != config.get('epochs'):
        raise RuntimeError('Expected fully trained heads60 fixed-schedule checkpoint')
    if head_ck.get('metrics', {}).get('train_images') != 1450:
        raise RuntimeError('Head checkpoint does not certify 1450 training images')
    model_type = config.get('model_type', 'vit_h')
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint).to(device)
    ext, heads, _ = build_backbone(sam, args.heads_checkpoint, device)
    ext.eval()
    heads.eval()

    interaction = ConflictAwareBidirectionalInteraction(
        semantic_channels=ext.deep_channels, structural_channels=ext.mid_channels,
        embed_dim=args.embed_dim, num_heads=args.num_heads, topk_ratio=args.topk_ratio,
    ).to(device)
    final_layer = interaction.refine_decoder[-1]
    if not isinstance(final_layer, nn.Conv2d):
        raise RuntimeError('Unexpected CRSS refine-decoder last layer')
    nn.init.zeros_(final_layer.weight)
    if final_layer.bias is not None:
        nn.init.zeros_(final_layer.bias)
    opt = torch.optim.AdamW(interaction.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({
        'frozen_head_checkpoint': args.heads_checkpoint,
        'train_images': len(dataset), 'epochs': args.epochs,
        'topk_ratio': args.topk_ratio, 'smoke_only': args.max_train_batches > 0,
        'trainable_interaction_params': sum(p.numel() for p in interaction.parameters() if p.requires_grad),
        'checkpoint_selection': 'fixed_final_epoch_without_test_validation',
    }, indent=2), flush=True)

    backbone_sha = file_sha256(args.heads_checkpoint)
    history = []
    for epoch in range(1, args.epochs + 1):
        interaction.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        batches = 0
        progress = tqdm(loader, desc=f'Conflict Top-10% epoch {epoch}/{args.epochs}')
        for step, sample in enumerate(progress, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            img = sample['image'].to(device, non_blocking=True)
            gt = sample['mask'].to(device, non_blocking=True)
            with torch.no_grad():
                mid, deep = ext(img)
                sem, structural = heads(mid, deep)
            outputs = interaction(semantic_feat=deep, structural_feat=mid,
                                  semantic_logits=sem, structural_logits=structural)
            refined = F.interpolate(outputs['final_logits'], (args.loss_size, args.loss_size),
                                    mode='bilinear', align_corners=False)
            target = F.interpolate(gt, (args.loss_size, args.loss_size), mode='nearest')
            loss = seg_loss(refined, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, batch={step}: {loss.item()}')
            (loss / args.accumulation_steps).backward()
            batches += 1
            last = args.max_train_batches > 0 and batches == args.max_train_batches
            if batches % args.accumulation_steps == 0 or last or batches == len(loader):
                torch.nn.utils.clip_grad_norm_(interaction.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            progress.set_postfix(loss=f'{np.mean(losses[-20:]):.4f}')
            if last:
                break
        if not batches:
            raise RuntimeError('No training batches were processed')
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               'train_batches': batches, 'train_images_dataset': len(dataset)}
        history.append(row)
        ck = {'interaction': interaction.state_dict(), 'args': vars(args), 'epoch': epoch,
              'metrics': row, 'backbone_checkpoint': str(args.heads_checkpoint),
              'backbone_sha256': backbone_sha, 'selection': 'fixed_final_epoch',
              'smoke': args.max_train_batches > 0, 'interaction_type': 'original_conflict_topk_v1'}
        torch.save(ck, work / 'latest.pth')
        (work / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    print(f'Finished fixed-schedule training: {work / "latest.pth"}', flush=True)


if __name__ == '__main__':
    main()
