"""Frozen-SAM four-block structural-feature probe for the Polyp1450 term project.

Purpose: isolate extraction depth 14/15/16/17 with a shared, pretrained,
completely frozen heads60 SAM+LoRA backbone. Train four newly initialized,
identical architecture structural-only heads on the *same* 1450 training images.
This is an exploratory frozen-backbone *probe*, NOT a full end-to-end
retraining of the original heads60 experiment. No test masks are read.

Usage (run from CRSS-SAM repo root, after placing this file in scripts/):
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_block_probe \
      --epochs 1 --max-train-batches 2 \
      --work-dir experiments/polyp1450/seed123/block_probe_smoke

    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_block_probe \
      --epochs 20 --work-dir experiments/polyp1450/seed123/block_probe_14_17

Resume only after interruption, without overwriting the existing run:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_block_probe \
      --epochs 20 --work-dir experiments/polyp1450/seed123/block_probe_14_17 --resume
"""

import argparse
import copy
import hashlib
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
from crss_model_lora import SmallSegHead, seg_loss, structural_boundary_loss
from polyp_checkpoint import build_backbone

BLOCKS = (14, 15, 16, 17)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def epoch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-root', default='data/Polyp1450/train')
    ap.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    ap.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    ap.add_argument('--work-dir', default='experiments/polyp1450/seed123/block_probe_14_17')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accumulation-steps', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--image-size', type=int, default=1024)
    ap.add_argument('--loss-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--max-train-batches', type=int, default=0,
                    help='Smoke only: if >0, do not interpret checkpoints as full-data results')
    ap.add_argument('--resume', action='store_true',
                    help='Resume the same fixed experiment from latest.pth')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for full ViT-H feature extraction')
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError('epochs, batch-size and accumulation-steps must be positive')
    if args.max_train_batches < 0:
        raise ValueError('--max-train-batches must be >= 0')
    if args.image_size != 1024:
        raise ValueError('SAM ViT-H expects image-size 1024')

    device = torch.device('cuda:0')   # CUDA_VISIBLE_DEVICES=0 on server
    work = Path(args.work_dir)
    ckpath = work / 'latest.pth'
    if args.resume and not ckpath.is_file():
        raise RuntimeError(f'Cannot resume: missing {ckpath}')
    if not args.resume and ckpath.exists():
        raise RuntimeError(f'Checkpoint already exists; refusing overwrite: {ckpath}')
    work.mkdir(parents=True, exist_ok=True)

    print('CUDA-visible device 0:', torch.cuda.get_device_name(0), flush=True)
    epoch_seed(args.seed)
    dataset = KvasirTeacherDataset(args.train_root, args.image_size,
                                   augment=True, require_teacher=False)
    if len(dataset) != 1450:
        raise RuntimeError(f'Expected 1450 training images, found {len(dataset)}')
    if not Path(args.heads_checkpoint).is_file() or not Path(args.sam_checkpoint).is_file():
        raise FileNotFoundError('Missing SAM or heads60 checkpoint')
    backbone_hash = sha256_file(args.heads_checkpoint)

    # Load exactly the LoRA and SAM weights from the completed term-project run.
    # The original checkpoint's mid-block hook (block 15) remains in place.
    sam = sam_model_registry['vit_h'](checkpoint=args.sam_checkpoint).to(device)
    ext, pretrained_heads, original_ck = build_backbone(
        sam, args.heads_checkpoint, device
    )
    ext.eval()
    pretrained_heads.eval()
    if ext.mid_block != 15 or ext.lora_start_block != 16:
        raise RuntimeError(
            f'Expected original mid_block=15 and lora_start_block=16, got '
            f'{ext.mid_block}, {ext.lora_start_block}; do not compare mismatched backbones'
        )
    for p in ext.parameters():
        p.requires_grad_(False)

    # Save three additional Transformer outputs; block 15 is returned by ext().
    captured = {}
    handles = []
    for block in BLOCKS:
        if block == 15:
            continue
        def record(module, inputs, output, which=block):
            captured[which] = output.detach().permute(0, 3, 1, 2).contiguous()
        handles.append(ext.sam.image_encoder.blocks[block].register_forward_hook(record))

    # All heads have same architecture, identical initial weights, same inputs
    # and the same optimization schedule. Only the hooked feature tensor differs.
    reference = SmallSegHead(ext.mid_channels, hidden=128)
    initial_state = copy.deepcopy(reference.state_dict())
    del reference
    probes = nn.ModuleDict()
    for block in BLOCKS:
        head = SmallSegHead(ext.mid_channels, hidden=128)
        head.load_state_dict(initial_state, strict=True)
        probes[str(block)] = head
    probes.to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 1
    history = []
    if args.resume:
        # Local checkpoint created by this script; never load untrusted files.
        state = torch.load(ckpath, map_location='cpu', weights_only=False)
        prev_args = state['config']
        for field in ('seed', 'batch_size', 'accumulation_steps', 'lr',
                      'weight_decay', 'image_size', 'loss_size', 'max_train_batches'):
            if prev_args[field] != getattr(args, field):
                raise RuntimeError(f'Resume setting differs: {field}')
        if state['backbone_sha256'] != backbone_hash or state['blocks'] != list(BLOCKS):
            raise RuntimeError('Resume mismatch: backbone checkpoint or block list')
        probes.load_state_dict(state['probe_heads'])
        optimizer.load_state_dict(state['optimizer'])
        # Move optimizer buffers from CPU to active CUDA device if needed.
        for group in optimizer.state.values():
            for k, val in group.items():
                if isinstance(val, torch.Tensor):
                    group[k] = val.to(device)
        history = state['history']
        start_epoch = int(state['epoch']) + 1
        if start_epoch > args.epochs:
            print(f'Already completed {state["epoch"]} epochs; nothing to do.')
            return

    manifest = {
        'method': 'frozen_backbone_multi_block_structural_head_probe',
        'interpretation': ('Exploratory fixed-backbone feature-depth comparison. '
                           'Not an independent validation of the pretrained full-data backbone; '
                           'not a full LoRA/dual-head retraining and not a final segmentation benchmark.'),
        'blocks': list(BLOCKS),
        'fixed_backbone_mid_block': ext.mid_block,
        'fixed_lora_start_block': ext.lora_start_block,
        'backbone_sha256': backbone_hash,
        'backbone_training_epoch': original_ck.get('epoch'),
        'all_training_images': len(dataset),
        'training_loss': 'seg_loss(logits,gt) + structural_boundary_loss(logits,gt)',
        'same_head_initialization': True,
        'no_teacher_maps_required': True,
        'uses_test_images_or_labels': False,
        'smoke_only': args.max_train_batches > 0,
        'configuration': vars(args),
    }
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({
        'mode': 'frozen-backbone 4-head depth probe',
        'blocks': list(BLOCKS),
        'training_images': len(dataset),
        'epochs': args.epochs,
        'trainable_probe_parameters': sum(p.numel() for p in probes.parameters()),
        'backbone_checkpoint': args.heads_checkpoint,
        'smoke_only': args.max_train_batches > 0,
        'resume': args.resume,
    }, indent=2), flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Reproducible per-epoch seed even if resuming after an interruption.
            epoch_seed(args.seed + epoch)
            generator = torch.Generator().manual_seed(args.seed * 100000 + epoch)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True,
                                generator=generator)
            probes.train()
            optimizer.zero_grad(set_to_none=True)
            loss_totals = {str(b): 0.0 for b in BLOCKS}
            n_batches = 0
            iterator = tqdm(loader, desc=f'Block 14/15/16/17 epoch {epoch}/{args.epochs}')
            stop_at = min(len(loader), args.max_train_batches) if args.max_train_batches else len(loader)
            for batch in iterator:
                image = batch['image'].to(device, non_blocking=True)
                gt = batch['mask'].to(device, non_blocking=True)
                # No trainable SAM / LoRA parameters; use a single forward for all four features.
                captured.clear()
                with torch.no_grad():
                    mid15, _ = ext(image)
                feature_maps = {15: mid15}
                for block in BLOCKS:
                    if block != 15:
                        if block not in captured:
                            raise RuntimeError(f'Forward hook did not fire for block {block}')
                        feature_maps[block] = captured[block]
                gt_small = F.interpolate(gt, size=(args.loss_size, args.loss_size), mode='nearest')
                all_losses = []
                for block in BLOCKS:
                    logits = probes[str(block)](feature_maps[block])
                    logits = F.interpolate(logits, size=gt_small.shape[-2:],
                                           mode='bilinear', align_corners=False)
                    loss = seg_loss(logits, gt_small) + structural_boundary_loss(logits, gt_small)
                    loss_totals[str(block)] += float(loss.detach().item())
                    all_losses.append(loss)
                # Independent heads have disjoint parameters, so summing losses
                # yields per-head gradients equivalent to training each head alone.
                (sum(all_losses) / args.accumulation_steps).backward()
                n_batches += 1
                if n_batches % args.accumulation_steps == 0 or n_batches == stop_at:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                iterator.set_postfix({f'b{block}': f'{loss_totals[str(block)] / n_batches:.3f}'
                                      for block in BLOCKS})
                if n_batches >= stop_at:
                    break

            metrics = {'epoch': epoch, 'train_batches': n_batches,
                       'train_images_dataset': len(dataset),
                       'mean_train_loss_by_block': {
                           key: value / max(1, n_batches) for key, value in loss_totals.items()
                       }}
            history.append(metrics)
            checkpoint = {
                'epoch': epoch,
                'config': vars(args),
                'blocks': list(BLOCKS),
                'backbone_sha256': backbone_hash,
                'probe_heads': probes.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
                'smoke_only': args.max_train_batches > 0,
            }
            save_atomic(checkpoint, ckpath)
            (work / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(metrics), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print(f'Finished {args.epochs} epochs: {ckpath}', flush=True)


if __name__ == '__main__':
    main()
