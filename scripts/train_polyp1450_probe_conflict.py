"""Train conflict Top-10% refinement with a frozen structural block probe.

Method: fixed SAM + LoRA (heads60), original *semantic* head, frozen
20-epoch block-specific structural probe, newly learned ICGCN v1
ConflictAwareBidirectionalInteraction.  This is a frozen-backbone hybrid,
not an end-to-end block-depth re-training of the 60-epoch CRSS-SAM model.

Run from CRSS-SAM root after copying into scripts/:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_probe_conflict \
    --structural-block 14 --epochs 1 --max-train-batches 2 \
    --work-dir experiments/polyp1450/seed123/conflict_probe_b14_smoke

  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_probe_conflict \
    --structural-block 14 --epochs 20 \
    --work-dir experiments/polyp1450/seed123/conflict_probe_b14_20ep

For fair isolation of feature depth, repeat using --structural-block 15
on the same probe checkpoint, epochs and fixed schedule.
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
from crss_model_lora import SmallSegHead, seg_loss
from crss_conflict_interaction import ConflictAwareBidirectionalInteraction
from polyp_checkpoint import build_backbone, file_sha256

BLOCKS = (14, 15, 16, 17)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frozen_probe_models(sam_checkpoint, heads_checkpoint, probe_checkpoint, block, device):
    """Return (extractor, original_semantic_head, frozen_structural_probe, hash info).

    A forward hook caches block-specific B,C,H,W feature maps; call
    `extract_block_feature()` after extractor(image).  Block 15 uses the
    original mid feature; other blocks use their own forward hook.
    """
    if block not in BLOCKS:
        raise ValueError(f'Allowed structural blocks: {BLOCKS}; got {block}')
    for path in (sam_checkpoint, heads_checkpoint, probe_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}')

    head_ck = torch.load(heads_checkpoint, map_location='cpu', weights_only=False)
    hc = head_ck.get('args', {})
    if (not hc.get('fixed_schedule') or head_ck.get('epoch') != hc.get('epochs')
            or head_ck.get('metrics', {}).get('train_images') != 1450):
        raise RuntimeError('heads60 is not a full 1450-image fixed-schedule checkpoint')
    if int(hc.get('mid_block', -1)) != 15 or int(hc.get('lora_start_block', -1)) != 16:
        raise RuntimeError('Expected original heads60 mid_block=15 and LoRA start=16')
    backbone_sha = file_sha256(heads_checkpoint)

    probe_ck = torch.load(probe_checkpoint, map_location='cpu', weights_only=False)
    pc = probe_ck.get('config', {})
    if (probe_ck.get('epoch') != 20 or int(pc.get('epochs', -1)) != 20
            or int(pc.get('max_train_batches', -1)) != 0 or probe_ck.get('smoke_only')
            or probe_ck.get('blocks') != list(BLOCKS)):
        raise RuntimeError('Expected completed 20-epoch 4-block structural-probe checkpoint')
    if probe_ck.get('backbone_sha256') != backbone_sha:
        raise RuntimeError('Probe is not associated with the original heads60 checkpoint')
    probe_sha = file_sha256(probe_checkpoint)

    sam = sam_model_registry[hc.get('model_type', 'vit_h')](checkpoint=sam_checkpoint).to(device)
    ext, original_heads, _ = build_backbone(sam, heads_checkpoint, device)
    if ext.mid_block != 15 or ext.lora_start_block != 16:
        raise RuntimeError('Unexpected SAM extractor configuration')
    ext.eval()
    original_heads.eval()
    for p in ext.parameters():
        p.requires_grad_(False)
    for p in original_heads.parameters():
        p.requires_grad_(False)

    probe_head = SmallSegHead(ext.mid_channels, hidden=128).to(device)
    prefix = f'{block}.'
    probe_state = {k[len(prefix):]: v for k, v in probe_ck['probe_heads'].items()
                   if k.startswith(prefix)}
    if not probe_state:
        raise RuntimeError(f'No structural-probe weights for block {block}')
    probe_head.load_state_dict(probe_state, strict=True)
    probe_head.eval()
    for p in probe_head.parameters():
        p.requires_grad_(False)

    return ext, original_heads.semantic, probe_head, {
        'heads_checkpoint_sha256': backbone_sha,
        'probe_checkpoint_sha256': probe_sha,
        'probe_training_epoch': 20,
        'original_heads_epoch': head_ck['epoch'],
    }


class BlockFeatureTap:
    """Read block-specific features without changing the SAM forward graph."""
    def __init__(self, ext, block):
        self.ext = ext
        self.block = block
        self.captured = None
        self.handle = None
        if block != ext.mid_block:
            self.handle = ext.sam.image_encoder.blocks[block].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.captured = output.detach().permute(0, 3, 1, 2).contiguous()

    def __call__(self, mid15):
        if self.block == self.ext.mid_block:
            return mid15
        if self.captured is None:
            raise RuntimeError(f'Block {self.block} forward hook did not fire')
        return self.captured

    def reset(self):
        self.captured = None

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-root', default='data/Polyp1450/train')
    ap.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    ap.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    ap.add_argument('--probe-checkpoint', default='experiments/polyp1450/seed123/block_probe_14_17/latest.pth')
    ap.add_argument('--structural-block', type=int, choices=BLOCKS, required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accumulation-steps', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--image-size', type=int, default=1024)
    ap.add_argument('--loss-size', type=int, default=256)
    ap.add_argument('--topk-ratio', type=float, default=0.10)
    ap.add_argument('--embed-dim', type=int, default=128)
    ap.add_argument('--num-heads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--max-train-batches', type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        ap.error('Use CUDA_VISIBLE_DEVICES=0: exactly GPU 0 must be visible')
    if args.epochs < 1 or args.batch_size != 1 or args.accumulation_steps < 1:
        ap.error('epochs>0, batch-size=1, accumulation-steps>0 are required')
    if args.max_train_batches < 0 or abs(args.topk_ratio - 0.10) > 1e-9:
        ap.error('max-train-batches>=0 and fixed topk-ratio=0.10 are required')
    if args.image_size != 1024 or args.loss_size != 256:
        ap.error('Use input=1024 and loss=256 to reproduce established training')

    work = Path(args.work_dir)
    checkpoint_path = work / 'latest.pth'
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f'Use an empty work directory; will not overwrite: {work}')
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    set_seed(args.seed)
    print('CUDA-visible GPU:', torch.cuda.get_device_name(0), flush=True)

    dataset = KvasirTeacherDataset(args.train_root, args.image_size,
                                   augment=True, require_teacher=False)
    if len(dataset) != 1450:
        raise RuntimeError(f'Expected 1450 training images, found {len(dataset)}')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    ext, sem_head, probe_head, provenance = load_frozen_probe_models(
        args.sam_checkpoint, args.heads_checkpoint, args.probe_checkpoint,
        args.structural_block, device)
    tap = BlockFeatureTap(ext, args.structural_block)

    interaction = ConflictAwareBidirectionalInteraction(
        semantic_channels=ext.deep_channels, structural_channels=ext.mid_channels,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        topk_ratio=args.topk_ratio).to(device)
    last_conv = interaction.refine_decoder[-1]
    if not isinstance(last_conv, nn.Conv2d):
        raise RuntimeError('Refinement decoder final layer expected Conv2d')
    nn.init.zeros_(last_conv.weight)
    if last_conv.bias is not None:
        nn.init.zeros_(last_conv.bias)
    optimizer = torch.optim.AdamW(interaction.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    print(json.dumps({
        'experiment': 'frozen probe structural feature + conflict Top-10%',
        'structural_block': args.structural_block,
        'training_images': len(dataset), 'epochs': args.epochs,
        'trainable_interaction_parameters': sum(p.numel() for p in interaction.parameters()),
        'smoke_only': bool(args.max_train_batches),
        'selection': 'fixed final epoch; no test-driven checkpoint selection',
        **provenance,
    }, indent=2), flush=True)
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            interaction.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            n_batches = 0
            max_batches = min(len(loader), args.max_train_batches) if args.max_train_batches else len(loader)
            iterator = tqdm(loader, desc=f'Block {args.structural_block} conflict epoch {epoch}/{args.epochs}')
            for sample in iterator:
                img = sample['image'].to(device, non_blocking=True)
                gt = sample['mask'].to(device, non_blocking=True)
                tap.reset()
                with torch.no_grad():
                    mid15, deep = ext(img)
                    selected_feature = tap(mid15)
                    semantic_logits = sem_head(deep)
                    structural_logits = probe_head(selected_feature)
                out = interaction(
                    semantic_feat=deep, structural_feat=selected_feature,
                    semantic_logits=semantic_logits, structural_logits=structural_logits)
                final = F.interpolate(out['final_logits'], (args.loss_size, args.loss_size),
                                      mode='bilinear', align_corners=False)
                gt_small = F.interpolate(gt, (args.loss_size, args.loss_size), mode='nearest')
                loss = seg_loss(final, gt_small)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss epoch {epoch}, step {n_batches+1}')
                (loss / args.accumulation_steps).backward()
                n_batches += 1
                if n_batches % args.accumulation_steps == 0 or n_batches == max_batches:
                    torch.nn.utils.clip_grad_norm_(interaction.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().item()))
                iterator.set_postfix(loss=f'{np.mean(losses[-20:]):.4f}')
                if n_batches == max_batches:
                    break
            row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
                   'train_batches': n_batches, 'train_images_dataset': len(dataset)}
            history.append(row)
            ck = {
                'interaction': interaction.state_dict(), 'args': vars(args), 'epoch': epoch,
                'metrics': row, 'provenance': provenance,
                'selection': 'fixed_final_epoch',
                'interaction_type': 'original_conflict_topk_v1_frozen_probe',
                'structural_block': args.structural_block,
                'smoke': bool(args.max_train_batches),
            }
            tmp = checkpoint_path.with_suffix('.tmp')
            torch.save(ck, tmp)
            tmp.replace(checkpoint_path)
            (work / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    finally:
        tap.close()
    print(f'Completed {args.epochs} epochs: {checkpoint_path}', flush=True)


if __name__ == '__main__':
    main()
