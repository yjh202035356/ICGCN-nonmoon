"""Train an attenuation-only reliability gate on the frozen Polyp1450 CRSS Conflict-B15.

Existing trained components (SAM ViT-H, LoRA, two heads and Conflict v1 interaction)
are FROZEN.  Only a small convolutional gate is trained.  Top-K remains 10%.

Z_gated = Z_sem + M_top10 * sigmoid(G(cues)) * delta_original.

Intended as an exploratory controlled extension, NOT a prospectively held-out
model-selection experiment.  No test images or labels are read during training.

Run from the CRSS-SAM repo root after placing this file into scripts/:
 CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_reliability_gate \
   --epochs 1 --max-train-batches 2 \
   --work-dir experiments/polyp1450/seed123/reliability_gate_smoke
 CUDA_VISIBLE_DEVICES=0 python -m scripts.train_polyp1450_reliability_gate \
   --epochs 20 --work-dir experiments/polyp1450/seed123/reliability_gate_20ep
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from crss_dataset import KvasirTeacherDataset
from crss_model_lora import seg_loss
from crss_conflict_interaction import ConflictAwareBidirectionalInteraction
from polyp_checkpoint import build_backbone, file_sha256


class ReliabilityGate(nn.Module):
    """Confidence-aware, attenuation-only gate predicting a scalar per 64x64 position."""

    def __init__(self, hidden=32, initial_gate_logit=4.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, initial_gate_logit)

    def forward(self, semantic_logits, structural_logits, conflict_output):
        baseline = conflict_output['base_logits'].detach()
        correction = conflict_output['correction'].detach()
        mask = conflict_output['conflict_mask'].detach()
        if baseline.shape != correction.shape or mask.shape != baseline.shape:
            raise RuntimeError('Shape mismatch among baseline, correction and Top-K mask')
        semantic_prob = baseline.sigmoid()
        structural_prob = F.interpolate(
            structural_logits.detach(), size=baseline.shape[-2:], mode='bilinear',
            align_corners=False,
        ).sigmoid()
        disagreement = (semantic_prob - structural_prob).abs()
        uncertainty = 1.0 - (2.0 * semantic_prob - 1.0).abs()
        corrected_prob = (baseline + mask * correction).sigmoid()
        correction_signal = correction.tanh()
        cues = torch.cat([
            semantic_prob, structural_prob, disagreement, uncertainty,
            corrected_prob, correction_signal,
        ], dim=1)
        gate = self.network(cues).sigmoid()
        result = baseline + mask * gate * correction
        return result, gate, mask


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_cuda(device):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU 0 is required')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0; only one GPU must be visible')
    if device.type != 'cuda' or device.index not in (None, 0):
        raise RuntimeError('Set device to cuda:0 after CUDA_VISIBLE_DEVICES=0')
    print('CUDA-visible GPU:', torch.cuda.get_device_name(0), flush=True)


def load_frozen_models(args, device):
    head_ck = torch.load(args.heads_checkpoint, map_location='cpu', weights_only=False)
    h_args = head_ck.get('args', {})
    if (not h_args.get('fixed_schedule') or head_ck.get('epoch') != h_args.get('epochs')
            or head_ck.get('metrics', {}).get('train_images') != 1450):
        raise RuntimeError('Expected completed 1450-image fixed-schedule heads60 checkpoint')
    backbone_sha = file_sha256(args.heads_checkpoint)

    conflict_ck = torch.load(args.conflict_checkpoint, map_location='cpu', weights_only=False)
    c_args = conflict_ck.get('args', {})
    if (conflict_ck.get('smoke') or conflict_ck.get('epoch') != 20 or
            int(c_args.get('epochs', -1)) != 20 or int(c_args.get('max_train_batches', -1)) != 0 or
            conflict_ck.get('selection') != 'fixed_final_epoch' or
            conflict_ck.get('interaction_type') != 'original_conflict_topk_v1' or
            abs(float(c_args.get('topk_ratio', -1.0)) - .10) > 1e-9 or
            conflict_ck.get('backbone_sha256') != backbone_sha):
        raise RuntimeError('Expected original complete 20-epoch Conflict-B15 with matching heads60')
    conflict_sha = file_sha256(args.conflict_checkpoint)

    sam = sam_model_registry[h_args.get('model_type', 'vit_h')](checkpoint=args.sam_checkpoint).to(device)
    ext, heads, _ = build_backbone(sam, args.heads_checkpoint, device)
    interaction = ConflictAwareBidirectionalInteraction(
        semantic_channels=ext.deep_channels,
        structural_channels=ext.mid_channels,
        embed_dim=int(c_args.get('embed_dim', 128)),
        num_heads=int(c_args.get('num_heads', 4)),
        topk_ratio=.10,
    ).to(device)
    interaction.load_state_dict(conflict_ck['interaction'], strict=True)
    for model in (ext, heads, interaction):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return ext, heads, interaction, backbone_sha, conflict_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', default='data/Polyp1450/train')
    parser.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--heads-checkpoint', default='experiments/polyp1450/seed123/heads60/latest.pth')
    parser.add_argument('--conflict-checkpoint', default='experiments/polyp1450/seed123/conflict_top10_20ep/latest.pth')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-train-batches', type=int, default=0, help='Smoke only if positive')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lambda-keep', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.batch_size != 1 or args.epochs < 1 or args.accumulation_steps < 1 or args.max_train_batches < 0:
        parser.error('Use batch-size=1, positive epochs/accumulation and nonnegative max-train-batches')
    if args.lambda_keep < 0:
        parser.error('lambda-keep must be nonnegative')
    device = torch.device(args.device)
    verify_cuda(device)
    seed_all(args.seed)
    work = Path(args.work_dir)
    if (work / 'latest.pth').exists():
        raise FileExistsError(f'{work / "latest.pth"} already exists; do not overwrite')
    work.mkdir(parents=True, exist_ok=True)
    dataset = KvasirTeacherDataset(args.train_root, image_size=1024, augment=True, require_teacher=False)
    if len(dataset) != 1450:
        raise RuntimeError(f'Expected 1450 training images, got {len(dataset)}')
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ext, heads, interaction, backbone_sha, conflict_sha = load_frozen_models(args, device)
    gate = ReliabilityGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({
        'method': 'frozen_conflict_v1_plus_attenuation_gate',
        'train_images': len(dataset), 'epochs': args.epochs,
        'trainable_gate_params': sum(p.numel() for p in gate.parameters()),
        'topk_ratio': 0.10, 'lambda_keep': args.lambda_keep,
        'smoke_only': bool(args.max_train_batches),
        'selection': 'fixed_final_epoch_not_test_selected',
    }, indent=2), flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        seg_losses = []
        selected_gates = []
        n_batches = 0
        progress = tqdm(loader, desc=f'Reliability Gate epoch {epoch}/{args.epochs}')
        for step, sample in enumerate(progress, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            image = sample['image'].to(device, non_blocking=True)
            gt = sample['mask'].to(device, non_blocking=True)
            with torch.no_grad():
                mid, deep = ext(image)
                sem_logits, struct_logits = heads(mid, deep)
                orig = interaction(
                    semantic_feat=deep, structural_feat=mid,
                    semantic_logits=sem_logits, structural_logits=struct_logits,
                )
            gated_logits, gate_map, mask = gate(sem_logits, struct_logits, orig)
            pred = F.interpolate(gated_logits, (256, 256), mode='bilinear', align_corners=False)
            target = F.interpolate(gt, (256, 256), mode='nearest')
            loss_seg = seg_loss(pred, target)
            keep_penalty = (mask * (1.0 - gate_map)).sum() / mask.sum().clamp_min(1.0)
            loss = loss_seg + args.lambda_keep * keep_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, step={step}')
            (loss / args.accumulation_steps).backward()
            n_batches += 1
            last = args.max_train_batches > 0 and n_batches >= args.max_train_batches
            if n_batches % args.accumulation_steps == 0 or n_batches == len(loader) or last:
                nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            seg_losses.append(float(loss_seg.item()))
            selected_gates.append(float((mask * gate_map).sum().item() / mask.sum().clamp_min(1).item()))
            progress.set_postfix(loss=f'{np.mean(losses[-20:]):.4f}', gate=f'{np.mean(selected_gates[-20:]):.3f}')
            if last:
                break
        if n_batches == 0:
            raise RuntimeError('No training batches processed')
        row = {
            'epoch': epoch, 'mean_train_loss': float(np.mean(losses)),
            'mean_train_seg_loss': float(np.mean(seg_losses)),
            'mean_selected_gate': float(np.mean(selected_gates)),
            'train_batches': n_batches, 'train_images_dataset': len(dataset),
        }
        history.append(row)
        checkpoint = {
            'gate': gate.state_dict(), 'args': vars(args), 'epoch': epoch,
            'metrics': row, 'backbone_sha256': backbone_sha,
            'conflict_sha256': conflict_sha,
            'method': 'frozen_conflict_v1_plus_attenuation_gate',
            'selection': 'fixed_final_epoch', 'smoke': bool(args.max_train_batches),
        }
        torch.save(checkpoint, work / 'latest.pth')
        (work / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    print('Completed:', work / 'latest.pth', flush=True)


if __name__ == '__main__':
    main()
