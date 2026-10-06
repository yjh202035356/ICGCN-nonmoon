import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from crss_dataset import KvasirTeacherDataset
from crss_model import FrozenSAMFeatureExtractor, DualCueHeads
from crss_resolver_v2 import (
    FeatureAwareReliabilityResolver,
    soft_reliability_target_low,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(prob, gt, threshold=0.5, eps=1e-6):
    pred = (prob >= threshold).float()
    inter = (pred * gt).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return ((2*inter + eps)/(denom + eps)).mean().item()


def soft_dice_loss(prob, gt, eps=1e-6):
    inter = (prob * gt).sum(dim=(1,2,3))
    denom = prob.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return (1.0 - (2*inter + eps)/(denom + eps)).mean()


def boundary_band(gt, width=2):
    k = 2 * width + 1
    dil = F.max_pool2d(gt, k, 1, width)
    ero = -F.max_pool2d(-gt, k, 1, width)
    return ((dil - ero) > 0).float()


def weighted_reliability_loss(
    logit_low, target_low, gt_low,
    conflict_low,
    foreground_weight=2.0,
    boundary_weight=2.0,
    conflict_weight=2.0,
):
    per_pixel = F.binary_cross_entropy_with_logits(
        logit_low, target_low, reduction="none"
    )

    # Diagnosis showed background dominates while foreground/boundary are weak.
    bnd = boundary_band(gt_low, width=1)

    weight = torch.ones_like(per_pixel)
    weight = weight + foreground_weight * gt_low
    weight = weight + boundary_weight * bnd
    weight = weight + conflict_weight * conflict_low.detach()

    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)


def fused_seg_loss(prob, gt):
    bce = F.binary_cross_entropy(
        prob.clamp(1e-6, 1-1e-6), gt
    )
    return bce + soft_dice_loss(prob, gt)


@torch.no_grad()
def validate(ext, heads, resolver, loader, device, out_size, tau):
    resolver.eval()

    sems, strs, avgs, fuseds, oracles = [], [], [], [], []
    fg_correct = fg_total = 0
    bg_correct = bg_total = 0
    bnd_correct = bnd_total = 0

    for batch in tqdm(loader, desc="resolver-v2-val", leave=False):
        image = batch["image"].to(device)
        gt = batch["mask"].to(device)

        mid, deep = ext(image)
        sem_l, str_l = heads(mid, deep)

        sem_l = F.interpolate(
            sem_l, (out_size, out_size),
            mode="bilinear", align_corners=False
        )
        str_l = F.interpolate(
            str_l, (out_size, out_size),
            mode="bilinear", align_corners=False
        )
        gt_s = F.interpolate(
            gt, (out_size, out_size),
            mode="nearest"
        )

        p_sem = torch.sigmoid(sem_l)
        p_str = torch.sigmoid(str_l)
        p_avg = 0.5 * (p_sem + p_str)

        out = resolver(p_sem, p_str, mid, deep)
        p_fused = out["p_fused"]

        e_sem = torch.abs(p_sem - gt_s)
        e_str = torch.abs(p_str - gt_s)
        p_oracle = torch.where(e_sem < e_str, p_sem, p_str)

        sems.append(dice_score(p_sem, gt_s))
        strs.append(dice_score(p_str, gt_s))
        avgs.append(dice_score(p_avg, gt_s))
        fuseds.append(dice_score(p_fused, gt_s))
        oracles.append(dice_score(p_oracle, gt_s))

        # Resolver choice accuracy by region at output resolution.
        conflict = torch.abs(p_sem - p_str)
        advantage = torch.abs(e_sem - e_str)
        valid = (conflict > 0.10) & (advantage > 0.05)

        oracle_sem = e_sem < e_str
        pred_sem = out["reliability"] >= 0.5
        correct = oracle_sem == pred_sem

        fg = gt_s >= 0.5
        bg = ~fg
        bnd = boundary_band(gt_s, width=3) > 0

        v = valid & fg
        fg_correct += int((correct & v).sum().item())
        fg_total += int(v.sum().item())

        v = valid & bg
        bg_correct += int((correct & v).sum().item())
        bg_total += int(v.sum().item())

        v = valid & bnd
        bnd_correct += int((correct & v).sum().item())
        bnd_total += int(v.sum().item())

    return {
        "semantic_dice": float(np.mean(sems)),
        "structural_dice": float(np.mean(strs)),
        "naive_average_dice": float(np.mean(avgs)),
        "resolver_v2_fused_dice": float(np.mean(fuseds)),
        "oracle_dice": float(np.mean(oracles)),
        "resolver_v2_gain_vs_naive": float(np.mean(fuseds)-np.mean(avgs)),
        "remaining_gap_to_oracle": float(np.mean(oracles)-np.mean(fuseds)),
        "foreground_choice_accuracy": fg_correct/fg_total if fg_total else None,
        "background_choice_accuracy": bg_correct/bg_total if bg_total else None,
        "boundary_choice_accuracy": bnd_correct/bnd_total if bnd_total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--heads_checkpoint", required=True)

    ap.add_argument("--model_type", default="vit_h",
                    choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--mid_block", type=int, default=15)
    ap.add_argument("--out_size", type=int, default=256)

    ap.add_argument("--work_dir", default="workdir/crss_resolver_v2_smoke")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.10)

    ap.add_argument("--lambda_rel", type=float, default=1.0)
    ap.add_argument("--lambda_fuse", type=float, default=1.0)
    ap.add_argument("--foreground_weight", type=float, default=2.0)
    ap.add_argument("--boundary_weight", type=float, default=2.0)
    ap.add_argument("--conflict_weight", type=float, default=2.0)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_all(args.seed)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    tr = KvasirTeacherDataset(
        args.train_root, image_size=1024, augment=True
    )
    va = KvasirTeacherDataset(
        args.val_root, image_size=1024, augment=False
    )

    tr_loader = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    va_loader = DataLoader(
        va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    ).to(args.device)

    ext = FrozenSAMFeatureExtractor(
        sam, mid_block=args.mid_block
    ).to(args.device)

    heads = DualCueHeads(
        ext.mid_channels, ext.deep_channels
    ).to(args.device)
    hck = torch.load(args.heads_checkpoint, map_location="cpu")
    heads.load_state_dict(hck["heads"])
    heads.eval()
    for p in heads.parameters():
        p.requires_grad = False

    resolver = FeatureAwareReliabilityResolver(
        mid_channels=ext.mid_channels,
        deep_channels=ext.deep_channels,
        proj_channels=16,
        hidden=64,
    ).to(args.device)

    opt = torch.optim.AdamW(
        resolver.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best = -1.0
    history = []

    for epoch in range(1, args.epochs+1):
        resolver.train()
        opt.zero_grad(set_to_none=True)

        losses, rel_losses, fuse_losses = [], [], []

        pbar = tqdm(
            tr_loader,
            desc=f"resolver-v2 epoch {epoch}/{args.epochs}"
        )

        for step, batch in enumerate(pbar, 1):
            image = batch["image"].to(args.device)
            gt = batch["mask"].to(args.device)

            with torch.no_grad():
                mid, deep = ext(image)
                sem_l, str_l = heads(mid, deep)

                sem_l = F.interpolate(
                    sem_l, (args.out_size,args.out_size),
                    mode="bilinear", align_corners=False
                )
                str_l = F.interpolate(
                    str_l, (args.out_size,args.out_size),
                    mode="bilinear", align_corners=False
                )
                gt_s = F.interpolate(
                    gt, (args.out_size,args.out_size),
                    mode="nearest"
                )
                p_sem = torch.sigmoid(sem_l)
                p_str = torch.sigmoid(str_l)

            out = resolver(
                p_sem, p_str, mid.detach(), deep.detach()
            )

            target_low, gt_low = soft_reliability_target_low(
                p_sem, p_str, gt_s,
                target_hw=out["reliability_logit_low"].shape[-2:],
                tau=args.tau
            )

            l_rel = weighted_reliability_loss(
                out["reliability_logit_low"],
                target_low,
                gt_low,
                out["conflict_low"],
                foreground_weight=args.foreground_weight,
                boundary_weight=args.boundary_weight,
                conflict_weight=args.conflict_weight,
            )

            l_fuse = fused_seg_loss(
                out["p_fused"], gt_s
            )

            loss = (
                args.lambda_rel*l_rel
                + args.lambda_fuse*l_fuse
            )

            (loss/args.accumulation_steps).backward()

            if (
                step % args.accumulation_steps == 0
                or step == len(tr_loader)
            ):
                opt.step()
                opt.zero_grad(set_to_none=True)

            losses.append(loss.item())
            rel_losses.append(l_rel.item())
            fuse_losses.append(l_fuse.item())

            pbar.set_postfix(
                loss=f"{np.mean(losses[-20:]):.4f}",
                rel=f"{np.mean(rel_losses[-20:]):.4f}",
                fuse=f"{np.mean(fuse_losses[-20:]):.4f}",
            )

        metrics = validate(
            ext, heads, resolver,
            va_loader, args.device,
            args.out_size, args.tau
        )

        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        metrics["train_rel_loss"] = float(np.mean(rel_losses))
        metrics["train_fuse_loss"] = float(np.mean(fuse_losses))
        history.append(metrics)

        print(json.dumps(metrics, indent=2))

        ckpt = {
            "resolver": resolver.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        }

        torch.save(ckpt, work/"latest.pth")

        if metrics["resolver_v2_fused_dice"] > best:
            best = metrics["resolver_v2_fused_dice"]
            torch.save(ckpt, work/"best.pth")
            print(f"[best] resolver_v2_fused_dice = {best:.6f}")

        (work/"history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

    print("done")
    print("best checkpoint:", work/"best.pth")


if __name__ == "__main__":
    main()
