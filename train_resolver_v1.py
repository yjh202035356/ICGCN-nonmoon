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
from crss_resolver import PixelReliabilityResolver, soft_reliability_target


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score_prob(prob, target, threshold=0.5, eps=1e-6):
    pred = (prob >= threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def soft_dice_loss_prob(prob, target, eps=1e-6):
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def frozen_predictions(extractor, heads, image, gt, out_size):
    with torch.no_grad():
        mid, deep = extractor(image)
        sem_logits, str_logits = heads(mid, deep)
        sem_logits = F.interpolate(
            sem_logits, (out_size, out_size),
            mode="bilinear", align_corners=False
        )
        str_logits = F.interpolate(
            str_logits, (out_size, out_size),
            mode="bilinear", align_corners=False
        )
        gt_small = F.interpolate(
            gt, (out_size, out_size),
            mode="nearest"
        )
        p_sem = torch.sigmoid(sem_logits)
        p_str = torch.sigmoid(str_logits)
    return p_sem, p_str, gt_small


def reliability_loss(logit, target, conflict, conflict_weight):
    """
    Soft-target BCE.
    Harder/high-conflict pixels get somewhat more weight, but all pixels remain.
    conflict_weight=0 makes this ordinary soft-target BCE.
    """
    per_pixel = F.binary_cross_entropy_with_logits(
        logit, target, reduction="none"
    )
    weight = 1.0 + conflict_weight * conflict.detach()
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)


def fused_seg_loss(prob, gt):
    bce = F.binary_cross_entropy(
        prob.clamp(1e-6, 1.0 - 1e-6), gt
    )
    dice = soft_dice_loss_prob(prob, gt)
    return bce + dice


@torch.no_grad()
def validate(extractor, heads, resolver, loader, device, out_size, tau):
    resolver.eval()

    sem_dice = []
    str_dice = []
    avg_dice = []
    fused_dice = []
    oracle_dice = []
    rel_mae = []

    for batch in tqdm(loader, desc="resolver-val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)

        p_sem, p_str, gt_small = frozen_predictions(
            extractor, heads, image, gt, out_size
        )

        out = resolver(p_sem, p_str)
        p_fused = out["p_fused"]
        p_avg = 0.5 * (p_sem + p_str)

        r_star = soft_reliability_target(
            p_sem, p_str, gt_small, tau=tau
        )
        rel_mae.append(
            torch.abs(out["reliability"] - r_star).mean().item()
        )

        e_sem = torch.abs(p_sem - gt_small)
        e_str = torch.abs(p_str - gt_small)
        p_oracle = torch.where(e_sem < e_str, p_sem, p_str)

        sem_dice.append(dice_score_prob(p_sem, gt_small).item())
        str_dice.append(dice_score_prob(p_str, gt_small).item())
        avg_dice.append(dice_score_prob(p_avg, gt_small).item())
        fused_dice.append(dice_score_prob(p_fused, gt_small).item())
        oracle_dice.append(dice_score_prob(p_oracle, gt_small).item())

    return {
        "semantic_dice": float(np.mean(sem_dice)),
        "structural_dice": float(np.mean(str_dice)),
        "naive_average_dice": float(np.mean(avg_dice)),
        "resolver_fused_dice": float(np.mean(fused_dice)),
        "oracle_dice": float(np.mean(oracle_dice)),
        "resolver_gain_vs_naive": float(np.mean(fused_dice) - np.mean(avg_dice)),
        "remaining_gap_to_oracle": float(np.mean(oracle_dice) - np.mean(fused_dice)),
        "reliability_mae": float(np.mean(rel_mae)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--heads_checkpoint", required=True)

    ap.add_argument("--model_type", default="vit_h",
                    choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--mid_block", type=int, default=15)
    ap.add_argument("--out_size", type=int, default=256)

    ap.add_argument("--work_dir", default="workdir/crss_resolver_v1")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--lambda_rel", type=float, default=1.0)
    ap.add_argument("--lambda_fuse", type=float, default=1.0)
    ap.add_argument("--conflict_weight", type=float, default=4.0)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_all(args.seed)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    train_ds = KvasirTeacherDataset(
        args.train_root, image_size=1024, augment=True
    )
    val_ds = KvasirTeacherDataset(
        args.val_root, image_size=1024, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    ).to(args.device)
    extractor = FrozenSAMFeatureExtractor(
        sam, mid_block=args.mid_block
    ).to(args.device)

    heads = DualCueHeads(
        extractor.mid_channels, extractor.deep_channels
    ).to(args.device)
    heads_ckpt = torch.load(
        args.heads_checkpoint, map_location="cpu"
    )
    heads.load_state_dict(heads_ckpt["heads"])
    heads.eval()
    for p in heads.parameters():
        p.requires_grad = False

    resolver = PixelReliabilityResolver(hidden=32).to(args.device)

    optimizer = torch.optim.AdamW(
        resolver.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_fused = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        resolver.train()
        optimizer.zero_grad(set_to_none=True)

        losses = []
        rel_losses = []
        fuse_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"resolver epoch {epoch}/{args.epochs}"
        )

        for step, batch in enumerate(pbar, start=1):
            image = batch["image"].to(
                args.device, non_blocking=True
            )
            gt = batch["mask"].to(
                args.device, non_blocking=True
            )

            p_sem, p_str, gt_small = frozen_predictions(
                extractor, heads, image, gt, args.out_size
            )

            out = resolver(p_sem, p_str)

            r_star = soft_reliability_target(
                p_sem, p_str, gt_small, tau=args.tau
            )

            l_rel = reliability_loss(
                out["reliability_logit"],
                r_star,
                out["conflict"],
                args.conflict_weight
            )
            l_fuse = fused_seg_loss(
                out["p_fused"], gt_small
            )

            loss = (
                args.lambda_rel * l_rel
                + args.lambda_fuse * l_fuse
            )

            (loss / args.accumulation_steps).backward()

            if (
                step % args.accumulation_steps == 0
                or step == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            rel_losses.append(l_rel.item())
            fuse_losses.append(l_fuse.item())

            pbar.set_postfix(
                loss=f"{np.mean(losses[-20:]):.4f}",
                rel=f"{np.mean(rel_losses[-20:]):.4f}",
                fuse=f"{np.mean(fuse_losses[-20:]):.4f}",
            )

        metrics = validate(
            extractor, heads, resolver,
            val_loader, args.device,
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
        torch.save(ckpt, work / "latest.pth")

        if metrics["resolver_fused_dice"] > best_fused:
            best_fused = metrics["resolver_fused_dice"]
            torch.save(ckpt, work / "best.pth")
            print(
                f"[best] resolver_fused_dice = {best_fused:.6f}"
            )

        (work / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

    print("done")
    print("best checkpoint:", work / "best.pth")


if __name__ == "__main__":
    main()
