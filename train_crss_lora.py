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
from crss_model_lora import (
    LoRASAMFeatureExtractor,
    DualCueHeads,
    seg_loss,
    structural_boundary_loss,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(prob, target, threshold=0.5, eps=1e-6):
    pred = (prob >= threshold).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    return (
        (2 * inter + eps) / (denom + eps)
    ).mean().item()


def get_lora_state_dict(ext):
    """
    전체 SAM checkpoint를 저장하지 않고
    trainable LoRA parameter만 저장.
    """
    return {
        name: param.detach().cpu()
        for name, param in ext.named_parameters()
        if param.requires_grad
    }


@torch.no_grad()
def validate(
    ext,
    heads,
    loader,
    device,
    loss_size,
    max_batches=0,
):
    heads.eval()
    ext.sam.eval()

    sem_ds = []
    str_ds = []
    avg_ds = []

    for step, b in enumerate(loader, 1):
        image = b["image"].to(device)
        gt = b["mask"].to(device)

        mid, deep = ext(image)

        sem_l, str_l = heads(mid, deep)

        sem_l = F.interpolate(
            sem_l,
            (loss_size, loss_size),
            mode="bilinear",
            align_corners=False,
        )

        str_l = F.interpolate(
            str_l,
            (loss_size, loss_size),
            mode="bilinear",
            align_corners=False,
        )

        gt = F.interpolate(
            gt,
            (loss_size, loss_size),
            mode="nearest",
        )

        psem = torch.sigmoid(sem_l)
        pstr = torch.sigmoid(str_l)
        pavg = 0.5 * (psem + pstr)

        sem_ds.append(dice_score(psem, gt))
        str_ds.append(dice_score(pstr, gt))
        avg_ds.append(dice_score(pavg, gt))

        if max_batches > 0 and step >= max_batches:
            break

    return {
        "semantic_dice": float(np.mean(sem_ds)),
        "structural_dice": float(np.mean(str_ds)),
        "naive_avg_dice": float(np.mean(avg_ds)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)

    ap.add_argument(
        "--model_type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
    )

    ap.add_argument("--mid_block", type=int, default=15)

    ap.add_argument(
        "--lora_start_block",
        type=int,
        default=None,
    )
    
    ap.add_argument(
        "--work_dir",
        default="experiments/kvasir/seed123/lora_qv_r4",
    )

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=4)

    # E0와 동일한 LR을 먼저 사용
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--loss_size", type=int, default=256)

    # E0와 동일한 loss weights
    ap.add_argument("--lambda_sem_gt", type=float, default=0.5)
    ap.add_argument("--lambda_distill", type=float, default=1.0)
    ap.add_argument("--lambda_str_gt", type=float, default=1.0)
    ap.add_argument("--lambda_boundary", type=float, default=1.0)

    # LoRA 설정
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=float, default=4.0)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")

    # smoke test용
    # 0이면 전체 dataset 사용
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)

    # 메모리가 부족할 경우 사용
    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()

    seed_all(args.seed)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    tr = KvasirTeacherDataset(
        args.train_root,
        args.image_size,
        augment=True,
    )

    va = KvasirTeacherDataset(
        args.val_root,
        args.image_size,
        augment=False,
    )

    trl = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val = DataLoader(
        va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------
    # SAM
    # --------------------------------------------------

    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    ).to(args.device)

    # E0와 head 초기화를 최대한 동일하게 만들기 위해
    # LoRA를 삽입하기 전에 head부터 생성한다.
    mid_channels = int(
        sam.image_encoder.blocks[0]
        .norm1.normalized_shape[0]
    )

    deep_channels = int(
        sam.image_encoder.neck[0].out_channels
    )

    heads = DualCueHeads(
        mid_channels,
        deep_channels,
    ).to(args.device)

    # LoRA 초기화가 이후 DataLoader shuffle RNG에
    # 영향을 주지 않도록 RNG state 보존
    cpu_rng_state = torch.get_rng_state()

    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()
    else:
        cuda_rng_state = None

    ext = LoRASAMFeatureExtractor(
        sam=sam,
        mid_block=args.mid_block,
        lora_start_block=args.lora_start_block,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    ).to(args.device)

    # LoRA 초기화 전 RNG 상태 복원
    torch.set_rng_state(cpu_rng_state)

    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    # SAM 자체는 eval mode 유지.
    # LoRA는 Linear이므로 eval mode에서도 gradient 학습 가능.
    ext.sam.eval()

    lora_params = list(ext.lora_parameters())

    head_trainable = sum(
        p.numel()
        for p in heads.parameters()
        if p.requires_grad
    )

    lora_trainable = sum(
        p.numel()
        for p in lora_params
        if p.requires_grad
    )

    total_trainable = head_trainable + lora_trainable

    print("=" * 70)
    print("CRSS-SAM: Q/V LoRA ablation")
    #print("E1: CRSS-SAM + late-block Q/V LoRA")
    print(f"mid_block          : {args.mid_block}")
    print(
        f"LoRA blocks        : "
        f"{ext.lora_start_block}"
        f" ~ "
        f"{len(sam.image_encoder.blocks) - 1}"
    )
    print(f"LoRA rank          : {args.lora_rank}")
    print(f"LoRA alpha         : {args.lora_alpha}")
    print(f"Head trainable     : {head_trainable:,}")
    print(f"LoRA trainable     : {lora_trainable:,}")
    print(f"Total trainable    : {total_trainable:,}")
    print("=" * 70)

    # Head + LoRA만 optimizer에 포함
    opt = torch.optim.AdamW(
        [
            {
                "params": heads.parameters(),
                "lr": args.lr,
            },
            {
                "params": lora_params,
                "lr": args.lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp
    )

    best = -1.0
    history = []

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        heads.train()

        # Frozen SAM은 eval 유지
        ext.sam.eval()

        opt.zero_grad(set_to_none=True)

        losses = []

        pbar = tqdm(
            trl,
            desc=f"epoch {epoch}/{args.epochs}",
        )

        for step, b in enumerate(pbar, 1):
            image = b["image"].to(
                args.device,
                non_blocking=True,
            )

            gt = b["mask"].to(
                args.device,
                non_blocking=True,
            )

            teacher = b["teacher"].to(
                args.device,
                non_blocking=True,
            )

            with torch.cuda.amp.autocast(
                enabled=args.amp
            ):
                # E0와 가장 큰 차이:
                # 여기서는 torch.no_grad()를 사용하지 않음.
                # semantic loss → deep feature → LoRA로 gradient 전달.
                mid, deep = ext(image)

                sem_l, str_l = heads(
                    mid,
                    deep,
                )

                sem_l = F.interpolate(
                    sem_l,
                    (args.loss_size, args.loss_size),
                    mode="bilinear",
                    align_corners=False,
                )

                str_l = F.interpolate(
                    str_l,
                    (args.loss_size, args.loss_size),
                    mode="bilinear",
                    align_corners=False,
                )

                gt_s = F.interpolate(
                    gt,
                    (args.loss_size, args.loss_size),
                    mode="nearest",
                )

                teacher_s = F.interpolate(
                    teacher,
                    (args.loss_size, args.loss_size),
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0, 1)

                # ------------------------------
                # E0와 동일한 loss
                # ------------------------------

                l_sem_gt = seg_loss(
                    sem_l,
                    gt_s,
                )

                l_dist = F.smooth_l1_loss(
                    torch.sigmoid(sem_l),
                    teacher_s,
                )

                l_str_gt = seg_loss(
                    str_l,
                    gt_s,
                )

                l_bnd = structural_boundary_loss(
                    str_l,
                    gt_s,
                )

                loss = (
                    args.lambda_sem_gt * l_sem_gt
                    + args.lambda_distill * l_dist
                    + args.lambda_str_gt * l_str_gt
                    + args.lambda_boundary * l_bnd
                )

                scaled_loss = (
                    loss / args.accumulation_steps
                )

            scaler.scale(
                scaled_loss
            ).backward()

            last_requested_batch = (
                args.max_train_batches > 0
                and step >= args.max_train_batches
            )

            should_step = (
                step % args.accumulation_steps == 0
                or step == len(trl)
                or last_requested_batch
            )

            if should_step:
                scaler.step(opt)
                scaler.update()

                opt.zero_grad(
                    set_to_none=True
                )

            losses.append(loss.item())

            pbar.set_postfix(
                loss=f"{np.mean(losses[-20:]):.4f}"
            )

            if last_requested_batch:
                break

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        m = validate(
            ext,
            heads,
            val,
            args.device,
            args.loss_size,
            max_batches=args.max_val_batches,
        )

        m["epoch"] = epoch
        m["train_loss"] = float(
            np.mean(losses)
        )

        history.append(m)

        print(
            json.dumps(
                m,
                indent=2,
            )
        )

        # E0와 동일한 best criterion
        score = 0.5 * (
            m["semantic_dice"]
            + m["structural_dice"]
        )

        ck = {
            "heads": heads.state_dict(),

            # 전체 SAM은 저장하지 않고
            # LoRA만 별도 저장
            "lora": get_lora_state_dict(ext),

            "args": vars(args),
            "epoch": epoch,
            "metrics": m,
            "score": score,
        }

        torch.save(
            ck,
            work / "latest.pth",
        )

        if score > best:
            best = score

            torch.save(
                ck,
                work / "best.pth",
            )

            print(
                f"[best] {best:.6f}"
            )

        (
            work / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()