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
from crss_model_lora import (
    LoRASAMFeatureExtractor,
    DualCueHeads,
    seg_loss,
)
from crss_conflict_interaction_noattn import (
    ConflictAwareBidirectionalInteraction,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_binary(
    prob,
    gt,
    threshold=0.5,
    eps=1e-6,
):
    pred = prob >= threshold
    gt = gt >= 0.5

    inter = (
        pred & gt
    ).sum().item()

    denom = (
        pred.sum().item()
        + gt.sum().item()
    )

    return (
        2.0 * inter + eps
    ) / (
        denom + eps
    )


def load_lora_weights(
    ext,
    lora_state,
):
    named_params = dict(
        ext.named_parameters()
    )

    missing = []

    for name, tensor in (
        lora_state.items()
    ):
        if name not in named_params:
            missing.append(name)
            continue

        param = named_params[name]

        if param.shape != tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: "
                f"model={tuple(param.shape)}, "
                f"checkpoint={tuple(tensor.shape)}"
            )

        param.data.copy_(
            tensor.to(
                device=param.device,
                dtype=param.dtype,
            )
        )

    if missing:
        raise RuntimeError(
            "LoRA parameters not found:\n"
            + "\n".join(missing)
        )

    print(
        f"Loaded {len(lora_state)} "
        f"LoRA tensors."
    )


def build_frozen_backbone(
    args,
    device,
):
    # --------------------------------------------------
    # Full-LoRA checkpoint
    # --------------------------------------------------

    ck = torch.load(
        args.backbone_checkpoint,
        map_location="cpu",
    )

    ck_args = ck.get(
        "args",
        {},
    )

    mid_block = int(
        ck_args.get(
            "mid_block",
            15,
        )
    )

    lora_rank = int(
        ck_args.get(
            "lora_rank",
            4,
        )
    )

    lora_alpha = float(
        ck_args.get(
            "lora_alpha",
            4.0,
        )
    )

    raw_lora_start = (
        ck_args.get(
            "lora_start_block",
            None,
        )
    )

    if raw_lora_start is None:
        lora_start_block = (
            mid_block + 1
        )
    else:
        lora_start_block = int(
            raw_lora_start
        )

    image_size = int(
        ck_args.get(
            "image_size",
            args.image_size,
        )
    )

    # --------------------------------------------------
    # SAM
    # --------------------------------------------------

    sam = sam_model_registry[
        args.model_type
    ](
        checkpoint=args.sam_checkpoint
    ).to(device)

    ext = LoRASAMFeatureExtractor(
        sam=sam,
        mid_block=mid_block,
        lora_start_block=lora_start_block,
        rank=lora_rank,
        alpha=lora_alpha,
    ).to(device)

    heads = DualCueHeads(
        ext.mid_channels,
        ext.deep_channels,
    ).to(device)

    # --------------------------------------------------
    # Restore
    # --------------------------------------------------

    heads.load_state_dict(
        ck["heads"]
    )

    load_lora_weights(
        ext,
        ck["lora"],
    )

    # --------------------------------------------------
    # Freeze backbone completely
    # --------------------------------------------------

    for p in ext.parameters():
        p.requires_grad = False

    for p in heads.parameters():
        p.requires_grad = False

    ext.eval()
    heads.eval()

    print("=" * 72)
    print("Frozen CRSS backbone")
    print(
        "Checkpoint epoch    :",
        ck.get("epoch"),
    )
    print(
        "mid_block           :",
        mid_block,
    )
    print(
        "LoRA start block    :",
        lora_start_block,
    )
    print(
        "LoRA blocks         :",
        f"{lora_start_block} ~ "
        f"{ext.n_blocks - 1}",
    )
    print(
        "LoRA rank           :",
        lora_rank,
    )
    print(
        "LoRA alpha          :",
        lora_alpha,
    )
    print(
        "mid channels        :",
        ext.mid_channels,
    )
    print(
        "deep channels       :",
        ext.deep_channels,
    )
    print("=" * 72)

    return (
        ext,
        heads,
        image_size,
        ck,
    )


def forward_frozen_backbone(
    ext,
    heads,
    image,
):
    """
    Full-LoRA SAM + dual heads는
    완전히 고정.

    gradient graph를 만들 필요가 없으므로
    no_grad로 memory/time 절약.
    """

    with torch.no_grad():

        mid, deep = ext(
            image
        )

        semantic_logits, (
            structural_logits
        ) = heads(
            mid,
            deep,
        )

    return (
        mid,
        deep,
        semantic_logits,
        structural_logits,
    )


def train_one_epoch(
    loader,
    ext,
    heads,
    interaction,
    optimizer,
    device,
    loss_size,
    accumulation_steps,
    max_batches,
):
    interaction.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    running_correction = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc="conflict-train",
    )

    for step, b in enumerate(
        pbar,
        start=1,
    ):
        if (
            max_batches > 0
            and step > max_batches
        ):
            break

        image = b["image"].to(
            device,
            non_blocking=True,
        )

        gt = b["mask"].to(
            device,
            non_blocking=True,
        )

        (
            mid,
            deep,
            semantic_logits,
            structural_logits,
        ) = forward_frozen_backbone(
            ext,
            heads,
            image,
        )

        out = interaction(
            semantic_feat=deep,
            structural_feat=mid,
            semantic_logits=semantic_logits,
            structural_logits=structural_logits,
        )

        final_logits = (
            out["final_logits"]
        )

        final_logits = F.interpolate(
            final_logits,
            size=(
                loss_size,
                loss_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        gt_loss = F.interpolate(
            gt,
            size=(
                loss_size,
                loss_size,
            ),
            mode="nearest",
        )

        loss = seg_loss(
            final_logits,
            gt_loss,
        )

        scaled_loss = (
            loss
            / accumulation_steps
        )

        scaled_loss.backward()

        if (
            step
            % accumulation_steps
            == 0
        ):
            torch.nn.utils.clip_grad_norm_(
                interaction.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += (
            loss.item()
        )

        running_correction += (
            out["correction"]
            .detach()
            .abs()
            .mean()
            .item()
        )

        n_batches += 1

        correction_mean = (
        out["correction"]
        .detach()
        .abs()
        .mean()
        .item()
    )

    pbar.set_postfix(
        loss=f"{loss.item():.4f}",
        corr=f"{correction_mean:.4f}",
    )

    # accumulation remainder
    if (
        n_batches > 0
        and n_batches
        % accumulation_steps
        != 0
    ):
        torch.nn.utils.clip_grad_norm_(
            interaction.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

    return {
        "train_loss": (
            running_loss
            / max(
                1,
                n_batches,
            )
        ),
        "mean_abs_correction": (
            running_correction
            / max(
                1,
                n_batches,
            )
        ),
    }


@torch.no_grad()
def validate(
    loader,
    ext,
    heads,
    interaction,
    device,
    eval_size,
    max_batches,
):
    interaction.eval()

    semantic_dices = []
    structural_dices = []
    average_dices = []
    final_dices = []

    corrections = []

    pbar = tqdm(
        loader,
        desc="conflict-val",
    )

    for step, b in enumerate(
        pbar,
        start=1,
    ):
        if (
            max_batches > 0
            and step > max_batches
        ):
            break

        image = b["image"].to(
            device,
            non_blocking=True,
        )

        gt = b["mask"].to(
            device,
            non_blocking=True,
        )

        (
            mid,
            deep,
            semantic_logits,
            structural_logits,
        ) = forward_frozen_backbone(
            ext,
            heads,
            image,
        )

        out = interaction(
            semantic_feat=deep,
            structural_feat=mid,
            semantic_logits=semantic_logits,
            structural_logits=structural_logits,
        )

        semantic_logits_eval = (
            F.interpolate(
                semantic_logits,
                size=(
                    eval_size,
                    eval_size,
                ),
                mode="bilinear",
                align_corners=False,
            )
        )

        structural_logits_eval = (
            F.interpolate(
                structural_logits,
                size=(
                    eval_size,
                    eval_size,
                ),
                mode="bilinear",
                align_corners=False,
            )
        )

        final_logits_eval = (
            F.interpolate(
                out["final_logits"],
                size=(
                    eval_size,
                    eval_size,
                ),
                mode="bilinear",
                align_corners=False,
            )
        )

        gt_eval = F.interpolate(
            gt,
            size=(
                eval_size,
                eval_size,
            ),
            mode="nearest",
        )

        ps = torch.sigmoid(
            semantic_logits_eval
        )

        pt = torch.sigmoid(
            structural_logits_eval
        )

        pf = torch.sigmoid(
            final_logits_eval
        )

        pa = 0.5 * (
            ps + pt
        )

        semantic_dices.append(
            dice_binary(
                ps,
                gt_eval,
            )
        )

        structural_dices.append(
            dice_binary(
                pt,
                gt_eval,
            )
        )

        average_dices.append(
            dice_binary(
                pa,
                gt_eval,
            )
        )

        final_dices.append(
            dice_binary(
                pf,
                gt_eval,
            )
        )

        corrections.append(
            out["correction"]
            .abs()
            .mean()
            .item()
        )

    return {
        "semantic_dice":
            float(
                np.mean(
                    semantic_dices
                )
            ),

        "structural_dice":
            float(
                np.mean(
                    structural_dices
                )
            ),

        "naive_average_dice":
            float(
                np.mean(
                    average_dices
                )
            ),

        "final_dice":
            float(
                np.mean(
                    final_dices
                )
            ),

        "gain_vs_semantic":
            float(
                np.mean(
                    final_dices
                )
                - np.mean(
                    semantic_dices
                )
            ),

        "mean_abs_correction":
            float(
                np.mean(
                    corrections
                )
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train_root",
        required=True,
    )

    ap.add_argument(
        "--val_root",
        required=True,
    )

    ap.add_argument(
        "--sam_checkpoint",
        required=True,
    )

    ap.add_argument(
        "--backbone_checkpoint",
        required=True,
    )

    ap.add_argument(
        "--model_type",
        default="vit_h",
        choices=[
            "vit_h",
            "vit_l",
            "vit_b",
        ],
    )

    ap.add_argument(
        "--work_dir",
        required=True,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--image_size",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--loss_size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--eval_size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--embed_dim",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--num_heads",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--topk_ratio",
        type=float,
        default=0.10,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
    )

    args = ap.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    work_dir = Path(
        args.work_dir
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Frozen backbone
    # --------------------------------------------------

    (
        ext,
        heads,
        image_size,
        backbone_ck,
    ) = build_frozen_backbone(
        args,
        device,
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_ds = KvasirTeacherDataset(
        args.train_root,
        image_size,
        True,
    )

    val_ds = KvasirTeacherDataset(
        args.val_root,
        image_size,
        False,
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------
    # Conflict interaction
    # --------------------------------------------------

    interaction = (
        ConflictAwareBidirectionalInteraction(
            semantic_channels=(
                ext.deep_channels
            ),
            structural_channels=(
                ext.mid_channels
            ),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            topk_ratio=(
                args.topk_ratio
            ),
        )
        .to(device)
    )

    # --------------------------------------------------
    # Important:
    #
    # correction decoder의 마지막 Conv를
    # zero initialization.
    #
    # 시작 시:
    # final_logits == semantic_logits
    #
    # 즉 기존 Full-LoRA semantic baseline을
    # 망가뜨리지 않고 refinement 학습 시작.
    # --------------------------------------------------

    last_conv = (
        interaction
        .refine_decoder[-1]
    )

    if not isinstance(
        last_conv,
        nn.Conv2d,
    ):
        raise RuntimeError(
            "Expected final refinement "
            "layer to be Conv2d."
        )

    nn.init.zeros_(
        last_conv.weight
    )

    if (
        last_conv.bias
        is not None
    ):
        nn.init.zeros_(
            last_conv.bias
        )

    trainable = sum(
        p.numel()
        for p in interaction.parameters()
        if p.requires_grad
    )

    print("=" * 72)
    print(
        "E3: Conflict-Aware "
        "Selective Interaction"
    )
    print(
        "Top-K ratio          :",
        args.topk_ratio,
    )
    print(
        "Embedding dim        :",
        args.embed_dim,
    )
    print(
        "Attention heads      :",
        args.num_heads,
    )
    print(
        "Interaction params   :",
        f"{trainable:,}",
    )
    print(
        "Train images         :",
        len(train_ds),
    )
    print(
        "Val images           :",
        len(val_ds),
    )
    print("=" * 72)

    optimizer = torch.optim.AdamW(
        interaction.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------
    # Initial validation
    #
    # zero-init가 잘 됐다면
    # final_dice == semantic_dice에 매우 가까워야 함.
    # --------------------------------------------------

    init_metrics = validate(
        val_loader,
        ext,
        heads,
        interaction,
        device,
        args.eval_size,
        args.max_val_batches,
    )

    print()
    print(
        "[initial validation]"
    )
    print(
        json.dumps(
            init_metrics,
            indent=2,
        )
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_score = -1.0
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print()
        print(
            f"Epoch "
            f"{epoch}/{args.epochs}"
        )

        train_metrics = (
            train_one_epoch(
                train_loader,
                ext,
                heads,
                interaction,
                optimizer,
                device,
                args.loss_size,
                args.accumulation_steps,
                args.max_train_batches,
            )
        )

        val_metrics = validate(
            val_loader,
            ext,
            heads,
            interaction,
            device,
            args.eval_size,
            args.max_val_batches,
        )

        metrics = {
            "epoch":
                epoch,

            **train_metrics,

            **{
                f"val_{k}": v
                for k, v
                in val_metrics.items()
            },
        }

        history.append(
            metrics
        )

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        score = (
            val_metrics[
                "final_dice"
            ]
        )

        checkpoint = {
            "interaction":
                interaction.state_dict(),

            "epoch":
                epoch,

            "metrics":
                val_metrics,

            "backbone_checkpoint":
                args.backbone_checkpoint,

            "backbone_epoch":
                backbone_ck.get(
                    "epoch"
                ),

            "args":
                vars(args),
        }

        torch.save(
            checkpoint,
            work_dir
            / "latest.pth",
        )

        if score > best_score:
            best_score = score

            torch.save(
                checkpoint,
                work_dir
                / "best.pth",
            )

            print(
                "[best]",
                f"{best_score:.6f}",
            )

        (
            work_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 72)
    print(
        "Training finished."
    )
    print(
        "Best final Dice:",
        best_score,
    )
    print(
        "Saved to:",
        work_dir,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()