import argparse
import json
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
)
from crss_conflict_interaction import (
    ConflictAwareBidirectionalInteraction,
)


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

    for name, tensor in lora_state.items():
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
        f"Loaded {len(lora_state)} LoRA tensors."
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        required=True,
    )

    ap.add_argument(
        "--sam_checkpoint",
        required=True,
    )

    ap.add_argument(
        "--interaction_checkpoint",
        required=True,
    )

    ap.add_argument(
        "--backbone_checkpoint",
        default=None,
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
        "--eval_size",
        type=int,
        default=256,
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
        "--output",
        default=None,
    )

    args = ap.parse_args()

    device = torch.device(
        args.device
    )

    # ==================================================
    # Load interaction checkpoint
    # ==================================================

    interaction_ck = torch.load(
        args.interaction_checkpoint,
        map_location="cpu",
    )

    interaction_args = interaction_ck.get(
        "args",
        {},
    )

    interaction_epoch = interaction_ck.get(
        "epoch"
    )

    # interaction checkpoint 안에 저장된
    # backbone checkpoint를 기본적으로 사용
    backbone_checkpoint = (
        args.backbone_checkpoint
    )

    if backbone_checkpoint is None:
        backbone_checkpoint = (
            interaction_ck.get(
                "backbone_checkpoint"
            )
        )

    if backbone_checkpoint is None:
        raise RuntimeError(
            "Backbone checkpoint was not found. "
            "Specify --backbone_checkpoint."
        )

    # ==================================================
    # Load backbone checkpoint
    # ==================================================

    backbone_ck = torch.load(
        backbone_checkpoint,
        map_location="cpu",
    )

    backbone_args = backbone_ck.get(
        "args",
        {},
    )

    mid_block = int(
        backbone_args.get(
            "mid_block",
            15,
        )
    )

    lora_rank = int(
        backbone_args.get(
            "lora_rank",
            4,
        )
    )

    lora_alpha = float(
        backbone_args.get(
            "lora_alpha",
            4.0,
        )
    )

    raw_lora_start = (
        backbone_args.get(
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
        backbone_args.get(
            "image_size",
            1024,
        )
    )

    embed_dim = int(
        interaction_args.get(
            "embed_dim",
            128,
        )
    )

    num_heads = int(
        interaction_args.get(
            "num_heads",
            4,
        )
    )

    topk_ratio = float(
        interaction_args.get(
            "topk_ratio",
            0.10,
        )
    )

    print("=" * 74)
    print("CRSS-SAM Conflict Evaluation")
    print("=" * 74)
    print(
        "Interaction checkpoint :",
        args.interaction_checkpoint,
    )
    print(
        "Interaction epoch      :",
        interaction_epoch,
    )
    print(
        "Backbone checkpoint    :",
        backbone_checkpoint,
    )
    print(
        "Backbone epoch         :",
        backbone_ck.get("epoch"),
    )
    print(
        "mid_block              :",
        mid_block,
    )
    print(
        "LoRA start block       :",
        lora_start_block,
    )
    print(
        "LoRA rank              :",
        lora_rank,
    )
    print(
        "LoRA alpha             :",
        lora_alpha,
    )
    print(
        "Top-K ratio            :",
        topk_ratio,
    )
    print(
        "Embedding dim          :",
        embed_dim,
    )
    print(
        "Attention heads        :",
        num_heads,
    )
    print("=" * 74)

    # ==================================================
    # Dataset
    # ==================================================

    ds = KvasirTeacherDataset(
        args.data_root,
        image_size,
        False,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        "Evaluation images      :",
        len(ds),
    )
    print("=" * 74)

    # ==================================================
    # SAM + Full LoRA
    # ==================================================

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

    heads.load_state_dict(
        backbone_ck["heads"]
    )

    load_lora_weights(
        ext,
        backbone_ck["lora"],
    )

    # ==================================================
    # Conflict interaction
    # ==================================================

    interaction = (
        ConflictAwareBidirectionalInteraction(
            semantic_channels=(
                ext.deep_channels
            ),
            structural_channels=(
                ext.mid_channels
            ),
            embed_dim=embed_dim,
            num_heads=num_heads,
            topk_ratio=topk_ratio,
        )
        .to(device)
    )

    interaction.load_state_dict(
        interaction_ck["interaction"]
    )

    ext.eval()
    heads.eval()
    interaction.eval()

    # ==================================================
    # Dice metrics
    # ==================================================

    semantic_dices = []
    structural_dices = []
    average_dices = []
    final_dices = []
    oracle_dices = []

    # ==================================================
    # Pixel-level analysis
    # ==================================================

    total_pixels = 0

    semantic_error_pixels = 0
    final_error_pixels = 0

    conflict_pixels = 0

    conflict_semantic_error = 0
    conflict_final_error = 0

    # ==================================================
    # Recovery / damage analysis
    # ==================================================

    # Semantic이 틀렸는데 Final이 맞게 고친 pixel
    true_recovered_pixels = 0

    # Semantic은 맞았는데 Final이 틀리게 만든 pixel
    true_damaged_pixels = 0

    # conflict 영역에서 Semantic이 맞았던 pixel 수
    semantic_correct_conflict = 0

    # Structural이 Semantic의 오류를
    # 보완할 수 있었던 pixel
    structural_recoverable_pixels = 0

    # 그중 실제 Final이 복구한 pixel
    structural_assisted_recovered_pixels = 0
    
    # conflict 영역에서
    # 어느 branch가 GT에 더 가까웠는지
    semantic_win = 0
    structural_win = 0
    ties = 0

    correction_abs = []

    with torch.no_grad():

        for b in tqdm(
            loader,
            desc="CRSS-conflict-test",
        ):
            image = b["image"].to(
                device,
                non_blocking=True,
            )

            gt = b["mask"].to(
                device,
                non_blocking=True,
            )

            # ==========================================
            # Frozen Full-LoRA backbone
            # ==========================================

            mid, deep = ext(
                image
            )

            semantic_logits, (
                structural_logits
            ) = heads(
                mid,
                deep,
            )

            # ==========================================
            # Conflict refinement
            # ==========================================

            out = interaction(
                semantic_feat=deep,
                structural_feat=mid,
                semantic_logits=semantic_logits,
                structural_logits=structural_logits,
            )

            final_logits = (
                out["final_logits"]
            )

            # ==========================================
            # Resize all predictions
            # ==========================================

            semantic_logits_eval = (
                F.interpolate(
                    semantic_logits,
                    size=(
                        args.eval_size,
                        args.eval_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            structural_logits_eval = (
                F.interpolate(
                    structural_logits,
                    size=(
                        args.eval_size,
                        args.eval_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            final_logits_eval = (
                F.interpolate(
                    final_logits,
                    size=(
                        args.eval_size,
                        args.eval_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            gt_eval = F.interpolate(
                gt,
                size=(
                    args.eval_size,
                    args.eval_size,
                ),
                mode="nearest",
            )

            # Top-K conflict mask는
            # discrete selection mask이므로 nearest
            conflict_mask = F.interpolate(
                out["conflict_mask"],
                size=(
                    args.eval_size,
                    args.eval_size,
                ),
                mode="nearest",
            ) >= 0.5

            # ==========================================
            # Probabilities
            # ==========================================

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

            # ==========================================
            # Oracle
            # ==========================================

            semantic_abs_error = (
                ps - gt_eval
            ).abs()

            structural_abs_error = (
                pt - gt_eval
            ).abs()

            choose_semantic = (
                semantic_abs_error
                < structural_abs_error
            )

            choose_structural = (
                structural_abs_error
                < semantic_abs_error
            )

            tie = (
                semantic_abs_error
                - structural_abs_error
            ).abs() <= 0.02

            oracle_prob = torch.where(
                choose_semantic,
                ps,
                pt,
            )

            # ==========================================
            # Dice
            # ==========================================

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

            oracle_dices.append(
                dice_binary(
                    oracle_prob,
                    gt_eval,
                )
            )

            # ==========================================
            # Binary predictions
            # ==========================================

            gt_bin = (
                gt_eval >= 0.5
            )

            sem_pred = (
                ps >= 0.5
            )

            str_pred = (
                pt >= 0.5
            )

            final_pred = (
                pf >= 0.5
            )

            sem_wrong = (
                sem_pred != gt_bin
            )

            str_wrong = (
                str_pred != gt_bin
            )

            final_wrong = (
                final_pred != gt_bin
            )

            sem_correct = ~sem_wrong
            str_correct = ~str_wrong
            final_correct = ~final_wrong

            # ==========================================
            # Overall error
            # ==========================================

            total_pixels += (
                gt_bin.numel()
            )

            semantic_error_pixels += int(
                sem_wrong.sum().item()
            )

            final_error_pixels += int(
                final_wrong.sum().item()
            )

            # ==========================================
            # Conflict-region error
            # ==========================================

            conflict_pixels += int(
                conflict_mask.sum().item()
            )

            conflict_semantic_error += int(
                (
                    sem_wrong
                    & conflict_mask
                )
                .sum()
                .item()
            )

            conflict_final_error += int(
                (
                    final_wrong
                    & conflict_mask
                )
                .sum()
                .item()
            )

            # ==========================================
            # True recovery / damage analysis
            #
            # True recovery:
            # Semantic wrong -> Final correct
            #
            # True damage:
            # Semantic correct -> Final wrong
            #
            # 둘 다 동일한 conflict 영역에서 계산
            # ==========================================

            true_recovered = (
                sem_wrong
                & final_correct
                & conflict_mask
            )

            true_damaged = (
                sem_correct
                & final_wrong
                & conflict_mask
            )

            true_recovered_pixels += int(
                true_recovered
                .sum()
                .item()
            )

            true_damaged_pixels += int(
                true_damaged
                .sum()
                .item()
            )

            # conflict 영역에서
            # Semantic이 원래 맞았던 pixel 수
            sem_correct_conf = (
                sem_correct
                & conflict_mask
            )

            semantic_correct_conflict += int(
                sem_correct_conf
                .sum()
                .item()
            )

            # ==========================================
            # Structural-assisted recovery
            #
            # Semantic wrong
            # Structural correct
            # -> Structural branch가 보완할 수 있는 영역
            # ==========================================

            structural_recoverable = (
                sem_wrong
                & str_correct
                & conflict_mask
            )

            structural_assisted_recovery = (
                structural_recoverable
                & final_correct
            )

            structural_recoverable_pixels += int(
                structural_recoverable
                .sum()
                .item()
            )

            structural_assisted_recovered_pixels += int(
                structural_assisted_recovery
                .sum()
                .item()
            )
           
            # ==========================================
            # Branch win inside conflict
            # ==========================================

            semantic_win += int(
                (
                    choose_semantic
                    & conflict_mask
                    & ~tie
                )
                .sum()
                .item()
            )

            structural_win += int(
                (
                    choose_structural
                    & conflict_mask
                    & ~tie
                )
                .sum()
                .item()
            )

            ties += int(
                (
                    tie
                    & conflict_mask
                )
                .sum()
                .item()
            )

            correction_abs.append(
                out["correction"]
                .abs()
                .mean()
                .item()
            )

    # ==================================================
    # Aggregate
    # ==================================================

    semantic_dice = float(
        np.mean(
            semantic_dices
        )
    )

    structural_dice = float(
        np.mean(
            structural_dices
        )
    )

    average_dice = float(
        np.mean(
            average_dices
        )
    )

    final_dice = float(
        np.mean(
            final_dices
        )
    )

    oracle_dice = float(
        np.mean(
            oracle_dices
        )
    )

    gain_vs_semantic = (
        final_dice
        - semantic_dice
    )

    oracle_gap = (
        oracle_dice
        - semantic_dice
    )

    if oracle_gap > 0:
        oracle_gap_closed = (
            gain_vs_semantic
            / oracle_gap
        )
    else:
        oracle_gap_closed = None

    semantic_error_rate = (
        semantic_error_pixels
        / total_pixels
    )

    final_error_rate = (
        final_error_pixels
        / total_pixels
    )

    if conflict_pixels > 0:
        conflict_sem_error_rate = (
            conflict_semantic_error
            / conflict_pixels
        )

        conflict_final_error_rate = (
            conflict_final_error
            / conflict_pixels
        )
    else:
        conflict_sem_error_rate = None
        conflict_final_error_rate = None

    true_recovery_rate = (
        true_recovered_pixels
        / conflict_semantic_error
        if conflict_semantic_error > 0
        else None
    )

    true_damage_rate = (
        true_damaged_pixels
        / semantic_correct_conflict
        if semantic_correct_conflict > 0
        else None
    )

    structural_assisted_recovery_rate = (
        structural_assisted_recovered_pixels
        / structural_recoverable_pixels
        if structural_recoverable_pixels > 0
        else None
    )

    total_wins = (
        semantic_win
        + structural_win
        + ties
    )

    result = {
        "n_images":
            len(ds),

        "interaction_epoch":
            interaction_epoch,

        "backbone_epoch":
            backbone_ck.get(
                "epoch"
            ),

        "topk_ratio":
            topk_ratio,

        "semantic_dice":
            semantic_dice,

        "structural_dice":
            structural_dice,

        "naive_average_dice":
            average_dice,

        "final_dice":
            final_dice,

        "oracle_dice":
            oracle_dice,

        "final_gain_vs_semantic":
            gain_vs_semantic,

        "oracle_gap_from_semantic":
            oracle_gap,

        "oracle_gap_closed_fraction":
            oracle_gap_closed,

        "mean_abs_correction":
            float(
                np.mean(
                    correction_abs
                )
            ),

        "semantic_error_rate":
            semantic_error_rate,

        "final_error_rate":
            final_error_rate,

        "overall_error_rate_reduction":
            (
                semantic_error_rate
                - final_error_rate
            ),

        "conflict_pixel_fraction":
            (
                conflict_pixels
                / total_pixels
            ),

        "conflict_semantic_error_rate":
            conflict_sem_error_rate,

        "conflict_final_error_rate":
            conflict_final_error_rate,

        "conflict_error_rate_reduction":
            (
                conflict_sem_error_rate
                - conflict_final_error_rate
                if (
                    conflict_sem_error_rate
                    is not None
                    and conflict_final_error_rate
                    is not None
                )
                else None
            ),

        "true_recovered_pixels":
            true_recovered_pixels,

        "true_damaged_pixels":
            true_damaged_pixels,

        "net_corrected_pixels":
            (
                true_recovered_pixels
                - true_damaged_pixels
            ),

        "true_recovery_rate":
            true_recovery_rate,

        "true_damage_rate":
            true_damage_rate,

        "structural_recoverable_pixels":
            structural_recoverable_pixels,

        "structural_assisted_recovered_pixels":
            structural_assisted_recovered_pixels,

        "structural_assisted_recovery_rate":
            structural_assisted_recovery_rate,

        "conflict_semantic_win_rate":
            (
                semantic_win
                / total_wins
                if total_wins > 0
                else None
            ),

        "conflict_structural_win_rate":
            (
                structural_win
                / total_wins
                if total_wins > 0
                else None
            ),

        "conflict_tie_rate":
            (
                ties
                / total_wins
                if total_wins > 0
                else None
            ),
    }

    # ==================================================
    # Output
    # ==================================================

    if args.output is None:
        out_path = (
            Path(
                args.interaction_checkpoint
            )
            .parent
            / "test_metrics.json"
        )
    else:
        out_path = Path(
            args.output
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 74)
    print(
        "CRSS-SAM Test Result"
    )
    print("=" * 74)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("=" * 74)
    print(
        "saved:",
        out_path,
    )


if __name__ == "__main__":
    main()