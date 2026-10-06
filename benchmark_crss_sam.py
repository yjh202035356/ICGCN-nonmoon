import argparse
import json
import time
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


def binary_metrics(prob_fg, gt, threshold=0.5, eps=1e-6):
    pred = prob_fg >= threshold
    truth = gt >= 0.5

    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return dice, iou


def load_lora_weights(ext, lora_state):
    named_params = dict(ext.named_parameters())

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


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def infer_once(
    ext,
    heads,
    interaction,
    image,
    eval_size,
):
    mid, deep = ext(image)

    semantic_logits, structural_logits = heads(
        mid,
        deep,
    )

    out = interaction(
        semantic_feat=deep,
        structural_feat=mid,
        semantic_logits=semantic_logits,
        structural_logits=structural_logits,
    )

    final_logits = F.interpolate(
        out["final_logits"],
        size=(
            eval_size,
            eval_size,
        ),
        mode="bilinear",
        align_corners=False,
    )

    prob_fg = torch.sigmoid(
        final_logits
    )

    return prob_fg


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
        "--batch_size",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--eval_size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--warmup_batches",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    device = torch.device(
        args.device
    )

    # ==================================================
    # Load checkpoints
    # ==================================================

    interaction_ck = torch.load(
        args.interaction_checkpoint,
        map_location="cpu",
    )

    interaction_args = interaction_ck.get(
        "args",
        {},
    )

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

    # ==================================================
    # Dataset
    # ==================================================

    ds = KvasirTeacherDataset(
        args.data_root,
        image_size=image_size,
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ==================================================
    # Build CRSS-SAM
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
    # Parameter counts
    # ==================================================

    ext_params = sum(
        p.numel()
        for p in ext.parameters()
    )

    heads_params = sum(
        p.numel()
        for p in heads.parameters()
    )

    interaction_params = sum(
        p.numel()
        for p in interaction.parameters()
    )

    total_params = (
        ext_params
        + heads_params
        + interaction_params
    )

    lora_params = sum(
        tensor.numel()
        for tensor in backbone_ck[
            "lora"
        ].values()
    )

    task_adapted_params = (
        lora_params
        + heads_params
        + interaction_params
    )

    final_stage_trainable_params = (
        interaction_params
    )

    # ==================================================
    # Baseline GPU memory after model load
    # ==================================================

    if device.type == "cuda":
        synchronize_if_cuda(device)

        baseline_allocated_mb = (
            torch.cuda.memory_allocated(
                device
            )
            / (1024 ** 2)
        )

        baseline_reserved_mb = (
            torch.cuda.memory_reserved(
                device
            )
            / (1024 ** 2)
        )
    else:
        baseline_allocated_mb = None
        baseline_reserved_mb = None

    print("=" * 74)
    print("CRSS-SAM Benchmark")
    print("=" * 74)
    print(
        "Interaction checkpoint :",
        args.interaction_checkpoint,
    )
    print(
        "Interaction epoch      :",
        interaction_ck.get(
            "epoch",
            -1,
        ),
    )
    print(
        "Backbone checkpoint    :",
        backbone_checkpoint,
    )
    print(
        "Backbone epoch         :",
        backbone_ck.get(
            "epoch",
            -1,
        ),
    )
    print(
        "Top-K ratio            :",
        topk_ratio,
    )
    print(
        "Evaluation images      :",
        len(ds),
    )
    print(
        "Batch size             :",
        args.batch_size,
    )
    print(
        "Total params           :",
        f"{total_params:,}",
    )
    print(
        "LoRA params            :",
        f"{lora_params:,}",
    )
    print(
        "Head params            :",
        f"{heads_params:,}",
    )
    print(
        "Interaction params     :",
        f"{interaction_params:,}",
    )
    print(
        "Task-adapted params    :",
        f"{task_adapted_params:,}",
    )
    print(
        "Final-stage trainable  :",
        f"{final_stage_trainable_params:,}",
    )
    print(
        "Warmup batches         :",
        args.warmup_batches,
    )
    print("=" * 74)

    # ==================================================
    # Warm-up
    # ==================================================

    warmup_done = 0

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            warmup_output = infer_once(
                ext,
                heads,
                interaction,
                image,
                args.eval_size,
            )

            warmup_done += 1

            del warmup_output
            del image

            if warmup_done >= args.warmup_batches:
                break

    synchronize_if_cuda(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==================================================
    # Reset peak-memory stats before measured pass
    # ==================================================

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

        measurement_baseline_mb = (
            torch.cuda.memory_allocated(
                device
            )
            / (1024 ** 2)
        )
    else:
        measurement_baseline_mb = None

    # ==================================================
    # Full test evaluation + timing
    #
    # Timed region includes:
    #   SAM + LoRA image encoder
    #   semantic / structural heads
    #   conflict selection
    #   bidirectional interaction
    #   refinement
    #   resize to eval_size
    #   sigmoid
    #
    # Excludes:
    #   dataloader
    #   H2D transfer
    #   metric calculation
    # ==================================================

    per_image_dice = []
    per_image_iou = []
    per_image_latency_ms = []

    with torch.inference_mode():

        for batch in tqdm(
            loader,
            desc="crss-benchmark",
        ):
            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            gt = batch["mask"].to(
                device,
                non_blocking=True,
            )

            synchronize_if_cuda(device)

            t0 = time.perf_counter()

            prob_fg = infer_once(
                ext,
                heads,
                interaction,
                image,
                args.eval_size,
            )

            synchronize_if_cuda(device)

            elapsed_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            batch_n = image.shape[0]

            per_image_latency_ms.extend(
                [
                    elapsed_ms
                    / batch_n
                ]
                * batch_n
            )

            gt_eval = F.interpolate(
                gt,
                size=(
                    args.eval_size,
                    args.eval_size,
                ),
                mode="nearest",
            )

            for i in range(batch_n):
                d, j = binary_metrics(
                    prob_fg[
                        i:i + 1
                    ],
                    gt_eval[
                        i:i + 1
                    ],
                )

                per_image_dice.append(
                    d
                )

                per_image_iou.append(
                    j
                )

    # ==================================================
    # GPU memory
    # ==================================================

    if device.type == "cuda":
        synchronize_if_cuda(device)

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(
                device
            )
            / (1024 ** 2)
        )

        incremental_peak_mb = (
            peak_allocated_mb
            - measurement_baseline_mb
        )
    else:
        peak_allocated_mb = None
        peak_reserved_mb = None
        incremental_peak_mb = None

    # ==================================================
    # Aggregate
    # ==================================================

    latency = np.asarray(
        per_image_latency_ms,
        dtype=np.float64,
    )

    mean_latency_ms = float(
        latency.mean()
    )

    median_latency_ms = float(
        np.median(
            latency
        )
    )

    p95_latency_ms = float(
        np.percentile(
            latency,
            95,
        )
    )

    fps = float(
        1000.0
        / mean_latency_ms
    )

    result = {
        "method":
            "CRSS-SAM",

        "n_images":
            len(ds),

        "interaction_epoch":
            int(
                interaction_ck.get(
                    "epoch",
                    -1,
                )
            ),

        "backbone_epoch":
            int(
                backbone_ck.get(
                    "epoch",
                    -1,
                )
            ),

        "topk_ratio":
            topk_ratio,

        "batch_size":
            args.batch_size,

        "warmup_batches":
            args.warmup_batches,

        "foreground_dice":
            float(
                np.mean(
                    per_image_dice
                )
            ),

        "foreground_iou":
            float(
                np.mean(
                    per_image_iou
                )
            ),

        "total_params":
            int(
                total_params
            ),

        "lora_params":
            int(
                lora_params
            ),

        "head_params":
            int(
                heads_params
            ),

        "interaction_params":
            int(
                interaction_params
            ),

        "task_adapted_params_total":
            int(
                task_adapted_params
            ),

        "final_stage_trainable_params":
            int(
                final_stage_trainable_params
            ),

        "task_adapted_param_fraction":
            float(
                task_adapted_params
                / total_params
            ),

        "mean_latency_ms_per_image":
            mean_latency_ms,

        "median_latency_ms_per_image":
            median_latency_ms,

        "p95_latency_ms_per_image":
            p95_latency_ms,

        "fps":
            fps,

        "gpu_model_allocated_mb":
            (
                float(
                    baseline_allocated_mb
                )
                if baseline_allocated_mb
                is not None
                else None
            ),

        "gpu_model_reserved_mb":
            (
                float(
                    baseline_reserved_mb
                )
                if baseline_reserved_mb
                is not None
                else None
            ),

        "gpu_peak_allocated_mb":
            (
                float(
                    peak_allocated_mb
                )
                if peak_allocated_mb
                is not None
                else None
            ),

        "gpu_peak_reserved_mb":
            (
                float(
                    peak_reserved_mb
                )
                if peak_reserved_mb
                is not None
                else None
            ),

        "gpu_incremental_peak_mb":
            (
                float(
                    incremental_peak_mb
                )
                if incremental_peak_mb
                is not None
                else None
            ),

        "timing_scope":
            (
                "SAM+LoRA forward + heads + conflict interaction + "
                "refinement + resize + sigmoid; excludes dataloader, "
                "H2D transfer, and metrics"
            ),
    }

    # ==================================================
    # Save
    # ==================================================

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
    print("CRSS-SAM Benchmark Result")
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
