import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crss_dataset import KvasirTeacherDataset
from psf_sam_controlled_common import (
    build_psf_sam,
    count_parameters,
    make_records,
    outputs_to_logits,
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


@torch.inference_mode()
def infer_once(model, image):
    outputs = model(
        make_records(image),
        multimask_output=True,
    )
    logits = outputs_to_logits(outputs)
    prob_fg = torch.softmax(
        logits,
        dim=1,
    )[:, 1:2]
    return prob_fg


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--repo_root", default="baselines/PSF-SAM")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup_batches", type=int, default=10)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()
    device = torch.device(args.device)

    ds = KvasirTeacherDataset(
        args.data_root,
        image_size=1024,
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_psf_sam(
        args.sam_checkpoint,
        args.repo_root,
        args.device,
    )

    ckpt = torch.load(
        args.model_checkpoint,
        map_location="cpu",
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    total_params, trainable_params = count_parameters(model)

    if device.type == "cuda":
        synchronize_if_cuda(device)
        baseline_allocated_mb = (
            torch.cuda.memory_allocated(device) / (1024 ** 2)
        )
        baseline_reserved_mb = (
            torch.cuda.memory_reserved(device) / (1024 ** 2)
        )
    else:
        baseline_allocated_mb = None
        baseline_reserved_mb = None

    print("=" * 74)
    print("PSF-SAM Controlled Benchmark")
    print("=" * 74)
    print("Checkpoint epoch      :", ckpt.get("epoch", -1))
    print("Evaluation images     :", len(ds))
    print("Batch size            :", args.batch_size)
    print("Total params          :", f"{total_params:,}")
    print("Trainable params      :", f"{trainable_params:,}")
    print("Warmup batches        :", args.warmup_batches)
    print("=" * 74)

    warmup_done = 0

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            warmup_output = infer_once(
                model,
                image,
            )

            warmup_done += 1

            del warmup_output
            del image

            if warmup_done >= args.warmup_batches:
                break

    synchronize_if_cuda(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        measurement_baseline_mb = (
            torch.cuda.memory_allocated(device) / (1024 ** 2)
        )
    else:
        measurement_baseline_mb = None

    per_image_dice = []
    per_image_iou = []
    per_image_latency_ms = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="psf-benchmark",
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
                model,
                image,
            )

            synchronize_if_cuda(device)
            elapsed_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            batch_n = image.shape[0]

            per_image_latency_ms.extend(
                [elapsed_ms / batch_n] * batch_n
            )

            for i in range(batch_n):
                d, j = binary_metrics(
                    prob_fg[i:i + 1],
                    gt[i:i + 1],
                )
                per_image_dice.append(d)
                per_image_iou.append(j)

    if device.type == "cuda":
        synchronize_if_cuda(device)

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )

        incremental_peak_mb = (
            peak_allocated_mb
            - measurement_baseline_mb
        )
    else:
        peak_allocated_mb = None
        peak_reserved_mb = None
        incremental_peak_mb = None

    latency = np.asarray(
        per_image_latency_ms,
        dtype=np.float64,
    )

    mean_latency_ms = float(latency.mean())
    median_latency_ms = float(np.median(latency))
    p95_latency_ms = float(np.percentile(latency, 95))
    fps = float(1000.0 / mean_latency_ms)

    result = {
        "method": "PSF-SAM-controlled",
        "n_images": len(ds),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "foreground_dice": float(np.mean(per_image_dice)),
        "foreground_iou": float(np.mean(per_image_iou)),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_param_fraction": float(
            trainable_params / total_params
        ),
        "mean_latency_ms_per_image": mean_latency_ms,
        "median_latency_ms_per_image": median_latency_ms,
        "p95_latency_ms_per_image": p95_latency_ms,
        "fps": fps,
        "gpu_model_allocated_mb": (
            float(baseline_allocated_mb)
            if baseline_allocated_mb is not None
            else None
        ),
        "gpu_model_reserved_mb": (
            float(baseline_reserved_mb)
            if baseline_reserved_mb is not None
            else None
        ),
        "gpu_peak_allocated_mb": (
            float(peak_allocated_mb)
            if peak_allocated_mb is not None
            else None
        ),
        "gpu_peak_reserved_mb": (
            float(peak_reserved_mb)
            if peak_reserved_mb is not None
            else None
        ),
        "gpu_incremental_peak_mb": (
            float(incremental_peak_mb)
            if incremental_peak_mb is not None
            else None
        ),
        "timing_scope": (
            "model forward + outputs_to_logits + softmax; "
            "excludes dataloader, H2D transfer, and metrics"
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 74)
    print("PSF-SAM Benchmark Result")
    print("=" * 74)
    print(json.dumps(result, indent=2))
    print("=" * 74)
    print("saved:", out)


if __name__ == "__main__":
    main()
