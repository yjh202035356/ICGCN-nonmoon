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
from crss_model_lora import LoRASAMFeatureExtractor, DualCueHeads


def dice_binary(prob, gt, threshold=0.5, eps=1e-6):
    pred = prob >= threshold
    gt = gt >= 0.5

    inter = (pred & gt).sum().item()
    denom = pred.sum().item() + gt.sum().item()

    return (2 * inter + eps) / (denom + eps)


def masked_corr(a, b, mask):
    a = a[mask].float()
    b = b[mask].float()

    if a.numel() < 10:
        return float("nan")

    a = a - a.mean()
    b = b - b.mean()

    den = torch.sqrt(
        (a * a).sum() * (b * b).sum()
    ).clamp_min(1e-8)

    return float(
        ((a * b).sum() / den).item()
    )


def load_lora_weights(ext, lora_state):
    """
    checkpoint에 저장된 LoRA parameter만
    현재 extractor의 대응 parameter에 복원.
    """
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
            "LoRA parameters not found in model:\n"
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
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--model_type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
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
        "--output",
        default=None,
    )

    args = ap.parse_args()

    # --------------------------------------------------
    # Load checkpoint first
    # --------------------------------------------------

    ck = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    ck_args = ck.get("args", {})

    mid_block = int(
        ck_args.get("mid_block", 15)
    )

    lora_rank = int(
        ck_args.get("lora_rank", 4)
    )

    lora_alpha = float(
        ck_args.get("lora_alpha", 4.0)
    )

    image_size = int(
        ck_args.get("image_size", 1024)
    )

    # --------------------------------------------------
    # IMPORTANT:
    # old Late-LoRA checkpoints may not have
    # lora_start_block stored.
    #
    # Late-LoRA default:
    #     mid_block + 1 = 16
    #
    # Full-LoRA:
    #     lora_start_block = 0
    # --------------------------------------------------

    raw_lora_start = ck_args.get(
        "lora_start_block",
        None,
    )

    if raw_lora_start is None:
        lora_start_block = mid_block + 1
    else:
        lora_start_block = int(
            raw_lora_start
        )

    print("=" * 70)
    print("CRSS-SAM LoRA evaluation")
    print("Checkpoint           :", args.checkpoint)
    print("Checkpoint epoch     :", ck.get("epoch"))
    print("mid_block            :", mid_block)
    print("LoRA start block     :", lora_start_block)
    print("LoRA rank            :", lora_rank)
    print("LoRA alpha           :", lora_alpha)
    print("image_size           :", image_size)
    print("=" * 70)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    ds = KvasirTeacherDataset(
        args.data_root,
        image_size,
        False,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"Evaluation images    : {len(ds)}"
    )
    print("=" * 70)

    # --------------------------------------------------
    # SAM + LoRA
    # --------------------------------------------------

    sam = sam_model_registry[
        args.model_type
    ](
        checkpoint=args.sam_checkpoint
    ).to(args.device)

    ext = LoRASAMFeatureExtractor(
        sam=sam,
        mid_block=mid_block,
        lora_start_block=lora_start_block,
        rank=lora_rank,
        alpha=lora_alpha,
    ).to(args.device)

    heads = DualCueHeads(
        ext.mid_channels,
        ext.deep_channels,
    ).to(args.device)

    # --------------------------------------------------
    # Restore checkpoint
    # --------------------------------------------------

    heads.load_state_dict(
        ck["heads"]
    )

    load_lora_weights(
        ext,
        ck["lora"],
    )

    ext.sam.eval()
    heads.eval()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    semd = []
    strd = []
    avgd = []
    orad = []
    corrs = []

    semwin = 0
    strwin = 0
    ties = 0

    all_conf = []
    all_err = []

    with torch.no_grad():

        for b in tqdm(
            loader,
            desc="CRSS-LoRA-eval",
        ):
            image = b["image"].to(
                args.device,
                non_blocking=True,
            )

            gt = b["mask"].to(
                args.device,
                non_blocking=True,
            )

            # ------------------------------------------
            # SAM feature extraction
            # ------------------------------------------

            mid, deep = ext(image)

            # ------------------------------------------
            # Semantic / Structural predictions
            # ------------------------------------------

            sl, tl = heads(
                mid,
                deep,
            )

            sl = F.interpolate(
                sl,
                (
                    args.eval_size,
                    args.eval_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            tl = F.interpolate(
                tl,
                (
                    args.eval_size,
                    args.eval_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            gt = F.interpolate(
                gt,
                (
                    args.eval_size,
                    args.eval_size,
                ),
                mode="nearest",
            )

            # Semantic probability
            ps = torch.sigmoid(sl)

            # Structural probability
            pt = torch.sigmoid(tl)

            # Simple baseline fusion
            pa = 0.5 * (
                ps + pt
            )

            # ------------------------------------------
            # Pixel-wise oracle
            #
            # GT를 알고 있다고 가정하고
            # 각 pixel에서 더 정확한 branch 선택.
            #
            # 실제 inference에서는 사용할 수 없음.
            # Complementarity 분석용 upper bound.
            # ------------------------------------------

            es = (
                ps - gt
            ).abs()

            et = (
                pt - gt
            ).abs()

            choose_s = es < et
            choose_t = et < es

            tie = (
                es - et
            ).abs() <= 0.02

            po = torch.where(
                choose_s,
                ps,
                pt,
            )

            # ------------------------------------------
            # Dice
            # ------------------------------------------

            semd.append(
                dice_binary(
                    ps,
                    gt,
                )
            )

            strd.append(
                dice_binary(
                    pt,
                    gt,
                )
            )

            avgd.append(
                dice_binary(
                    pa,
                    gt,
                )
            )

            orad.append(
                dice_binary(
                    po,
                    gt,
                )
            )

            # ------------------------------------------
            # Prediction correlation
            # ------------------------------------------

            informative = (
                (gt > 0.5)
                | (ps > 0.1)
                | (pt > 0.1)
            )

            corrs.append(
                masked_corr(
                    ps,
                    pt,
                    informative,
                )
            )

            # ------------------------------------------
            # Conflict Map
            #
            # C = |P_sem - P_str|
            # ------------------------------------------

            conf = (
                ps - pt
            ).abs()

            # Current threshold-based conflict analysis
            cm = conf > 0.10

            semwin += int(
                (
                    choose_s
                    & cm
                    & ~tie
                ).sum().item()
            )

            strwin += int(
                (
                    choose_t
                    & cm
                    & ~tie
                ).sum().item()
            )

            ties += int(
                (
                    tie
                    & cm
                ).sum().item()
            )

            # ------------------------------------------
            # Error concentration
            #
            # At least one branch is wrong.
            # ------------------------------------------

            err = (
                (
                    (ps >= 0.5)
                    !=
                    (gt >= 0.5)
                )
                |
                (
                    (pt >= 0.5)
                    !=
                    (gt >= 0.5)
                )
            )

            all_conf.append(
                conf
                .detach()
                .cpu()
                .flatten()
            )

            all_err.append(
                err
                .detach()
                .cpu()
                .flatten()
                .float()
            )

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    conf = torch.cat(
        all_conf
    )

    err = torch.cat(
        all_err
    )

    total = (
        semwin
        + strwin
        + ties
    )

    sem_mean = float(
        np.mean(semd)
    )

    str_mean = float(
        np.mean(strd)
    )

    avg_mean = float(
        np.mean(avgd)
    )

    ora_mean = float(
        np.mean(orad)
    )

    best_head = max(
        sem_mean,
        str_mean,
    )

    result = {
        "n_images":
            len(ds),

        "checkpoint_epoch":
            ck.get("epoch"),

        "mid_block":
            mid_block,

        "lora_start_block":
            lora_start_block,

        "lora_rank":
            lora_rank,

        "lora_alpha":
            lora_alpha,

        "semantic_dice":
            sem_mean,

        "structural_dice":
            str_mean,

        "naive_average_dice":
            avg_mean,

        "oracle_dice":
            ora_mean,

        "oracle_gain_vs_best_head":
            ora_mean - best_head,

        "prediction_corr_informative_pixels":
            float(
                np.nanmean(corrs)
            ),

        "conflict_semantic_win_rate":
            (
                semwin / total
                if total
                else None
            ),

        "conflict_structural_win_rate":
            (
                strwin / total
                if total
                else None
            ),

        "conflict_tie_rate":
            (
                ties / total
                if total
                else None
            ),

        "overall_error_rate":
            float(
                err.mean().item()
            ),
    }

    # --------------------------------------------------
    # Top-K conflict difficulty analysis
    #
    # 높은 conflict 영역이 전체 error를
    # 얼마나 포착하는지 확인.
    # --------------------------------------------------

    order = torch.argsort(
        conf,
        descending=True,
    )

    n = conf.numel()

    total_err = err.sum().item()

    for r in (
        0.1,
        0.2,
        0.3,
    ):
        k = max(
            1,
            int(n * r),
        )

        idx = order[:k]

        result[
            f"top{int(r * 100)}_difficulty_error_rate"
        ] = float(
            err[idx].mean().item()
        )

        result[
            f"top{int(r * 100)}_difficulty_error_capture"
        ] = (
            float(
                err[idx].sum().item()
                / total_err
            )
            if total_err > 0
            else 0.0
        )

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------

    if args.output is None:
        out = (
            Path(args.checkpoint)
            .parent
            / "sanity_metrics.json"
        )
    else:
        out = Path(
            args.output
        )

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

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CRSS-SAM evaluation result")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("=" * 70)
    print(
        "saved:",
        out,
    )


if __name__ == "__main__":
    main()