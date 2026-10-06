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

from crss_model import (
    FrozenSAMFeatureExtractor,
    DualCueHeads,
    seg_loss,
    structural_boundary_loss,
)

from crss_teacher_refine_v2 import (
    SAMAffinityTeacherRefiner,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(
    prob,
    target,
    threshold=0.5,
    eps=1e-6,
):
    pred = (
        prob >= threshold
    ).float()

    inter = (
        pred * target
    ).sum(
        dim=(1, 2, 3)
    )

    denom = (
        pred.sum(
            dim=(1, 2, 3)
        )
        +
        target.sum(
            dim=(1, 2, 3)
        )
    )

    return (
        (
            2 * inter
            + eps
        )
        /
        (
            denom
            + eps
        )
    ).mean().item()


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

    sem_ds = []
    str_ds = []
    avg_ds = []

    for step, b in enumerate(
        loader,
        1,
    ):
        image = b["image"].to(
            device
        )

        gt = b["mask"].to(
            device
        )

        mid, deep = ext(
            image
        )

        sem_l, str_l = heads(
            mid,
            deep,
        )

        sem_l = F.interpolate(
            sem_l,
            (
                loss_size,
                loss_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        str_l = F.interpolate(
            str_l,
            (
                loss_size,
                loss_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        gt = F.interpolate(
            gt,
            (
                loss_size,
                loss_size,
            ),
            mode="nearest",
        )

        psem = torch.sigmoid(
            sem_l
        )

        pstr = torch.sigmoid(
            str_l
        )

        pavg = 0.5 * (
            psem + pstr
        )

        sem_ds.append(
            dice_score(
                psem,
                gt,
            )
        )

        str_ds.append(
            dice_score(
                pstr,
                gt,
            )
        )

        avg_ds.append(
            dice_score(
                pavg,
                gt,
            )
        )

        if (
            max_batches > 0
            and step >= max_batches
        ):
            break

    return {
        "semantic_dice":
            float(
                np.mean(sem_ds)
            ),

        "structural_dice":
            float(
                np.mean(str_ds)
            ),

        "naive_avg_dice":
            float(
                np.mean(avg_ds)
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train_root",
        required=True,
    )

    ap.add_argument(
        "--resume",
        type=str,
        default="",
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
        "--model_type",
        default="vit_h",
        choices=[
            "vit_h",
            "vit_l",
            "vit_b",
        ],
    )

    ap.add_argument(
        "--mid_block",
        type=int,
        default=15,
    )

    ap.add_argument(
        "--work_dir",
        default=(
            "experiments/kvasir/"
            "seed123/teacher_refine"
        ),
    )

    # E0와 동일
    ap.add_argument(
        "--epochs",
        type=int,
        default=60,
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
        "--num_workers",
        type=int,
        default=4,
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

    # ------------------------------------------
    # E0와 동일한 loss weights
    # ------------------------------------------

    ap.add_argument(
        "--lambda_sem_gt",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--lambda_distill",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda_str_gt",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda_boundary",
        type=float,
        default=1.0,
    )

    # ------------------------------------------
    # Teacher Refinement parameters
    # ------------------------------------------

    ap.add_argument(
        "--affinity_temperature",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--teacher_weight",
        type=float,
        default=0.5,
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

    # smoke test
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

    # ==========================================
    # Seed
    # ==========================================

    seed_all(
        args.seed
    )

    work = Path(
        args.work_dir
    )

    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================
    # Dataset
    # ==========================================

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

    # ==========================================
    # Frozen SAM
    # ==========================================

    sam = sam_model_registry[
        args.model_type
    ](
        checkpoint=args.sam_checkpoint
    ).to(
        args.device
    )

    ext = FrozenSAMFeatureExtractor(
        sam,
        args.mid_block,
    ).to(
        args.device
    )

    # ==========================================
    # Heads
    # ==========================================

    heads = DualCueHeads(
        ext.mid_channels,
        ext.deep_channels,
    ).to(
        args.device
    )

    # ==========================================
    # Teacher Refiner
    # ==========================================

    refiner = SAMAffinityTeacherRefiner(
        temperature=(
            args.affinity_temperature
        ),
        teacher_weight=(
            args.teacher_weight
        ),
    ).to(
        args.device
    )

    # 중요:
    # E2 역시 E0와 동일하게
    # Heads만 trainable.
    opt = torch.optim.AdamW(
        heads.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    head_trainable = sum(
        p.numel()
        for p in heads.parameters()
        if p.requires_grad
    )

    print("=" * 70)
    print(
        "E2: Frozen CRSS "
        "+ SAM-affinity Teacher Refinement"
    )
    print(
        f"mid_block             : "
        f"{args.mid_block}"
    )
    print(
        f"affinity temperature  : "
        f"{args.affinity_temperature}"
    )
    print(
        f"original teacher weight: "
        f"{args.teacher_weight}"
    )
    print(
        f"Head trainable        : "
        f"{head_trainable:,}"
    )
    print(
        "SAM trainable         : 0"
    )
    print(
        "Refiner trainable     : 0"
    )
    print("=" * 70)

    best = -1.0
    history = []
    start_epoch = 1

    # ==========================================
    # Resume
    # ==========================================

    if args.resume:

        resume_path = Path(
            args.resume
        )

        print("=" * 70)
        print(
            f"Resume checkpoint: "
            f"{resume_path}"
        )

        resume_ck = torch.load(
            resume_path,
            map_location="cpu",
        )

        # epoch 60의 Head weight 복원
        heads.load_state_dict(
            resume_ck["heads"]
        )

        start_epoch = (
            int(
                resume_ck.get(
                    "epoch",
                    0,
                )
            )
            + 1
        )

        # 기존 history 복원
        old_history_path = (
            resume_path.parent
            / "history.json"
        )

        if old_history_path.exists():

            history = json.loads(
                old_history_path.read_text(
                    encoding="utf-8"
                )
            )

            if history:

                best = max(
                    0.5
                    * (
                        x["semantic_dice"]
                        + x["structural_dice"]
                    )
                    for x in history
                )

        else:

            best = float(
                resume_ck.get(
                    "score",
                    -1.0,
                )
            )

        # 앞으로 생성되는 checkpoint에는
        # optimizer state도 저장하지만,
        # 기존 60epoch checkpoint에는 없음.
        if "optimizer" in resume_ck:

            opt.load_state_dict(
                resume_ck["optimizer"]
            )

            print(
                "Optimizer state restored."
            )

        else:

            print(
                "[WARNING] "
                "Optimizer state was not stored "
                "in the old checkpoint."
            )

            print(
                "[WARNING] "
                "Weights resume from epoch 60, "
                "but AdamW state starts fresh."
            )

        print(
            f"Resume training from epoch "
            f"{start_epoch}"
        )

        print(
            f"Previous best score: "
            f"{best:.6f}"
        )

        print("=" * 70)

    # ==========================================
    # Training
    # ==========================================

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        heads.train()

        opt.zero_grad(
            set_to_none=True
        )

        losses = []
        refine_deltas = []

        pbar = tqdm(
            trl,
            desc=(
                f"epoch "
                f"{epoch}/"
                f"{args.epochs}"
            ),
        )

        for step, b in enumerate(
            pbar,
            1,
        ):

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

            # --------------------------------------
            # Frozen SAM
            # --------------------------------------

            with torch.no_grad():

                mid, deep = ext(
                    image
                )

                refined_teacher, affinity = (
                    refiner(
                        deep,
                        teacher,
                        output_size=(
                            args.loss_size,
                            args.loss_size,
                        ),
                    )
                )

            # --------------------------------------
            # Heads
            # --------------------------------------

            sem_l, str_l = heads(
                mid,
                deep,
            )

            sem_l = F.interpolate(
                sem_l,
                (
                    args.loss_size,
                    args.loss_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            str_l = F.interpolate(
                str_l,
                (
                    args.loss_size,
                    args.loss_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            gt_s = F.interpolate(
                gt,
                (
                    args.loss_size,
                    args.loss_size,
                ),
                mode="nearest",
            )

            original_teacher = F.interpolate(
                teacher,
                (
                    args.loss_size,
                    args.loss_size,
                ),
                mode="bilinear",
                align_corners=False,
            ).clamp(
                0,
                1,
            )

            # --------------------------------------
            # Loss
            # --------------------------------------

            l_sem_gt = seg_loss(
                sem_l,
                gt_s,
            )

            # E0와 달라지는 유일한 핵심 부분
            l_dist = F.smooth_l1_loss(
                torch.sigmoid(
                    sem_l
                ),
                refined_teacher,
            )

            l_str_gt = seg_loss(
                str_l,
                gt_s,
            )

            l_bnd = (
                structural_boundary_loss(
                    str_l,
                    gt_s,
                )
            )

            loss = (
                args.lambda_sem_gt
                * l_sem_gt

                + args.lambda_distill
                * l_dist

                + args.lambda_str_gt
                * l_str_gt

                + args.lambda_boundary
                * l_bnd
            )

            (
                loss
                / args.accumulation_steps
            ).backward()

            # 얼마나 teacher가 변했는지 기록
            delta = (
                refined_teacher
                - original_teacher
            ).abs().mean().item()

            refine_deltas.append(
                delta
            )

            last_requested_batch = (
                args.max_train_batches > 0
                and
                step
                >= args.max_train_batches
            )

            should_step = (
                step
                % args.accumulation_steps
                == 0

                or step
                == len(trl)

                or last_requested_batch
            )

            if should_step:

                opt.step()

                opt.zero_grad(
                    set_to_none=True
                )

            losses.append(
                loss.item()
            )

            pbar.set_postfix(
                loss=(
                    f"{np.mean(losses[-20:]):.4f}"
                ),
                refine_delta=(
                    f"{np.mean(refine_deltas[-20:]):.4f}"
                ),
            )

            if last_requested_batch:
                break

        # ==========================================
        # Validation
        # ==========================================

        m = validate(
            ext,
            heads,
            val,
            args.device,
            args.loss_size,
            max_batches=(
                args.max_val_batches
            ),
        )

        m["epoch"] = epoch

        m["train_loss"] = float(
            np.mean(losses)
        )

        m[
            "teacher_refine_delta"
        ] = float(
            np.mean(
                refine_deltas
            )
        )

        history.append(
            m
        )

        print(
            json.dumps(
                m,
                indent=2,
            )
        )

        # E0와 동일한 checkpoint criterion
        score = 0.5 * (
            m["semantic_dice"]
            +
            m["structural_dice"]
        )

        ck = {
            "heads":
                heads.state_dict(),

            "optimizer":
                opt.state_dict(),
                
            "args":
                vars(args),

            "epoch":
                epoch,

            "metrics":
                m,

            "score":
                score,
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
                f"[best] "
                f"{best:.6f}"
            )

        (
            work
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()