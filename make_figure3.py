from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/home/prml/StudentsWork/Jihoon/CRSS-SAM")

TEST_ROOT = ROOT / "data/Kvasir/test"
IMAGE_DIR = TEST_ROOT / "images"
GT_DIR = TEST_ROOT / "labels"

OUT_ROOT = ROOT / "qualitative_outputs"

PRED_DIRS = {
    "Simple-ViLMedSAM": OUT_ROOT / "simple_vilmedsam",
    "PSF-SAM": OUT_ROOT / "psf_sam",
    "Semantic": OUT_ROOT / "semantic",
    "CRSS-SAM": OUT_ROOT / "crss",
}

FIG_PATH = OUT_ROOT / "figure3.png"
CSV_PATH = OUT_ROOT / "per_image_dice.csv"
SELECTED_PATH = OUT_ROOT / "figure3_selected_cases.txt"

EVAL_SIZE = 256

IMG_EXTS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}


# ---------------------------------------------------------
# Final qualitative cases
# ---------------------------------------------------------

SELECTED_STEMS = {
    "Improvement": "cju33belnbyhm0878yxl42233",
    "Typical": "cju42dwedlmk60871jbgu4ehi",
    "Failure": "cju1dg44i4z3w0801nyz4p6zf",
}


def index_by_stem(folder):
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
    }


def resize_longest_and_pad(
    arr,
    size=256,
    is_mask=False,
):
    h, w = arr.shape[:2]

    scale = size / max(h, w)

    nh = int(round(h * scale))
    nw = int(round(w * scale))

    interp = (
        cv2.INTER_NEAREST
        if is_mask
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        arr,
        (nw, nh),
        interpolation=interp,
    )

    if arr.ndim == 3:
        out = np.zeros(
            (size, size, 3),
            dtype=np.uint8,
        )
        out[:nh, :nw] = resized
    else:
        out = np.zeros(
            (size, size),
            dtype=np.uint8,
        )
        out[:nh, :nw] = resized

    return out


def read_mask(path):
    x = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if x is None:
        raise RuntimeError(
            f"Failed to read mask: {path}"
        )

    if x.shape != (
        EVAL_SIZE,
        EVAL_SIZE,
    ):
        x = cv2.resize(
            x,
            (EVAL_SIZE, EVAL_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

    return x >= 127


def dice(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt,
    ).sum()

    denom = (
        pred.sum()
        + gt.sum()
    )

    return (
        2.0 * inter + eps
    ) / (
        denom + eps
    )


# ---------------------------------------------------------
# ROI selection
# ---------------------------------------------------------

def make_square_roi(
    mask,
    pad_ratio=0.35,
    min_side=120,
    max_side=235,
):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return 0, EVAL_SIZE, 0, EVAL_SIZE

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    side = max(width, height)

    side = int(
        side * (1.0 + 2.0 * pad_ratio)
    )

    side = max(side, min_side)
    side = min(side, max_side)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))

    x0 = max(0, min(
        x0,
        EVAL_SIZE - side,
    ))

    y0 = max(0, min(
        y0,
        EVAL_SIZE - side,
    ))

    x1 = x0 + side
    y1 = y0 + side

    return y0, y1, x0, x1


def determine_roi(
    case_name,
    gt,
    masks,
):
    if case_name == "Improvement":
        # Focus on regions relevant to the
        # semantic-to-CRSS correction.
        roi_mask = (
            gt
            | masks["Semantic"]
            | masks["CRSS-SAM"]
            | masks["PSF-SAM"]
        )

        return make_square_roi(
            roi_mask,
            pad_ratio=0.25,
            min_side=125,
            max_side=210,
        )

    if case_name == "Typical":
        # GT-centered crop is sufficient
        # because predictions almost overlap.
        roi_mask = gt

        return make_square_roi(
            roi_mask,
            pad_ratio=0.40,
            min_side=135,
            max_side=205,
        )

    # Failure:
    # include all predictions so false positives
    # and false negatives remain visible.
    roi_mask = gt.copy()

    for pred in masks.values():
        roi_mask |= pred

    return make_square_roi(
        roi_mask,
        pad_ratio=0.10,
        min_side=155,
        max_side=235,
    )


def crop_and_resize(
    x,
    roi,
    is_mask=False,
):
    y0, y1, x0, x1 = roi

    crop = x[
        y0:y1,
        x0:x1
    ]

    interp = (
        cv2.INTER_NEAREST
        if is_mask
        else cv2.INTER_LINEAR
    )

    # OpenCV resize does not support bool arrays.
    if is_mask:
        crop = crop.astype(np.uint8)

    crop = cv2.resize(
        crop,
        (EVAL_SIZE, EVAL_SIZE),
        interpolation=interp,
    )

    if is_mask:
        return crop > 0

    return crop


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def contour_overlay(
    image,
    gt,
    pred=None,
):
    vis = image.copy()

    gt_u8 = (
        gt.astype(np.uint8)
        * 255
    )

    gt_contours, _ = cv2.findContours(
        gt_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Green = Ground Truth
    cv2.drawContours(
        vis,
        gt_contours,
        -1,
        (0, 255, 0),
        3,
        lineType=cv2.LINE_AA,
    )

    if pred is not None:
        pred_u8 = (
            pred.astype(np.uint8)
            * 255
        )

        pred_contours, _ = cv2.findContours(
            pred_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Red = Prediction
        cv2.drawContours(
            vis,
            pred_contours,
            -1,
            (255, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )

    return vis


def main():

    images = index_by_stem(
        IMAGE_DIR
    )

    labels = index_by_stem(
        GT_DIR
    )

    pred_indices = {
        name: index_by_stem(path)
        for name, path
        in PRED_DIRS.items()
    }

    common = (
        set(images)
        & set(labels)
    )

    for idx in pred_indices.values():
        common &= set(idx)

    common = sorted(common)

    print(
        f"Common test samples: {len(common)}"
    )

    rows = []

    # -----------------------------------------------------
    # Calculate per-image Dice
    # -----------------------------------------------------

    for stem in common:

        gt_original = cv2.imread(
            str(labels[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        gt = (
            resize_longest_and_pad(
                gt_original,
                EVAL_SIZE,
                is_mask=True,
            )
            >= 127
        )

        masks = {
            name: read_mask(
                pred_indices[name][stem]
            )
            for name in PRED_DIRS
        }

        row = {
            "stem": stem,
            "simple_dice": dice(
                masks["Simple-ViLMedSAM"],
                gt,
            ),
            "psf_dice": dice(
                masks["PSF-SAM"],
                gt,
            ),
            "semantic_dice": dice(
                masks["Semantic"],
                gt,
            ),
            "crss_dice": dice(
                masks["CRSS-SAM"],
                gt,
            ),
        }

        row["crss_minus_semantic"] = (
            row["crss_dice"]
            - row["semantic_dice"]
        )

        row["crss_minus_psf"] = (
            row["crss_dice"]
            - row["psf_dice"]
        )

        rows.append(row)

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    row_lookup = {
        row["stem"]: row
        for row in rows
    }

    selected = [
        (
            case_name,
            row_lookup[stem],
        )
        for case_name, stem
        in SELECTED_STEMS.items()
    ]

    # -----------------------------------------------------
    # Print selected cases
    # -----------------------------------------------------

    lines = []

    for case_name, row in selected:

        line = (
            f"{case_name}: "
            f"{row['stem']} | "
            f"Simple={row['simple_dice']:.4f}, "
            f"PSF={row['psf_dice']:.4f}, "
            f"Semantic={row['semantic_dice']:.4f}, "
            f"CRSS={row['crss_dice']:.4f}, "
            f"CRSS-Semantic="
            f"{row['crss_minus_semantic']:+.4f}"
        )

        print(line)
        lines.append(line)

    SELECTED_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Overall sanity check
    # -----------------------------------------------------

    print()
    print("Mean Dice from saved prediction masks:")

    print(
        f"  Simple:   "
        f"{np.mean([x['simple_dice'] for x in rows]):.4f}"
    )

    print(
        f"  PSF:      "
        f"{np.mean([x['psf_dice'] for x in rows]):.4f}"
    )

    print(
        f"  Semantic: "
        f"{np.mean([x['semantic_dice'] for x in rows]):.4f}"
    )

    print(
        f"  CRSS:     "
        f"{np.mean([x['crss_dice'] for x in rows]):.4f}"
    )

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    columns = [
        "Input",
        "Ground Truth",
        "Simple-ViLMedSAM",
        "PSF-SAM",
        "Semantic",
        "CRSS-SAM (Ours)",
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(15, 8.3),
    )

    model_info = [
        (
            "Simple-ViLMedSAM",
            "simple_dice",
        ),
        (
            "PSF-SAM",
            "psf_dice",
        ),
        (
            "Semantic",
            "semantic_dice",
        ),
        (
            "CRSS-SAM",
            "crss_dice",
        ),
    ]

    for row_idx, (
        case_name,
        record,
    ) in enumerate(selected):

        stem = record["stem"]

        # -------------------------------------------------
        # Image
        # -------------------------------------------------

        image_bgr = cv2.imread(
            str(images[stem]),
            cv2.IMREAD_COLOR,
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_rgb = resize_longest_and_pad(
            image_rgb,
            EVAL_SIZE,
            is_mask=False,
        )

        # -------------------------------------------------
        # GT
        # -------------------------------------------------

        gt_original = cv2.imread(
            str(labels[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        gt = (
            resize_longest_and_pad(
                gt_original,
                EVAL_SIZE,
                is_mask=True,
            )
            >= 127
        )

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        masks = {
            name: read_mask(
                pred_indices[name][stem]
            )
            for name in PRED_DIRS
        }

        # Same ROI is used for every method
        # in this row.
        roi = determine_roi(
            case_name,
            gt,
            masks,
        )

        image_crop = crop_and_resize(
            image_rgb,
            roi,
            is_mask=False,
        )

        gt_crop = crop_and_resize(
            gt,
            roi,
            is_mask=True,
        )

        masks_crop = {
            name: crop_and_resize(
                mask,
                roi,
                is_mask=True,
            )
            for name, mask
            in masks.items()
        }

        # -------------------------------------------------
        # Input
        # -------------------------------------------------

        axes[row_idx, 0].imshow(
            image_crop
        )

        # -------------------------------------------------
        # GT
        # -------------------------------------------------

        axes[row_idx, 1].imshow(
            contour_overlay(
                image_crop,
                gt_crop,
            )
        )

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        for col_idx, (
            method,
            dice_key,
        ) in enumerate(
            model_info,
            start=2,
        ):

            axes[
                row_idx,
                col_idx
            ].imshow(
                contour_overlay(
                    image_crop,
                    gt_crop,
                    masks_crop[method],
                )
            )

            axes[
                row_idx,
                col_idx
            ].text(
                0.5,
                -0.055,
                f"Dice = {record[dice_key]:.3f}",
                transform=axes[
                    row_idx,
                    col_idx
                ].transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight=(
                    "bold"
                    if method == "CRSS-SAM"
                    else "normal"
                ),
            )

        # -------------------------------------------------
        # Row label
        # -------------------------------------------------

        axes[
            row_idx,
            0
        ].set_ylabel(
            case_name,
            fontsize=11,
            fontweight="bold",
            labelpad=7,
        )

        for col_idx in range(6):
            axes[
                row_idx,
                col_idx
            ].set_xticks([])

            axes[
                row_idx,
                col_idx
            ].set_yticks([])

            for spine in axes[
                row_idx,
                col_idx
            ].spines.values():
                spine.set_visible(False)

    # -----------------------------------------------------
    # Column titles
    # -----------------------------------------------------

    for col_idx, title in enumerate(
        columns
    ):
        axes[
            0,
            col_idx
        ].set_title(
            title,
            fontsize=10,
            fontweight="bold",
            pad=6,
        )

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    fig.text(
        0.5,
        0.012,
        "Green: Ground Truth    |    Red: Prediction    |    "
        "All methods in the same row use an identical ROI.",
        ha="center",
        fontsize=9,
    )

    plt.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.955,
        bottom=0.065,
        wspace=0.035,
        hspace=0.20,
    )

    plt.savefig(
        FIG_PATH,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close()

    print()
    print("Saved:")
    print(CSV_PATH)
    print(SELECTED_PATH)
    print(FIG_PATH)
    print()
    print(
        "Figure colors: "
        "Green = Ground Truth, "
        "Red = Prediction"
    )


if __name__ == "__main__":
    main()