from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path("experiments")

TARGETS = {
    "Random 10%": [
        "random_top10",
        "random10",
        "random_10",
        "conflict_random",
    ],
    "No-Interaction": [
        "no_interaction",
        "no-interaction",
        "nointeraction",
        "without_interaction",
        "conflict_no_interaction",
    ],
    "Simple-ViLMedSAM": [
        "simplevilmedsam",
        "simple-vilmedsam",
        "simple_vilmedsam",
        "vilmedsam",
    ],
}


INTERESTING_KEYS = [
    "epoch",
    "interaction_epoch",
    "backbone_epoch",
    "topk_ratio",
    "semantic_dice",
    "structural_dice",
    "naive_average_dice",
    "final_dice",
    "foreground_dice",
    "foreground_iou",
    "val_semantic_dice",
    "val_structural_dice",
    "val_naive_average_dice",
    "val_final_dice",
    "val_gain_vs_semantic",
    "final_gain_vs_semantic",
    "oracle_dice",
    "mean_abs_correction",
    "train_loss",
]


def print_scalar_dict(d: dict, indent="    "):
    found = False
    for key in INTERESTING_KEYS:
        if key in d:
            print(f"{indent}{key}: {d[key]}")
            found = True
    return found


def safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": str(e)}


def print_json_metrics(path: Path):
    data = safe_json(path)

    if isinstance(data, dict) and "__error__" in data:
        print(f"    [JSON read error] {data['__error__']}")
        return

    # 일반 metrics JSON
    if isinstance(data, dict):
        if print_scalar_dict(data):
            return

        print("    Top-level scalar values:")
        count = 0
        for k, v in data.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                print(f"      {k}: {v}")
                count += 1
                if count >= 20:
                    break
        return

    # history.json처럼 list인 경우
    if isinstance(data, list):
        print(f"    JSON list length: {len(data)}")

        if not data:
            return

        # best val_final_dice entry
        dict_rows = [x for x in data if isinstance(x, dict)]

        if dict_rows:
            rows_with_val = [
                x for x in dict_rows
                if isinstance(x.get("val_final_dice"), (int, float))
            ]

            if rows_with_val:
                best = max(
                    rows_with_val,
                    key=lambda x: x["val_final_dice"],
                )
                print("    Best history entry by val_final_dice:")
                print_scalar_dict(best, indent="      ")

            print("    Last history entry:")
            print_scalar_dict(dict_rows[-1], indent="      ")
        return

    print(f"    Unsupported JSON type: {type(data).__name__}")


def print_checkpoint_summary(path: Path):
    if torch is None:
        print("    torch import 실패 -> checkpoint 내부는 생략")
        return

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"    [checkpoint load error] {e}")
        return

    if not isinstance(ckpt, dict):
        print(f"    checkpoint type: {type(ckpt).__name__}")
        return

    print(f"    keys: {sorted(ckpt.keys())}")

    for key in [
        "epoch",
        "best_score",
        "best_final_dice",
        "val_final_dice",
        "val_semantic_dice",
        "val_structural_dice",
        "val_naive_average_dice",
        "topk_ratio",
        "backbone_checkpoint",
    ]:
        if key in ckpt:
            print(f"    {key}: {ckpt[key]}")

    # train_crss_conflict 계열은 metrics 안에 val 수치가 들어있는 경우가 있음
    metrics = ckpt.get("metrics")
    if isinstance(metrics, dict):
        print("    checkpoint.metrics:")
        if not print_scalar_dict(metrics, indent="      "):
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    print(f"      {k}: {v}")

    args = ckpt.get("args")
    if isinstance(args, dict):
        for key in [
            "topk_ratio",
            "seed",
            "lr",
            "epochs",
            "batch_size",
            "accumulation_steps",
            "work_dir",
        ]:
            if key in args:
                print(f"    args.{key}: {args[key]}")


def candidate_dirs():
    if not ROOT.exists():
        return []

    dirs = []
    for p in ROOT.rglob("*"):
        if p.is_dir():
            low = str(p).lower()
            if any(
                keyword in low
                for keywords in TARGETS.values()
                for keyword in keywords
            ):
                dirs.append(p)

    return sorted(
        set(dirs),
        key=lambda p: (len(p.parts), str(p)),
    )


def target_matches(label: str, path: Path):
    low = str(path).lower()
    return any(
        keyword in low
        for keyword in TARGETS[label]
    )


def summarize_dir(path: Path):
    print(f"\n  Directory: {path}")

    important_files = []

    for name in [
        "test_metrics.json",
        "benchmark_test.json",
        "teacher_quality.json",
        "best.pth",
        "latest.pth",
        "history.json",
    ]:
        p = path / name
        if p.exists():
            important_files.append(p)

    for p in sorted(path.glob("*.json")):
        if p not in important_files:
            important_files.append(p)

    if not important_files:
        print("    -> 결과 파일 없음")
        return

    has_test = (path / "test_metrics.json").exists()
    has_best = (path / "best.pth").exists()

    print(
        "    STATUS:",
        f"trained={'YES' if has_best else 'NO'},",
        f"test_evaluated={'YES' if has_test else 'NO'}",
    )

    for p in important_files:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"\n    File: {p.name} ({size_mb:.2f} MB)")

        if p.suffix.lower() == ".json":
            print_json_metrics(p)

        elif p.suffix.lower() == ".pth":
            if p.name == "best.pth":
                print_checkpoint_summary(p)
            else:
                print("    latest.pth 존재 확인")


def main():
    print("=" * 80)
    print("CRSS-SAM Missing Experiment Checker v2")
    print("=" * 80)
    print(f"Search root: {ROOT.resolve()}")

    dirs = candidate_dirs()

    for label in TARGETS:
        print("\n" + "=" * 80)
        print(f"[{label}]")
        print("=" * 80)

        matched = [
            p for p in dirs
            if target_matches(label, p)
        ]

        if not matched:
            print("  -> 관련 experiment directory를 찾지 못했습니다.")
            continue

        for p in matched:
            summarize_dir(p)

    print("\n" + "=" * 80)
    print("최근 결과 디렉터리 20개")
    print("=" * 80)

    all_result_dirs = []

    if ROOT.exists():
        for p in ROOT.rglob("*"):
            if not p.is_dir():
                continue

            has_result = any(
                (p / name).exists()
                for name in [
                    "best.pth",
                    "latest.pth",
                    "test_metrics.json",
                    "benchmark_test.json",
                ]
            )

            if not has_result:
                continue

            files = [
                f for f in p.iterdir()
                if f.is_file()
            ]

            if not files:
                continue

            mtime = max(
                f.stat().st_mtime
                for f in files
            )

            all_result_dirs.append(
                (mtime, p)
            )

    for _, p in sorted(
        all_result_dirs,
        reverse=True,
    )[:20]:
        print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
