"""Prepare or audit the canonical 1450/798 split. Does not resplit or edit data."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEST_COUNTS = {"Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
               "ETIS-LaribPolypDB": 196, "CVC-300": 60}


def index(folder):
    entries = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in EXTENSIONS:
            if path.stem in entries:
                raise RuntimeError(f"Duplicate stem: {path}")
            entries[path.stem] = path
    return entries


def folder(root, names):
    found = [Path(root) / name for name in names if (Path(root) / name).is_dir()]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one of {names} in {root}; found {found}")
    return found[0]


def paired(root, count):
    images = index(folder(root, ["images", "image"]))
    masks = index(folder(root, ["masks", "labels"]))
    if set(images) != set(masks) or len(images) != count:
        raise RuntimeError(f"Invalid pairs at {root}: images={len(images)}, masks={len(masks)}, expected={count}")
    return images, masks


def decoded_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def check_pairs(root, count, teacher=False):
    images, masks = paired(root, count)
    maps = index(Path(root) / "attribution_map") if teacher else {}
    if teacher and set(maps) != set(images):
        missing = sorted(set(images) - set(maps))
        raise RuntimeError(f"Incomplete teacher maps: {len(maps)}/{count}; missing={missing[:8]}")
    records = []
    for stem in sorted(images):
        with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
            image.load(); mask.load()
            if image.size != mask.size:
                raise RuntimeError(f"Image/mask dimension mismatch: {stem}")
            size = image.size
            if not (np.asarray(mask.convert("L")) > 127).any():
                raise RuntimeError(f"Empty polyp ground truth: {stem}")
        if teacher:
            with Image.open(maps[stem]) as attribution:
                attribution.load()
                values = np.asarray(attribution.convert("L"))
                if attribution.size != size or values.max() == values.min():
                    raise RuntimeError(f"Wrong-sized or constant teacher map: {maps[stem]}")
        records.append({"stem": stem, "image": str(images[stem].resolve()),
                        "mask": str(masks[stem].resolve()), "size": size,
                        "rgb_sha256": decoded_hash(images[stem])})
    return records


def link_directory(target, source):
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {target}")
    elif target.exists():
        # Existing data are left in place; callers validate their contents separately.
        return
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="dataset/TrainDataset")
    ap.add_argument("--train-root", default="data/Polyp1450/train")
    ap.add_argument("--test-root")
    ap.add_argument("--require-teacher", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--output", default="experiments/polyp1450/data_audit.json")
    args = ap.parse_args()
    train_root = Path(args.train_root)
    if not args.audit_only:
        source = Path(args.source)
        source_images, source_masks = paired(source, 1450)
        train_root.mkdir(parents=True, exist_ok=True)
        link_directory(train_root / "images", folder(source, ["images", "image"]))
        link_directory(train_root / "labels", folder(source, ["masks", "labels"]))
        for name in ["attribution_map", "attribution_overlay"]:
            (train_root / name).mkdir(exist_ok=True)
        dst_images, dst_masks = paired(train_root, 1450)
        for original, linked in [(source_images, dst_images), (source_masks, dst_masks)]:
            for stem in original:
                if original[stem].read_bytes() != linked[stem].read_bytes():
                    raise RuntimeError(f"Existing destination differs from canonical source: {stem}")
    train = check_pairs(train_root, 1450, args.require_teacher)
    clinic = sum(record["stem"].isdigit() for record in train)
    if clinic != 550:
        raise RuntimeError(f"Expected 550 numeric ClinicDB stems, got {clinic}")
    report = {"status": "PASS", "train_count": len(train), "Kvasir_train": 900,
              "ClinicDB_train": 550, "teacher_checked": args.require_teacher,
              "train": train, "test": {}, "exact_train_test_overlap": None}
    if args.test_root:
        train_hashes = {record["rgb_sha256"] for record in train}
        overlap = []
        for name, count in TEST_COUNTS.items():
            records = check_pairs(Path(args.test_root) / name, count)
            report["test"][name] = records
            overlap.extend(f"{name}/{record['stem']}" for record in records if record["rgb_sha256"] in train_hashes)
        if overlap:
            raise RuntimeError(f"Train/test decoded-RGB overlap: {overlap[:8]}")
        report["exact_train_test_overlap"] = 0
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PASS: train=1450 (Kvasir=900, ClinicDB=550), teacher_checked={args.require_teacher}")
    if args.test_root:
        print("PASS: test=798, exact train/test overlap=0")
    print(f"Audit: {out}")


if __name__ == "__main__":
    main()
