"""Run the existing BiomedCLIP/IBA teacher with fixed settings and lossless caches."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import vision_heatmap_iba
from scripts.generate_map import pad_to_square, unpad_square_from_resized_array
from scripts.prepare_polyp1450 import index, decoded_hash


def model_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".py", ".bin", ".safetensors", ".txt", ".model"}:
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-path", default="data/Polyp1450/train/images")
    ap.add_argument("--output-path", default="data/Polyp1450/train/attribution_map")
    ap.add_argument("--model-path", default="checkpoints/BiomedCLIP-vit-bert-hf")
    ap.add_argument("--text", default="polyp")
    ap.add_argument("--vbeta", type=float, default=0.1)
    ap.add_argument("--vvar", type=float, default=0.1)
    ap.add_argument("--vlayer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    model_path = Path(args.model_path).resolve()
    if not (model_path / "config.json").is_file():
        ap.error(f"Local BiomedCLIP model not found: {model_path}")
    images = index(args.input_path)
    selected = sorted(images)
    if args.max_images:
        selected = selected[:args.max_images]
    elif len(selected) != 1450:
        ap.error(f"Expected 1450 images for a full run, got {len(selected)}")
    if not selected:
        ap.error("No images")
    output = Path(args.output_path); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "teacher_generation.json"
    config = {"model_path": str(model_path), "model_files_sha256": model_fingerprint(model_path),
              "text": args.text, "vbeta": args.vbeta, "vvar": args.vvar, "vlayer": args.vlayer,
              "seed": args.seed, "seed_mode": "per_image_stem_sha256", "format": "lossless_uint8_png",
              "hyperparameter_search": False, "IBA_steps": 10, "IBA_lr": 1,
              "input_root": str(Path(args.input_path).resolve()), "selected_count": len(selected)}
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    cached = index(output)
    if cached and not args.overwrite and (old is None or old.get("config") != config):
        ap.error("Existing teacher cache has no matching provenance. Audit it separately, or use a fresh output directory.")
    if set(cached) - set(selected):
        ap.error("Output contains teacher maps outside the selected dataset")
    manifest = {"config": config, "status": "incomplete", "completed": {} if old is None or args.overwrite else old.get("completed", {})}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).to(args.device).eval()
    # Keep the original CLIP fixed; IBA inserts its own trainable bottleneck.
    for parameter in model.parameters():
        parameter.requires_grad = False
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    text_ids = torch.tensor([tokenizer.encode(args.text, add_special_tokens=True)], device=args.device)
    for stem in tqdm(selected, desc="Polyp teacher"):
        source_hash = decoded_hash(images[stem])
        with Image.open(images[stem]) as source:
            original = source.convert("RGB")
        destination = output / f"{stem}.png"
        completed = manifest["completed"].get(stem)
        if not args.overwrite and destination.is_file() and completed and completed.get("rgb_sha256") == source_hash:
            with Image.open(destination) as cached_map:
                cached_map.load(); values = np.asarray(cached_map.convert("L"))
                if cached_map.size == original.size and values.max() > values.min() and hashlib.sha256(destination.read_bytes()).hexdigest() == completed.get("map_sha256"):
                    continue
        seed = (args.seed + int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)) % (2**32)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        square = pad_to_square(original, fill_color=(0, 0, 0))
        pixels = processor(images=square, return_tensors="pt")["pixel_values"].to(args.device)
        values = vision_heatmap_iba(text_ids, pixels, model, args.vlayer, args.vbeta, args.vvar,
                                   device=args.device, ensemble=False, progbar=False)
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all() or values.max() <= values.min():
            raise RuntimeError(f"Non-finite or constant teacher output: {stem}")
        values = cv2.resize(values, square.size, interpolation=cv2.INTER_NEAREST)
        values = unpad_square_from_resized_array(values, original.width, original.height, square.width)
        encoded = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        if encoded.max() == encoded.min():
            raise RuntimeError(f"Constant teacher after crop/encoding: {stem}")
        temporary = destination.with_suffix(".tmp.png")
        Image.fromarray(encoded).save(temporary)
        temporary.replace(destination)
        # Never accumulate gradients on the fixed teacher between images.
        model.zero_grad(set_to_none=True)
        manifest["completed"][stem] = {"rgb_sha256": source_hash, "seed": seed,
                                       "map_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"COMPLETE: {len(selected)} teacher maps at {output}")


if __name__ == "__main__":
    main()
