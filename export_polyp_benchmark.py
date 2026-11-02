"""Export four predictions at original image resolution, without CLIP at inference."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from segment_anything import sam_model_registry
from crss_dataset import resize_longest_and_pad, SAM_PIXEL_MEAN_RGB
from crss_resolver_v2 import FeatureAwareReliabilityResolver
from polyp_checkpoint import build_backbone, file_sha256
from scripts.prepare_polyp1450 import TEST_COUNTS, paired


def restore_logits(logits, height, width):
    # Inverse of the training loader: longest side 1024, padding at bottom/right.
    scale = 1024 / max(height, width)
    nh, nw = int(round(height * scale)), int(round(width * scale))
    full = F.interpolate(logits.float(), (1024, 1024), mode="bilinear", align_corners=False)
    cropped = full[..., :nh, :nw]
    restored = F.interpolate(cropped, (height, width), mode="bilinear", align_corners=False)
    return restored.sigmoid()[0, 0].cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-root", default="dataset/TestDataset")
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--heads_checkpoint", required=True)
    ap.add_argument("--resolver_checkpoint", required=True)
    ap.add_argument("--output", default="experiments/polyp1450/predictions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=0, help="Smoke only; positive values are not benchmark results")
    args = ap.parse_args()
    hck = torch.load(args.heads_checkpoint, map_location="cpu")
    config = hck.get("args", {})
    model_type = config.get("model_type", "vit_h")
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint).to(args.device)
    ext, heads, hck = build_backbone(sam, args.heads_checkpoint, args.device)
    rck = torch.load(args.resolver_checkpoint, map_location="cpu")
    if not args.max_images:
        for label, checkpoint in [("heads", hck), ("resolver", rck)]:
            settings = checkpoint.get("args", {})
            if not settings.get("fixed_schedule") or settings.get("max_train_batches", 0) != 0:
                raise RuntimeError(f"{label}: use a full fixed-schedule training checkpoint, or --max-images for a smoke export")
            if checkpoint.get("epoch") != settings.get("epochs") or checkpoint.get("metrics", {}).get("train_images") != 1450:
                raise RuntimeError(f"{label}: the complete fixed schedule on all 1450 images has not finished")
    backbone_hash = file_sha256(args.heads_checkpoint)
    if rck.get("backbone_sha256") != backbone_hash:
        raise RuntimeError("Resolver checkpoint was not trained with this exact backbone checkpoint")
    resolver = FeatureAwareReliabilityResolver(ext.mid_channels, ext.deep_channels, proj_channels=16, hidden=64).to(args.device)
    resolver.load_state_dict(rck["resolver"], strict=True); resolver.eval()
    output_size = int(rck.get("args", {}).get("out_size", 256))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Use a fresh prediction output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    methods = ["semantic", "structural", "naive_average", "resolver_v2"]
    report = {"status": "incomplete", "provenance": "local_inference",
              "heads_checkpoint_sha256": backbone_hash, "resolver_checkpoint_sha256": file_sha256(args.resolver_checkpoint),
              "head_epoch": hck.get("epoch"), "resolver_epoch": rck.get("epoch"),
              "smoke": args.max_images > 0, "native_prediction_size": output_size,
              "published_maps": "original-size sigmoid, per-image min-max normalization, truncation to uint8 PNG",
              "restoration": "native logits -> 1024 -> remove bottom/right pad -> original size -> sigmoid; mixed probabilities converted to logits before same restoration",
              "raw_arrays": "un-normalized original-size float32 probabilities", "counts": {}}
    (output / "export_manifest.json").write_text(json.dumps(report, indent=2))
    for name, count in TEST_COUNTS.items():
        images, masks = paired(Path(args.test_root) / name, count)
        stems = sorted(images)[:args.max_images] if args.max_images else sorted(images)
        raw_dir = output / "raw_probabilities" / name; raw_dir.mkdir(parents=True)
        for method in methods:
            (output / method / name).mkdir(parents=True)
        for stem in tqdm(stems, desc=name):
            image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable image: {images[stem]}")
            h, w = image.shape[:2]
            with Image.open(masks[stem]) as gt:
                gt.load()
                if gt.size != (w, h):
                    raise RuntimeError(f"Ground-truth/image size mismatch: {stem}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_longest_and_pad(image, 1024, image_pad_value=SAM_PIXEL_MEAN_RGB)
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(args.device)
            mid, deep = ext(tensor)
            sem, structural = heads(mid, deep)
            sem = F.interpolate(sem, (output_size, output_size), mode="bilinear", align_corners=False)
            structural = F.interpolate(structural, (output_size, output_size), mode="bilinear", align_corners=False)
            ps, pt = sem.sigmoid(), structural.sigmoid()
            average = 0.5 * (ps + pt)
            fused = resolver(ps, pt, mid, deep)["p_fused"]
            logits = [sem, structural, torch.logit(average.clamp(1e-6, 1-1e-6)), torch.logit(fused.clamp(1e-6, 1-1e-6))]
            arrays = {}
            for method, prediction in zip(methods, logits):
                probability = restore_logits(prediction, h, w).astype(np.float32)
                if not np.isfinite(probability).all():
                    raise RuntimeError(f"Non-finite probability: {name}/{stem}/{method}")
                arrays[method] = probability
                normalized = (probability - probability.min()) / (probability.max() - probability.min() + 1e-8)
                Image.fromarray((normalized * 255).astype(np.uint8)).save(output / method / name / f"{stem}.png")
            np.savez_compressed(raw_dir / f"{stem}.npz", **arrays)
        report["counts"][name] = len(stems)
        (output / "export_manifest.json").write_text(json.dumps(report, indent=2))
    report["status"] = "smoke_complete" if args.max_images else "complete"
    (output / "export_manifest.json").write_text(json.dumps(report, indent=2))
    print(f"{report['status']}: {sum(report['counts'].values())} test images, four prediction variants")


if __name__ == "__main__":
    main()
