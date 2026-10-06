import importlib.util
import sys
from pathlib import Path


TRAINABLE_NAMES = [
    "polyp_feature",
    "Domain_generalization",
    "Adapter_mlp",
    "linear_classic",
]


def load_official_psf(repo_root="baselines/PSF-SAM"):
    repo_root = Path(repo_root).resolve()
    package_dir = repo_root / "PSF-SAM"
    package_name = "psf_sam_official"

    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        # Skip PSF-SAM __init__.py side effects

    from psf_sam_official.build_sam import sam_model_registry

    freeze_spec = importlib.util.spec_from_file_location(
        "psf_sam_freeze_official",
        repo_root / "freeze_except.py",
    )
    freeze_module = importlib.util.module_from_spec(freeze_spec)
    freeze_spec.loader.exec_module(freeze_module)

    return sam_model_registry, freeze_module.freeze_except


def build_psf_sam(checkpoint, repo_root="baselines/PSF-SAM", device="cuda"):
    registry, freeze_except = load_official_psf(repo_root)
    model = registry["vit_h"](checkpoint=checkpoint)
    model = freeze_except(model, TRAINABLE_NAMES)
    model.to(device)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_records(images):
    records = []
    for image in images:
        h, w = image.shape[-2:]
        records.append({
            "image": image,
            "original_size": (h, w),
        })
    return records


def outputs_to_logits(outputs):
    import torch
    xs = []
    for out in outputs:
        x = out["masks"]
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected PSF-SAM mask shape: {tuple(x.shape)}. "
                "Expected (1,2,H,W). Ensure multimask_output=True."
            )
        xs.append(x.squeeze(0))
    return torch.stack(xs, dim=0)
