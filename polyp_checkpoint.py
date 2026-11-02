"""Restore exactly the LoRA backbone associated with a dual-head checkpoint."""
import hashlib
import torch
from crss_model import FrozenSAMFeatureExtractor, DualCueHeads
from crss_model_lora import LoRASAMFeatureExtractor


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_backbone(sam, checkpoint, device, default_mid=15):
    ck = torch.load(checkpoint, map_location="cpu")
    config = ck.get("args", {})
    if int(config.get("image_size", 1024)) != 1024:
        raise RuntimeError("This benchmark adapter requires SAM input size 1024")
    mid = int(config.get("mid_block", default_mid))
    if "lora" in ck:
        start = config.get("lora_start_block")
        ext = LoRASAMFeatureExtractor(
            sam, mid_block=mid,
            lora_start_block=mid + 1 if start is None else int(start),
            rank=int(config.get("lora_rank", 4)),
            alpha=float(config.get("lora_alpha", 4.0)),
        ).to(device)
        parameters = dict(ext.named_parameters())
        expected = {name for name, value in parameters.items() if value.requires_grad}
        if set(ck["lora"]) != expected:
            raise RuntimeError("LoRA checkpoint does not contain exactly all required trainable tensors")
        with torch.no_grad():
            for name, value in ck["lora"].items():
                if parameters[name].shape != value.shape:
                    raise RuntimeError(f"LoRA shape mismatch: {name}")
                parameters[name].copy_(value.to(parameters[name]))
    else:
        ext = FrozenSAMFeatureExtractor(sam, mid_block=mid).to(device)
    heads = DualCueHeads(ext.mid_channels, ext.deep_channels).to(device)
    heads.load_state_dict(ck["heads"], strict=True)
    for module in (ext, heads):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return ext, heads, ck
