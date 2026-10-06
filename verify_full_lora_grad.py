import torch

from segment_anything import sam_model_registry

from crss_model_lora import (
    LoRASAMFeatureExtractor,
    DualCueHeads,
)


device = "cuda"

sam = sam_model_registry["vit_h"](
    checkpoint="checkpoints/sam_vit_h_4b8939.pth"
).to(device)

heads = DualCueHeads(
    int(
        sam.image_encoder.blocks[0]
        .norm1.normalized_shape[0]
    ),
    int(
        sam.image_encoder.neck[0]
        .out_channels
    ),
).to(device)

ext = LoRASAMFeatureExtractor(
    sam=sam,
    mid_block=15,
    lora_start_block=0,
    rank=4,
    alpha=4.0,
).to(device)

ext.sam.eval()

x = torch.rand(
    1,
    3,
    1024,
    1024,
    device=device,
) * 255.0

mid, deep = ext(x)

semantic_logits, structural_logits = heads(
    mid,
    deep,
)

# Semantic path만 사용해서 backward
loss = semantic_logits.mean()
loss.backward()


def grad_mean(block_idx, name):
    module = (
        ext.sam.image_encoder
        .blocks[block_idx]
        .attn.qkv
    )

    p = getattr(module, name).weight

    if p.grad is None:
        return None

    return p.grad.abs().mean().item()


for block_idx in [0, 15, 16, 31]:
    print(
        f"Block {block_idx:2d} | "
        f"q_A={grad_mean(block_idx, 'q_A')} | "
        f"q_B={grad_mean(block_idx, 'q_B')} | "
        f"v_A={grad_mean(block_idx, 'v_A')} | "
        f"v_B={grad_mean(block_idx, 'v_B')}"
    )