import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAQKV(nn.Module):
    """
    SAM의 fused QKV Linear layer를 감싸서
    Q와 V에만 LoRA를 적용한다.

    기존 SAM QKV weight는 frozen 상태로 유지한다.
    """

    def __init__(
        self,
        qkv: nn.Linear,
        rank=4,
        alpha=4.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(
                f"rank must be positive, got {rank}"
            )

        self.qkv = qkv

        # 기존 SAM QKV는 freeze
        for p in self.qkv.parameters():
            p.requires_grad = False

        dim = qkv.in_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # ------------------------------------------
        # Q LoRA
        # ------------------------------------------
        self.q_A = nn.Linear(
            dim,
            rank,
            bias=False,
        )

        self.q_B = nn.Linear(
            rank,
            dim,
            bias=False,
        )

        # ------------------------------------------
        # V LoRA
        # ------------------------------------------
        self.v_A = nn.Linear(
            dim,
            rank,
            bias=False,
        )

        self.v_B = nn.Linear(
            rank,
            dim,
            bias=False,
        )

        # ------------------------------------------
        # LoRA initialization
        #
        # A: random
        # B: zero
        #
        # 따라서 학습 시작 시점에는
        # 원래 SAM과 정확히 같은 출력이 나온다.
        # ------------------------------------------
        nn.init.kaiming_uniform_(
            self.q_A.weight,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.q_B.weight
        )

        nn.init.kaiming_uniform_(
            self.v_A.weight,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.v_B.weight
        )

    def forward(self, x):
        # Frozen SAM의 원래 QKV
        base = self.qkv(x)

        q, k, v = base.chunk(
            3,
            dim=-1,
        )

        # LoRA update
        delta_q = (
            self.q_B(
                self.q_A(x)
            )
            * self.scaling
        )

        delta_v = (
            self.v_B(
                self.v_A(x)
            )
            * self.scaling
        )

        # Q / V에만 adaptation 적용
        q = q + delta_q
        v = v + delta_v

        # K는 기존 SAM 그대로 사용
        return torch.cat(
            [q, k, v],
            dim=-1,
        )


class LoRASAMFeatureExtractor(nn.Module):
    """
    CRSS-SAM용 SAM feature extractor.

    SAM 원본 weight는 전체 frozen 상태로 유지한다.

    Parameters
    ----------
    mid_block:
        Structural branch에서 사용할
        intermediate feature를 추출하는 block 위치.

    lora_start_block:
        Q/V LoRA를 적용하기 시작할 block 위치.

    예시
    ----
    Late LoRA:
        mid_block = 15
        lora_start_block = 16

        Block 0~15  : Frozen SAM
        Block 16~31 : Frozen SAM + trainable Q/V LoRA

    Full LoRA:
        mid_block = 15
        lora_start_block = 0

        Block 0~31 : Frozen SAM + trainable Q/V LoRA

    주의
    ----
    Structural feature는 detach해서 Structural loss가
    SAM/LoRA로 gradient를 직접 보내지 않도록 한다.

    그러나 hook에서 저장한 복사본만 detach하기 때문에
    encoder의 실제 forward graph는 끊기지 않는다.

    따라서 Full LoRA에서는 Semantic loss의 gradient가
    Block 0의 LoRA까지 정상적으로 전달된다.
    """

    def __init__(
        self,
        sam,
        mid_block=15,
        lora_start_block=None,
        rank=4,
        alpha=4.0,
    ):
        super().__init__()

        self.sam = sam
        self.mid_block = mid_block
        self._mid = None

        # ------------------------------------------
        # SAM 원본 weight 전체 freeze
        # ------------------------------------------
        for p in self.sam.parameters():
            p.requires_grad = False

        n_blocks = len(
            self.sam.image_encoder.blocks
        )

        self.n_blocks = n_blocks

        # ------------------------------------------
        # mid block validation
        # ------------------------------------------
        if not 0 <= mid_block < n_blocks:
            raise ValueError(
                f"mid_block must be in "
                f"[0, {n_blocks - 1}], "
                f"got {mid_block}"
            )

        # ------------------------------------------
        # Backward compatibility
        #
        # lora_start_block을 지정하지 않으면
        # 기존 E1처럼 mid_block 다음 block부터 적용
        # ------------------------------------------
        if lora_start_block is None:
            lora_start_block = (
                mid_block + 1
            )

        if not 0 <= lora_start_block < n_blocks:
            raise ValueError(
                f"lora_start_block must be in "
                f"[0, {n_blocks - 1}], "
                f"got {lora_start_block}"
            )

        self.lora_start_block = (
            lora_start_block
        )

        self.lora_blocks = list(
            range(
                lora_start_block,
                n_blocks,
            )
        )

        # ------------------------------------------
        # 지정된 block부터 마지막 block까지
        # Q/V LoRA 적용
        # ------------------------------------------
        for i in self.lora_blocks:

            old_qkv = (
                self.sam
                .image_encoder
                .blocks[i]
                .attn
                .qkv
            )

            self.sam.image_encoder.blocks[
                i
            ].attn.qkv = LoRAQKV(
                old_qkv,
                rank=rank,
                alpha=alpha,
            )

        # ------------------------------------------
        # Structural feature hook
        # ------------------------------------------
        self.sam.image_encoder.blocks[
            mid_block
        ].register_forward_hook(
            self._save_mid
        )

        # Structural feature channel
        self.mid_channels = int(
            self.sam
            .image_encoder
            .blocks[0]
            .norm1
            .normalized_shape[0]
        )

        # SAM neck output channel
        self.deep_channels = int(
            self.sam
            .image_encoder
            .neck[0]
            .out_channels
        )

    def _save_mid(
        self,
        module,
        inputs,
        output,
    ):
        """
        Structural branch용 intermediate feature 저장.

        여기서 저장되는 feature만 detach한다.

        실제 encoder forward output 자체를
        변경하는 것이 아니기 때문에,
        Semantic branch의 computation graph는 유지된다.

        따라서 Full LoRA에서는 Semantic loss가
        early-block LoRA까지 학습시킬 수 있다.
        """

        self._mid = output.detach()

    def forward(
        self,
        image_0_255,
    ):
        # SAM preprocessing
        x = self.sam.preprocess(
            image_0_255
        )

        self._mid = None

        # 중요:
        # torch.no_grad()를 사용하면 안 된다.
        # LoRA까지 gradient가 흘러야 하기 때문.
        deep = self.sam.image_encoder(
            x
        )

        if self._mid is None:
            raise RuntimeError(
                "Mid-level hook did not fire."
            )

        # SAM transformer block output:
        # B x H x W x C
        #
        # Conv head 입력용:
        # B x C x H x W
        mid = (
            self._mid
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

        # deep은 detach하면 안 된다.
        # Semantic loss가 LoRA까지
        # backpropagation 되어야 한다.
        return mid, deep

    def lora_parameters(self):
        """
        학습 가능한 LoRA parameter만 반환.
        """

        params = []

        for module in self.modules():

            if isinstance(
                module,
                LoRAQKV,
            ):
                params.extend([
                    *module.q_A.parameters(),
                    *module.q_B.parameters(),
                    *module.v_A.parameters(),
                    *module.v_B.parameters(),
                ])

        return params

    def trainable_parameter_count(self):
        """
        Feature extractor 내부의
        trainable parameter 개수.

        SAM 원본은 frozen이므로
        실질적으로 LoRA parameter 수가 된다.
        """

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


class SmallSegHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden=128,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden,
                1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                hidden,
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                hidden,
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden,
                1,
                1,
            ),
        )

    def forward(self, x):
        return self.net(x)


class DualCueHeads(nn.Module):
    """
    Semantic / Structural 두 개의 독립적인 segmentation head.
    """

    def __init__(
        self,
        mid_channels,
        deep_channels,
        hidden=128,
    ):
        super().__init__()

        # Deep SAM feature
        self.semantic = SmallSegHead(
            deep_channels,
            hidden,
        )

        # Intermediate SAM feature
        self.structural = SmallSegHead(
            mid_channels,
            hidden,
        )

    def forward(
        self,
        mid_feat,
        deep_feat,
    ):
        semantic_logits = (
            self.semantic(
                deep_feat
            )
        )

        structural_logits = (
            self.structural(
                mid_feat
            )
        )

        return (
            semantic_logits,
            structural_logits,
        )


def soft_dice_loss(
    logits,
    target,
    eps=1e-6,
):
    prob = torch.sigmoid(
        logits
    )

    dims = tuple(
        range(
            1,
            prob.ndim,
        )
    )

    inter = (
        prob
        * target
    ).sum(
        dim=dims
    )

    denom = (
        prob.sum(
            dim=dims
        )
        + target.sum(
            dim=dims
        )
    )

    dice = (
        2.0 * inter
        + eps
    ) / (
        denom
        + eps
    )

    return (
        1.0
        - dice
    ).mean()


def seg_loss(
    logits,
    target,
):
    bce = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
        )
    )

    dice = soft_dice_loss(
        logits,
        target,
    )

    return bce + dice


def soft_boundary_map(x):
    """
    Soft morphological boundary.

    dilation - erosion
    """

    dil = F.max_pool2d(
        x,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    ero = -F.max_pool2d(
        -x,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    return (
        dil - ero
    ).clamp(
        0,
        1,
    )


def structural_boundary_loss(
    logits,
    target,
):
    pred_b = soft_boundary_map(
        torch.sigmoid(
            logits
        )
    )

    gt_b = soft_boundary_map(
        target
    )

    return (
        F.binary_cross_entropy(
            pred_b.clamp(
                1e-5,
                1 - 1e-5,
            ),
            gt_b,
        )
    )