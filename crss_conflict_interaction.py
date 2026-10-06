import torch
import torch.nn as nn
import torch.nn.functional as F


class ConflictAwareBidirectionalInteraction(nn.Module):
    """
    CRSS-SAM Conflict-Aware Bidirectional Interaction.

    입력
    ----
    semantic_feat:
        Deep SAM feature.
        Shape: B x C_sem x H x W

    structural_feat:
        Mid SAM feature.
        Shape: B x C_str x H x W

    semantic_logits:
        Semantic head logits.
        Shape: B x 1 x Hs x Ws

    structural_logits:
        Structural head logits.
        Shape: B x 1 x Ht x Wt

    핵심 과정
    ---------
    1. C = |P_sem - P_str|
    2. conflict score가 높은 Top-K spatial token 선택
    3. 선택된 token에서만
       Semantic -> Structural
       Structural -> Semantic
       bidirectional cross-attention 수행
    4. 선택된 위치만 refined feature로 교체
    5. Semantic prediction을 base로 두고
       conflict 위치에서만 correction 적용
    """

    def __init__(
        self,
        semantic_channels,
        structural_channels,
        embed_dim=128,
        num_heads=4,
        topk_ratio=0.10,
        dropout=0.0,
    ):
        super().__init__()

        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(
                f"topk_ratio must be in (0, 1], got {topk_ratio}"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible "
                f"by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.topk_ratio = topk_ratio

        # --------------------------------------------------
        # 서로 channel 수가 다른
        # semantic / structural feature를
        # 같은 embedding dimension으로 맞춤
        # --------------------------------------------------

        self.semantic_proj = nn.Sequential(
            nn.Conv2d(
                semantic_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=embed_dim,
            ),
            nn.GELU(),
        )

        self.structural_proj = nn.Sequential(
            nn.Conv2d(
                structural_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=embed_dim,
            ),
            nn.GELU(),
        )

        # --------------------------------------------------
        # Bidirectional cross-attention
        #
        # semantic query <- structural key/value
        # structural query <- semantic key/value
        # --------------------------------------------------

        self.sem_from_str = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.str_from_sem = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.sem_norm1 = nn.LayerNorm(
            embed_dim
        )

        self.str_norm1 = nn.LayerNorm(
            embed_dim
        )

        self.sem_ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim * 4,
            ),
            nn.GELU(),
            nn.Linear(
                embed_dim * 4,
                embed_dim,
            ),
        )

        self.str_ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim * 4,
            ),
            nn.GELU(),
            nn.Linear(
                embed_dim * 4,
                embed_dim,
            ),
        )

        self.sem_norm2 = nn.LayerNorm(
            embed_dim
        )

        self.str_norm2 = nn.LayerNorm(
            embed_dim
        )

        # --------------------------------------------------
        # Selective refinement decoder
        #
        # refined semantic
        # refined structural
        # conflict score
        #
        # -> semantic logits에 더할 correction 생성
        # --------------------------------------------------

        self.refine_decoder = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2 + 1,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=embed_dim,
            ),
            nn.GELU(),

            nn.Conv2d(
                embed_dim,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=embed_dim // 2,
            ),
            nn.GELU(),

            nn.Conv2d(
                embed_dim // 2,
                1,
                kernel_size=1,
            ),
        )

    def _topk_mask(
        self,
        conflict,
    ):
        """
        각 image별 conflict 상위 K% 위치 선택.

        conflict:
            B x 1 x H x W

        return
        ------
        mask:
            B x 1 x H x W

        indices:
            B x K
        """

        b, _, h, w = conflict.shape

        flat = conflict.flatten(1)

        n_tokens = h * w

        k = max(
            1,
            int(
                n_tokens
                * self.topk_ratio
            ),
        )

        indices = torch.topk(
            flat,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

        mask_flat = torch.zeros_like(
            flat
        )

        mask_flat.scatter_(
            1,
            indices,
            1.0,
        )

        mask = mask_flat.view(
            b,
            1,
            h,
            w,
        )

        return mask, indices

    def _gather_tokens(
        self,
        tokens,
        indices,
    ):
        """
        tokens:
            B x N x C

        indices:
            B x K

        return:
            B x K x C
        """

        index = indices.unsqueeze(
            -1
        ).expand(
            -1,
            -1,
            tokens.shape[-1],
        )

        return torch.gather(
            tokens,
            dim=1,
            index=index,
        )

    def _scatter_tokens(
        self,
        base_tokens,
        refined_tokens,
        indices,
    ):
        """
        선택된 Top-K 위치만 refined token으로 교체.
        """

        output = base_tokens.clone()

        index = indices.unsqueeze(
            -1
        ).expand(
            -1,
            -1,
            base_tokens.shape[-1],
        )

        output.scatter_(
            dim=1,
            index=index,
            src=refined_tokens,
        )

        return output

    def forward(
        self,
        semantic_feat,
        structural_feat,
        semantic_logits,
        structural_logits,
    ):
        # --------------------------------------------------
        # Spatial size 통일
        # deep feature spatial size를 기준으로 사용
        # --------------------------------------------------

        target_hw = (
            semantic_feat.shape[-2],
            semantic_feat.shape[-1],
        )

        if (
            structural_feat.shape[-2:]
            != target_hw
        ):
            structural_feat = F.interpolate(
                structural_feat,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        # --------------------------------------------------
        # Probability -> Conflict Map
        # --------------------------------------------------

        sem_prob = torch.sigmoid(
            F.interpolate(
                semantic_logits,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        )

        str_prob = torch.sigmoid(
            F.interpolate(
                structural_logits,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        )

        conflict = (
            sem_prob
            - str_prob
        ).abs()

        # Conflict score 자체는
        # selector 역할만 하도록 detach.
        conflict_for_selection = (
            conflict.detach()
        )

        conflict_mask, topk_indices = (
            self._topk_mask(
                conflict_for_selection
            )
        )

        # --------------------------------------------------
        # Feature projection
        # --------------------------------------------------

        sem = self.semantic_proj(
            semantic_feat
        )

        st = self.structural_proj(
            structural_feat
        )

        b, c, h, w = sem.shape

        # B x C x H x W
        # ->
        # B x N x C
        sem_tokens = (
            sem
            .flatten(2)
            .transpose(1, 2)
        )

        str_tokens = (
            st
            .flatten(2)
            .transpose(1, 2)
        )

        # --------------------------------------------------
        # Top-K conflict token만 추출
        # --------------------------------------------------

        sem_selected = (
            self._gather_tokens(
                sem_tokens,
                topk_indices,
            )
        )

        str_selected = (
            self._gather_tokens(
                str_tokens,
                topk_indices,
            )
        )

        # --------------------------------------------------
        # Structural -> Semantic
        # --------------------------------------------------

        sem_attn, _ = (
            self.sem_from_str(
                query=sem_selected,
                key=str_selected,
                value=str_selected,
                need_weights=False,
            )
        )

        sem_refined = self.sem_norm1(
            sem_selected
            + sem_attn
        )

        sem_refined = self.sem_norm2(
            sem_refined
            + self.sem_ffn(
                sem_refined
            )
        )

        # --------------------------------------------------
        # Semantic -> Structural
        # --------------------------------------------------

        str_attn, _ = (
            self.str_from_sem(
                query=str_selected,
                key=sem_selected,
                value=sem_selected,
                need_weights=False,
            )
        )

        str_refined = self.str_norm1(
            str_selected
            + str_attn
        )

        str_refined = self.str_norm2(
            str_refined
            + self.str_ffn(
                str_refined
            )
        )

        # --------------------------------------------------
        # 선택된 위치에만 refined token 삽입
        # --------------------------------------------------

        sem_tokens_refined = (
            self._scatter_tokens(
                sem_tokens,
                sem_refined,
                topk_indices,
            )
        )

        str_tokens_refined = (
            self._scatter_tokens(
                str_tokens,
                str_refined,
                topk_indices,
            )
        )

        sem_refined_map = (
            sem_tokens_refined
            .transpose(1, 2)
            .reshape(
                b,
                c,
                h,
                w,
            )
        )

        str_refined_map = (
            str_tokens_refined
            .transpose(1, 2)
            .reshape(
                b,
                c,
                h,
                w,
            )
        )

        # --------------------------------------------------
        # Correction prediction
        # --------------------------------------------------

        decoder_input = torch.cat(
            [
                sem_refined_map,
                str_refined_map,
                conflict,
            ],
            dim=1,
        )

        correction = (
            self.refine_decoder(
                decoder_input
            )
        )

        # --------------------------------------------------
        # Semantic prediction을 baseline으로 사용
        #
        # Low-conflict:
        # semantic prediction 그대로 유지
        #
        # High-conflict:
        # interaction 결과에 따른 correction 적용
        # --------------------------------------------------

        base_logits = F.interpolate(
            semantic_logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

        final_logits = (
            base_logits
            + conflict_mask
            * correction
        )

        return {
            "final_logits":
                final_logits,

            "base_logits":
                base_logits,

            "correction":
                correction,

            "conflict":
                conflict,

            "conflict_mask":
                conflict_mask,

            "semantic_refined":
                sem_refined_map,

            "structural_refined":
                str_refined_map,

            "topk_indices":
                topk_indices,
        }