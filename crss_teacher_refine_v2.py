import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMAffinityTeacherRefiner(nn.Module):
    """
    Teacher Refinement v2

    v1 문제:
      - SAM affinity를 전체 teacher map에 강하게 혼합
      - binary mask는 좋아져도 soft probability calibration이 악화될 수 있음

    v2:
      1. confident teacher pixels로 FG/BG prototype 생성
      2. teacher가 uncertain한 위치만 refinement
      3. SAM affinity 자체가 confident할 때만 강하게 수정
      4. confident original teacher는 최대한 보존
    """

    def __init__(
        self,
        temperature=0.1,
        teacher_weight=0.5,
        fg_threshold=0.7,
        bg_threshold=0.3,
        min_pixels=8,
        eps=1e-6,
    ):
        super().__init__()

        self.temperature = temperature

        # teacher_weight가 클수록 원본 teacher 보존
        self.teacher_weight = teacher_weight

        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.min_pixels = min_pixels
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        deep_feat,
        teacher,
        output_size,
    ):
        # --------------------------------------------------
        # 1. SAM feature normalize
        # --------------------------------------------------

        feat = deep_feat.float()

        feat = F.normalize(
            feat,
            p=2,
            dim=1,
        )

        h, w = feat.shape[-2:]

        # --------------------------------------------------
        # 2. Teacher -> SAM feature resolution
        # --------------------------------------------------

        teacher_low = F.interpolate(
            teacher.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        # --------------------------------------------------
        # 3. Confident FG / BG pixels
        # --------------------------------------------------

        fg_hard = (
            teacher_low >= self.fg_threshold
        ).float()

        bg_hard = (
            teacher_low <= self.bg_threshold
        ).float()

        fg_count = fg_hard.sum(
            dim=(2, 3),
            keepdim=True,
        )

        bg_count = bg_hard.sum(
            dim=(2, 3),
            keepdim=True,
        )

        # confident pixel이 너무 적을 경우에만
        # v1과 같은 soft weighting으로 fallback
        fg_soft = teacher_low

        bg_soft = 1.0 - teacher_low

        use_fg_soft = (
            fg_count < self.min_pixels
        )

        use_bg_soft = (
            bg_count < self.min_pixels
        )

        fg_weight = torch.where(
            use_fg_soft,
            fg_soft,
            fg_hard,
        )

        bg_weight = torch.where(
            use_bg_soft,
            bg_soft,
            bg_hard,
        )

        # --------------------------------------------------
        # 4. Foreground prototype
        # --------------------------------------------------

        fg_proto = (
            feat * fg_weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        fg_denom = fg_weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(
            self.eps
        )

        fg_proto = (
            fg_proto / fg_denom
        )

        # --------------------------------------------------
        # 5. Background prototype
        # --------------------------------------------------

        bg_proto = (
            feat * bg_weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        bg_denom = bg_weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(
            self.eps
        )

        bg_proto = (
            bg_proto / bg_denom
        )

        fg_proto = F.normalize(
            fg_proto,
            p=2,
            dim=1,
        )

        bg_proto = F.normalize(
            bg_proto,
            p=2,
            dim=1,
        )

        # --------------------------------------------------
        # 6. Feature ↔ prototype similarity
        # --------------------------------------------------

        sim_fg = (
            feat * fg_proto
        ).sum(
            dim=1,
            keepdim=True,
        )

        sim_bg = (
            feat * bg_proto
        ).sum(
            dim=1,
            keepdim=True,
        )

        logits = torch.cat(
            [
                sim_bg,
                sim_fg,
            ],
            dim=1,
        )

        logits = (
            logits
            / self.temperature
        )

        prob = torch.softmax(
            logits,
            dim=1,
        )

        affinity = prob[:, 1:2]

        # --------------------------------------------------
        # 7. Original resolution
        # --------------------------------------------------

        affinity_up = F.interpolate(
            affinity,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        teacher_up = F.interpolate(
            teacher.float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        # --------------------------------------------------
        # 8. Teacher uncertainty
        #
        # teacher=0 or 1 -> 0
        # teacher=0.5    -> 1
        # --------------------------------------------------

        teacher_uncertainty = (
            1.0
            - torch.abs(
                2.0 * teacher_up
                - 1.0
            )
        ).clamp(0, 1)

        # --------------------------------------------------
        # 9. Affinity confidence
        #
        # affinity=0.5 -> confidence 0
        # affinity=0/1 -> confidence 1
        # --------------------------------------------------

        affinity_confidence = torch.abs(
            2.0 * affinity_up
            - 1.0
        ).clamp(0, 1)

        # --------------------------------------------------
        # 10. Selective refinement gate
        # --------------------------------------------------

        max_refine_strength = (
            1.0 - self.teacher_weight
        )

        gate = (
            max_refine_strength
            * teacher_uncertainty
            * affinity_confidence
        )

        # --------------------------------------------------
        # 11. Refined teacher
        #
        # confident teacher:
        # gate ≈ 0 → 거의 그대로
        #
        # uncertain teacher + confident affinity:
        # gate ↑ → affinity 방향으로 수정
        # --------------------------------------------------

        refined = (
            teacher_up
            +
            gate
            * (
                affinity_up
                - teacher_up
            )
        )

        refined = refined.clamp(
            0,
            1,
        )

        return (
            refined,
            affinity_up,
        )