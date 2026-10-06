import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMAffinityTeacherRefiner(nn.Module):
    """
    Frozen SAM deep feature를 이용하여
    기존 CLIP teacher map을 affinity 기반으로 보정한다.

    Trainable parameter는 없음.

    과정:
      1. CLIP teacher를 SAM feature resolution으로 축소
      2. teacher score를 이용해 foreground/background prototype 생성
      3. SAM feature와 prototype cosine similarity 계산
      4. affinity probability 생성
      5. original teacher와 affinity map을 결합
    """

    def __init__(
        self,
        temperature=0.1,
        teacher_weight=0.5,
        eps=1e-6,
    ):
        super().__init__()

        self.temperature = temperature
        self.teacher_weight = teacher_weight
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        deep_feat,
        teacher,
        output_size,
    ):
        """
        deep_feat:
            [B, C, H, W]
            Frozen SAM deep feature

        teacher:
            [B, 1, H_img, W_img]
            original CLIP attribution

        output_size:
            (H_out, W_out)

        return:
            refined_teacher
            affinity_map
        """

        # ----------------------------------------------
        # 1. SAM feature normalization
        # ----------------------------------------------

        feat = deep_feat.float()

        feat = F.normalize(
            feat,
            p=2,
            dim=1,
        )

        h, w = feat.shape[-2:]

        # ----------------------------------------------
        # 2. Teacher → SAM feature resolution
        # ----------------------------------------------

        teacher_low = F.interpolate(
            teacher.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        # soft foreground/background weights
        fg_weight = teacher_low
        bg_weight = 1.0 - teacher_low

        # ----------------------------------------------
        # 3. Foreground / background prototype
        # ----------------------------------------------

        fg_proto = (
            feat * fg_weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        fg_denom = fg_weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(self.eps)

        fg_proto = (
            fg_proto / fg_denom
        )

        bg_proto = (
            feat * bg_weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        bg_denom = bg_weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(self.eps)

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

        # ----------------------------------------------
        # 4. Pixel ↔ prototype cosine similarity
        # ----------------------------------------------

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
            logits / self.temperature
        )

        prob = torch.softmax(
            logits,
            dim=1,
        )

        # foreground probability
        affinity = prob[:, 1:2]

        # ----------------------------------------------
        # 5. Original resolution으로 확대
        # ----------------------------------------------

        affinity_up = F.interpolate(
            affinity,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        teacher_up = F.interpolate(
            teacher.float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        # ----------------------------------------------
        # 6. Original CLIP + SAM affinity
        # ----------------------------------------------

        refined = (
            self.teacher_weight
            * teacher_up
            +
            (1.0 - self.teacher_weight)
            * affinity_up
        )

        refined = refined.clamp(
            0,
            1,
        )

        return (
            refined,
            affinity_up,
        )