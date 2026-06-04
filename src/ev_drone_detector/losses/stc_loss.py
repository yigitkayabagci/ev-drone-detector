"""Spatiotemporal Correlation (STC) Loss.

Faithfully reimplemented from:
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/utils/stcloss.py

NOTE: The paper describes a gamma exponent (L_stc = -w^gamma * log(p)),
but the actual code uses LINEAR multiplication without gamma.
We follow the CODE, not the paper.

The loss:
  1. Fixed SubMConv3d(1, 1, [k, k, t]) with all-ones weights counts neighbors
  2. w_stc = sigmoid(neighbor_sum - mean)
  3. loss = mean(label * w_stc * (-log(p)) + (1-label) * (1-w_stc) * (-log(1-p)))

w_stc is DETACHED — no gradient flows through the weight computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import spconv.pytorch as spconv


class STCLoss(nn.Module):
    """Spatiotemporal Correlation Loss for event segmentation.

    Args:
        k: Spatial kernel size for neighbor counting.
        tau: Temporal kernel size for neighbor counting.
    """

    def __init__(self, k: int = 3, tau: int = 5):
        super().__init__()
        self.k = k
        self.tau = tau

        # Fixed (non-learnable) sparse conv for neighbor density
        self.stc_conv = spconv.SubMConv3d(
            1, 1,
            kernel_size=[k, k, tau],
            stride=1,
            padding=[k // 2, k // 2, tau // 2],
            bias=False,
        )
        # Initialize weights to all 1s and freeze
        self.stc_conv.weight.data.fill_(1)
        self.stc_conv.requires_grad_(False)

        self.eps = 1e-5

    def forward(
        self,
        voxel: spconv.SparseConvTensor,
        p2v_map: torch.Tensor,
        preds: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute STC loss.

        Args:
            voxel: SparseConvTensor with predictions as features (from model output).
            p2v_map: (N_events,) point-to-voxel mapping.
            preds: (N_voxels, 1) sigmoid predictions.
            label: (N_events,) binary labels.

        Returns:
            Scalar STC loss.
        """
        # Compute spatiotemporal density
        stc_voxel = self.stc_conv(voxel)
        mean_stc = torch.mean(stc_voxel.features)
        stc_weights = torch.sigmoid(stc_voxel.features - mean_stc)

        # Map voxel weights to event-level and DETACH.
        # squeeze(-1) (not bare squeeze) so a single-event batch stays (1,) and
        # does not collapse to a 0-d scalar.
        stc_weights = stc_weights[p2v_map].squeeze(-1).detach()

        # Map predictions to event-level
        preds_events = preds[p2v_map].squeeze(-1)
        preds_events = torch.clamp(preds_events, 0, 1)

        # Weighted BCE (NO gamma exponent — matches original code)
        pos_loss = -torch.log(preds_events + self.eps)
        neg_loss = -torch.log(1 - preds_events + self.eps)

        loss = (label * stc_weights * pos_loss) + (
            (1 - label) * (1 - stc_weights) * neg_loss
        )

        return loss.mean()
