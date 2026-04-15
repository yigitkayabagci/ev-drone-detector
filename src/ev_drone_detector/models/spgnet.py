"""SPGNet (EV-SpSegNet) — Sparse Point-cloud segmentation network.

Faithfully reimplemented from:
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/model/evspsegnet.py

Architecture matches the original exactly:
  Encoder: 4 stages with SparseConv3d(kernel=3, stride=[2,2,4]) downsampling
           + GDBlock + PatchAttention (stages 2-4)
  Decoder: 4 UR blocks with channel_reduction (reshape+sum), NOT learned reduction
  Head: Linear(width, 1) + Sigmoid

IMPORTANT: Every GDBlock is wrapped with extra BN+ReLU via post_act_block,
matching the original code. GDBlock already has internal BN+ReLU, so the
extra BN normalizes after the residual addition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseModule

from ev_drone_detector.models.blocks import (
    GDBlock,
    SparseBasicBlock,
    norm_fn,
)
from ev_drone_detector.models.patch_attention import PatchAttention


def _post_act_block_spconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: list[int] | int = 1,
    padding: int = 1,
    indice_key: str | None = None,
) -> spconv.SparseSequential:
    """SparseConv3d + BN + ReLU (for downsampling).
    Original: post_act_block with conv_type='spconv'.
    """
    return spconv.SparseSequential(
        spconv.SparseConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
            indice_key=indice_key,
        ),
        norm_fn(out_channels),
        nn.ReLU(),
    )


def _post_act_block_inverse(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    indice_key: str | None = None,
) -> spconv.SparseSequential:
    """SparseInverseConv3d + BN + ReLU (for upsampling).
    Original: post_act_block with conv_type='inverseconv'.
    """
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size,
            indice_key=indice_key, bias=False,
        ),
        norm_fn(out_channels),
        nn.ReLU(),
    )


def _post_act_block_gd(
    in_channels: int,
    out_channels: int,
    dilations: list[int],
    ad_channels: int = 16,
) -> spconv.SparseSequential:
    """GDBlock + BN + ReLU.
    Original: post_act_block with conv_type='gd'.
    This wrapping is critical — every GDBlock in the original gets
    an extra BN+ReLU on top of its internal BN+ReLU.
    """
    return spconv.SparseSequential(
        GDBlock(in_channels, out_channels, dilations=dilations, ad_channels=ad_channels),
        norm_fn(out_channels),
        nn.ReLU(),
    )


class SPGNet(nn.Module):
    """EV-SpSegNet for event-based small object segmentation.

    Args:
        input_channel: Number of input features per event (4: x,y,t,polarity).
        width: Base channel width (default: 12).
        spatial_shape: Padded spatial dimensions [W, H, T].
                       Must be [352, 288, 8192] for DAVIS346 sensor.
        dilations: Dilation rates for GDSC blocks.
    """

    def __init__(
        self,
        input_channel: int = 4,
        width: int = 12,
        spatial_shape: list[int] | None = None,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        if spatial_shape is None:
            spatial_shape = [352, 288, 8192]
        if dilations is None:
            dilations = [1, 2, 3, 4]

        self.spatial_shape = spatial_shape
        w = width

        # ─── Encoder ───────────────────────────────────────────
        # Input conv: SubMConv3d + BN + ReLU
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, w, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(w),
            nn.ReLU(),
        )

        # Stage 1: GDBlock(w→w, ad=16) + BN + ReLU
        self.conv1 = _post_act_block_gd(w, w, dilations, ad_channels=16)

        # Stage 2: Downsample(w→2w) + GDBlock(2w→2w, ad=16) + PatchAttention
        self.conv2 = spconv.SparseSequential(
            _post_act_block_spconv(w, 2 * w, 3, stride=[2, 2, 4], padding=1, indice_key='spconv2'),
            _post_act_block_gd(2 * w, 2 * w, dilations, ad_channels=16),
        )
        s2 = [spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 4]
        self.pa2 = PatchAttention(2 * w, tuple(s2), indice_key='pa2')

        # Stage 3: Downsample(2w→4w) + GDBlock(4w→4w, ad=8) + PatchAttention
        self.conv3 = spconv.SparseSequential(
            _post_act_block_spconv(2 * w, 4 * w, 3, stride=[2, 2, 4], padding=1, indice_key='spconv3'),
            _post_act_block_gd(4 * w, 4 * w, dilations, ad_channels=8),
        )
        s3 = [s2[0] // 2, s2[1] // 2, s2[2] // 4]
        self.pa3 = PatchAttention(4 * w, tuple(s3), indice_key='pa3')

        # Stage 4: Downsample(4w→4w) + GDBlock(4w→4w, ad=0) + PatchAttention
        self.conv4 = spconv.SparseSequential(
            _post_act_block_spconv(4 * w, 4 * w, 3, stride=[2, 2, 4], padding=1, indice_key='spconv4'),
            _post_act_block_gd(4 * w, 4 * w, dilations, ad_channels=0),
        )
        s4 = [s3[0] // 2, s3[1] // 2, s3[2] // 4]
        self.pa4 = PatchAttention(4 * w, tuple(s4), indice_key='pa4')

        # ─── Decoder ──────────────────────────────────────────
        # UR block 4: x_conv4 (lateral) + x_conv4 (bottom) → upsample
        self.conv_up_t4 = SparseBasicBlock(4 * w, 4 * w, indice_key='subm4')
        self.conv_up_m4 = _post_act_block_gd(8 * w, 4 * w, dilations, ad_channels=16)
        self.inv_conv4 = _post_act_block_inverse(4 * w, 4 * w, 3, indice_key='spconv4')

        # UR block 3: x_conv3 (lateral) + x_up4 (bottom)
        self.conv_up_t3 = SparseBasicBlock(4 * w, 4 * w, indice_key='subm3')
        self.conv_up_m3 = _post_act_block_gd(8 * w, 4 * w, dilations, ad_channels=16)
        self.inv_conv3 = _post_act_block_inverse(4 * w, 2 * w, 3, indice_key='spconv3')

        # UR block 2: x_conv2 (lateral) + x_up3 (bottom)
        self.conv_up_t2 = SparseBasicBlock(2 * w, 2 * w, indice_key='subm2')
        self.conv_up_m2 = _post_act_block_gd(4 * w, 2 * w, dilations, ad_channels=16)
        self.inv_conv2 = _post_act_block_inverse(2 * w, w, 3, indice_key='spconv2')

        # UR block 1: x_conv1 (lateral) + x_up2 (bottom) → GDBlock (NOT InverseConv)
        self.conv_up_t1 = SparseBasicBlock(w, w, indice_key='subm1')
        self.conv_up_m1 = _post_act_block_gd(2 * w, w, dilations, ad_channels=16)
        self.conv5 = spconv.SparseSequential(
            _post_act_block_gd(w, w, dilations, ad_channels=16),
        )

        # ─── Output Head ──────────────────────────────────────
        self.semantic_linear = nn.Sequential(
            nn.Linear(w, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def channel_reduction(
        x: spconv.SparseConvTensor, out_channels: int,
    ) -> spconv.SparseConvTensor:
        """Parameter-free channel reduction via reshape + sum.

        Original: evspsegnet.py channel_reduction.
        Reshapes (N, C_in) → (N, C_out, C_in/C_out) → sum → (N, C_out).
        """
        features = x.features
        n, in_channels = features.shape
        assert in_channels % out_channels == 0 and in_channels >= out_channels
        return x.replace_feature(
            features.view(n, out_channels, -1).sum(dim=2)
        )

    def UR_block_forward(
        self,
        x_lateral: spconv.SparseConvTensor,
        x_bottom: spconv.SparseConvTensor,
        conv_t,
        conv_m,
        conv_inv,
    ) -> spconv.SparseConvTensor:
        """UR decoder block. Original: evspsegnet.py UR_block_forward.

        1. Transform lateral with SparseBasicBlock
        2. Concat bottom + transformed lateral
        3. GDBlock fusion → reduced channels
        4. Channel reduction on concat → same channels
        5. Residual: fused + reduced
        6. Inverse conv (or GDBlock for last stage)
        """
        x_trans = conv_t(x_lateral)
        x = x_trans.replace_feature(
            torch.cat((x_bottom.features, x_trans.features), dim=1)
        )
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = x.replace_feature(x_m.features + x.features)
        x = conv_inv(x)
        return x

    def forward(
        self, input: spconv.SparseConvTensor,
    ) -> tuple[torch.Tensor, spconv.SparseConvTensor]:
        """Forward pass.

        Args:
            input: SparseConvTensor with features (N, input_channel)
                   and indices (N, 4) as [batch, x, y, t].

        Returns:
            predictions: (N_voxels, 1) sigmoid probabilities.
            voxel: SparseConvTensor with predictions as features (for STC loss).
        """
        # ─── Encoder ───────────────────────────────────────────
        x = self.conv_input(input)
        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.pa2(x_conv2)

        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.pa3(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.pa4(x_conv4)

        # ─── Decoder ──────────────────────────────────────────
        # UR block 4: self-referential (x_conv4, x_conv4)
        x_up4 = self.UR_block_forward(
            x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4,
        )
        # UR block 3
        x_up3 = self.UR_block_forward(
            x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3,
        )
        # UR block 2
        x_up2 = self.UR_block_forward(
            x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2,
        )
        # UR block 1: uses conv5 (wrapped GDBlock) instead of InverseConv
        x_up1 = self.UR_block_forward(
            x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5,
        )

        # ─── Output ───────────────────────────────────────────
        output = self.semantic_linear(x_up1.features)  # (N_voxels, 1)
        voxel = x_up1.replace_feature(output)

        return output, voxel
