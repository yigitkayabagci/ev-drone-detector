"""Patch Attention module for SPGNet.

Faithfully reimplemented from:
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/model/evspsegnet.py

Provides global context modeling on sparse voxels:
1. Cascaded SparseMaxPool3d with kernel/stride (2,2,4) to reach target (11,9,8)
2. MultiheadAttention (1 head) on compressed features
3. Cascaded SparseInverseConv3d to upsample back
4. Residual connection + final 1x1 SubMConv3d
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseModule


class PatchAttention(SparseModule):
    """Patch-based self-attention for sparse 3D tensors.

    Original: evspsegnet.py patch_attention.
    Number of pool layers = log2(in_spatial_size[0] / att_spatial_size[0]).
    Target att_spatial_size is always (11, 9, 8).

    Args:
        channel: Feature channels.
        in_spatial_size: Spatial dimensions at this stage (W, H, T).
        att_spatial_size: Target dimensions after pooling (default: (11, 9, 8)).
        indice_key: Prefix for spconv index caching.
    """

    def __init__(
        self,
        channel: int,
        in_spatial_size: tuple[int, int, int],
        att_spatial_size: tuple[int, int, int] = (11, 9, 8),
        indice_key: str = "pa1",
    ):
        super().__init__()

        maxpool_num = int(math.log2(int(in_spatial_size[0] / att_spatial_size[0])))

        # Downsampling: cascaded SparseMaxPool3d
        self.layers = nn.ModuleList()
        for i in range(maxpool_num):
            self.layers.append(
                spconv.SparseMaxPool3d(
                    (2, 2, 4), stride=(2, 2, 4),
                    indice_key=indice_key + str(i),
                )
            )

        # Attention (1 head, not batch_first — original uses default)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channel, num_heads=1,
        )

        # Upsampling: cascaded SparseInverseConv3d (reverse order)
        self.inver_layer = nn.ModuleList()
        for i in range(maxpool_num):
            self.inver_layer.append(
                spconv.SparseInverseConv3d(
                    channel, channel, (2, 2, 4),
                    indice_key=indice_key + str(int(maxpool_num - 1) - i),
                    bias=False,
                )
            )

        # Final 1x1 conv after residual (exists in original)
        self.conv = spconv.SubMConv3d(
            channel, channel, 1, stride=1, padding=1, bias=False,
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x.features

        # Downsample
        for m in self.layers:
            x = m(x)

        # Attention
        x_features = x.features.unsqueeze(0)  # (1, N_compressed, C)
        attn_output, _ = self.multihead_attn(
            query=x_features, key=x_features, value=x_features,
        )
        attn_output = attn_output.squeeze(0)
        x = x.replace_feature(attn_output)

        # Upsample
        for m in self.inver_layer:
            x = m(x)

        # Residual + final conv
        x = x.replace_feature(x.features + identity)
        x = self.conv(x)

        return x
