"""Core building blocks for SPGNet.

Faithfully reimplemented from:
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/model/basemodel.py
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/model/evspsegnet.py

Components:
- SEModule: Squeeze-Excitation on sparse tensors (global avg pool → FC → sigmoid)
- GDConv: Grouped Dilated sparse 3D conv (splits channels evenly, each group temp→temp)
- GDBlock: GDSCA block (shortcut + pw→GDConv→SE→pw3 with additive residual)
- SparseBasicBlock: Standard sparse residual block
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseModule

# Match original repo's BatchNorm1d config
norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)


class SEModule(nn.Module):
    """Squeeze-Excitation module for sparse 3D tensors.

    Original: basemodel.py SparseAgvPool + FC.
    Global avg pool over all active voxels → FC → ReLU → FC → Sigmoid → scale.
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        se = x.features.mean(dim=0)  # (C,) global average
        se = self.fc(se)  # (C,) channel weights
        return x.replace_feature(se * x.features)


class GDConv(SparseModule):
    """Grouped Dilated Sparse 3D Convolution.

    Original: basemodel.py GDConv.
    Splits input channels evenly into groups. Each group is processed by
    a SubMConv3d with a different dilation rate. Results are concatenated.

    Key: Each group has temp = out_channels // num_splits channels for BOTH
    input and output. Input features are sliced, NOT separate channel counts.
    """

    def __init__(
        self,
        out_channels: int,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3, 4]

        self.num_splits = len(dilations)
        assert out_channels % self.num_splits == 0
        self.temp = out_channels // self.num_splits

        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                spconv.SubMConv3d(
                    self.temp, self.temp,
                    kernel_size=3, padding=d, dilation=d, stride=1,
                )
            )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        results = []
        for i in range(self.num_splits):
            # Slice features for this group
            features_split = x.features[:, i * self.temp : (i + 1) * self.temp]
            # Create new sparse tensor with same spatial structure
            x_split = spconv.SparseConvTensor(
                features_split, x.indices, x.spatial_shape, x.batch_size
            )
            results.append(self.convs[i](x_split).features)

        merged = torch.cat(results, dim=1)
        return x.replace_feature(merged)


class GDBlock(SparseModule):
    """Grouped Dilated Sparse Convolution Attention (GDSCA) block.

    Original: basemodel.py GDBlock.
    Architecture:
    - Shortcut: SubMConv3d(in, out, 1) + BN
    - Main: pw_conv(in, out+ad) + BN + ReLU → GDConv(out+ad) + BN + ReLU → SE(out+ad) → conv3(out+ad, out) + BN
    - Fusion: ADDITIVE residual (main + shortcut) → ReLU

    ad_channels: Extra channels in the main path for richer features.
    Varies by encoder stage: 16 (stage 1,2), 8 (stage 3), 0 (stage 4).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: list[int] | None = None,
        ad_channels: int = 16,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3, 4]

        mid_channels = out_channels + ad_channels

        # Shortcut: 1x1 conv + BN
        self.shortcut = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(out_channels),
        )

        # Main path
        self.pwconv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, mid_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(mid_channels),
            nn.ReLU(),
        )

        self.gdconv = spconv.SparseSequential(
            GDConv(mid_channels, dilations=dilations),
            norm_fn(mid_channels),
            nn.ReLU(),
        )

        self.se = SEModule(mid_channels, reduction=2)

        self.conv3 = spconv.SparseSequential(
            spconv.SubMConv3d(mid_channels, out_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(out_channels),
        )

        self.act = spconv.SparseSequential(nn.ReLU())

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        # Shortcut
        identity = spconv.SparseConvTensor(
            x.features, x.indices, x.spatial_shape, x.batch_size
        )
        identity = self.shortcut(identity)

        # Main path
        out = self.pwconv(x)
        out = self.gdconv(out)
        out = self.se(out)
        out = self.conv3(out)

        # Additive fusion + ReLU
        out = out.replace_feature(out.features + identity.features)
        out = self.act(out)
        return out


class SparseBasicBlock(SparseModule):
    """Standard sparse 3D residual block.

    Original: evspsegnet.py SparseBasicBlock.
    Two SubMConv3d layers with BN + ReLU and skip connection.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        indice_key: str | None = None,
    ):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=1, padding=1,
            bias=False, indice_key=indice_key,
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1,
            bias=False, indice_key=indice_key,
        )
        self.bn2 = norm_fn(planes)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x.features

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(self.relu(out.features))
        return out
