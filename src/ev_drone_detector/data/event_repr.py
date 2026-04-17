"""Event voxelization and batch collation — pure PyTorch.

Adapted from the original Ev-UAV repository:
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/dataset/basedataset.py

The original uses a CUDA-only custom op (HAIS_OP.voxelize_idx); here we
replicate the same semantics with torch.unique + scatter_add so it runs
on CPU or GPU without extra compiled extensions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import spconv.pytorch as spconv


def voxelize_events(
    features: torch.Tensor,
    coords: torch.Tensor,
    batch_idx: int | torch.Tensor,
    spatial_shape: Sequence[int],
) -> tuple[spconv.SparseConvTensor, torch.Tensor]:
    """Mean-pool events that land in the same (batch, x, y, t) voxel.

    Args:
        features: (N, C) float event features.
        coords: (N, 3) int64 integer voxel coordinates [x, y, t].
        batch_idx: Either an int (single sample) or an (N,) int tensor with
            the batch index of every event.
        spatial_shape: [W, H, T] padded spatial dimensions for the sparse tensor.

    Returns:
        voxel_tensor: SparseConvTensor with features (M, C) and indices
            (M, 4) as [batch, x, y, t].
        p2v_map: (N,) long tensor mapping each event to its voxel index.
    """
    n = coords.shape[0]
    coords = coords.long().contiguous()

    if isinstance(batch_idx, int):
        batch_col = torch.full((n, 1), batch_idx, dtype=torch.long, device=coords.device)
    else:
        batch_col = batch_idx.view(-1, 1).long().to(coords.device)

    keys = torch.cat([batch_col, coords], dim=1)  # (N, 4) [batch, x, y, t]

    unique_keys, p2v_map = torch.unique(keys, dim=0, return_inverse=True)
    m = unique_keys.shape[0]
    c = features.shape[1]

    features = features.float().contiguous()
    voxel_feats = torch.zeros(m, c, dtype=features.dtype, device=features.device)
    voxel_feats.scatter_add_(0, p2v_map.unsqueeze(1).expand(n, c), features)

    counts = torch.zeros(m, dtype=features.dtype, device=features.device)
    counts.scatter_add_(0, p2v_map, torch.ones(n, dtype=features.dtype, device=features.device))
    voxel_feats = voxel_feats / counts.clamp(min=1).unsqueeze(1)

    batch_size = int(unique_keys[:, 0].max().item()) + 1
    voxel_tensor = spconv.SparseConvTensor(
        features=voxel_feats,
        indices=unique_keys.int().contiguous(),
        spatial_shape=list(spatial_shape),
        batch_size=batch_size,
    )
    return voxel_tensor, p2v_map


def collate_events(batch: list[dict], spatial_shape: Sequence[int]) -> dict:
    """Stack per-sample dicts into a single sparse batch.

    Each sample must be a dict with keys:
        features: (N, C) float
        coords:   (N, 3) int64 [x, y, t]
        labels:   (N,)   float/int binary labels
    Optional:
        idx:      (N,) sample-level metadata carried through unchanged

    Returns:
        {
          "voxel_tensor": SparseConvTensor,
          "labels":       (N_total,) float tensor of event-level labels,
          "p2v_map":      (N_total,) long tensor mapping event -> voxel,
          "idx":          (N_total,) if present on every sample,
        }
    """
    features_list, coords_list, labels_list, idx_list, batch_idx_list = [], [], [], [], []

    for i, sample in enumerate(batch):
        n = sample["coords"].shape[0]
        features_list.append(sample["features"])
        coords_list.append(sample["coords"])
        labels_list.append(sample["labels"])
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        if "idx" in sample:
            idx_list.append(sample["idx"])

    features = torch.cat(features_list, dim=0).float().contiguous()
    coords = torch.cat(coords_list, dim=0).long().contiguous()
    labels = torch.cat(labels_list, dim=0).float().contiguous()
    batch_idx = torch.cat(batch_idx_list, dim=0)

    voxel_tensor, p2v_map = voxelize_events(features, coords, batch_idx, spatial_shape)

    out = {
        "voxel_tensor": voxel_tensor,
        "labels": labels,
        "p2v_map": p2v_map,
    }
    if len(idx_list) == len(batch):
        out["idx"] = torch.cat(idx_list, dim=0)
    return out
