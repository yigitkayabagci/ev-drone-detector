"""Dataset loaders for EV-UAV and synthetic event data.

Adapted from the original Ev-UAV repository:
  https://github.com/ChenYichen9527/Ev-UAV/blob/main/dataset/ev_uav.py

Expected .npz layout (standard EV-UAV format):
  ev_loc:   (N, 3) int        integer voxel coords [x, y, t]
  evs_norm: (N, >=5) float    columns [x_norm, y_norm, t_norm, polarity, seg_label, idx?]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EvUAVDataset(Dataset):
    """Loads .npz event files from a directory.

    Args:
        root_dir: Directory containing the split's .npz files (e.g. data/train).
            If it contains no .npz files at the top level, rglob is used.
        max_events: Hard cap on events per training sample (random downsample).
            Ignored for val/test.
        split: One of "train", "val", "test". Only "train" downsamples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        max_events: int = 700000,
        split: str = "train",
    ):
        self.root_dir = Path(root_dir)
        self.max_events = max_events
        self.split = split

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir.resolve()}"
            )

        files = sorted(self.root_dir.glob("*.npz"))
        if not files:
            files = sorted(self.root_dir.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No .npz files found under {self.root_dir.resolve()}"
            )
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(str(self.files[idx]), allow_pickle=True)
        evs_norm = data["evs_norm"]
        ev_loc = data["ev_loc"]

        features = evs_norm[:, 0:4].astype(np.float32)
        if evs_norm.shape[1] >= 5:
            labels = evs_norm[:, 4].astype(np.float32)
        else:
            labels = np.zeros(features.shape[0], dtype=np.float32)

        if evs_norm.shape[1] >= 6:
            idx_arr = evs_norm[:, 5].astype(np.int64)
        else:
            idx_arr = None

        n = features.shape[0]
        if self.split == "train" and n > self.max_events:
            sel = np.random.choice(n, self.max_events, replace=False)
            features = features[sel]
            ev_loc = ev_loc[sel]
            labels = labels[sel]
            if idx_arr is not None:
                idx_arr = idx_arr[sel]

        out = {
            "features": torch.from_numpy(features),
            "coords": torch.from_numpy(ev_loc.astype(np.int64)),
            "labels": torch.from_numpy(labels),
        }
        if idx_arr is not None:
            out["idx"] = torch.from_numpy(idx_arr)
        return out


class SyntheticEventDataset(Dataset):
    """Deterministic synthetic event stream for pipeline smoke tests.

    Each sample is a mix of uniform background events plus a cluster of
    events simulating a tiny moving drone (labelled 1).

    Args:
        num_samples: Number of samples in the dataset.
        num_events: Events per sample.
        resolution: (W, H) sensor resolution for coord ranges.
        whole_t: Temporal window length (same units as ev_loc t).
        seed: Base seed; sample i uses seed + i.
    """

    def __init__(
        self,
        num_samples: int = 50,
        num_events: int = 5000,
        resolution: tuple[int, int] | list[int] = (346, 260),
        whole_t: int = 8000,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.num_events = num_events
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.T = int(whole_t)
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed * 100003 + idx)
        n = self.num_events
        n_drone = max(n // 5, 10)
        n_bg = n - n_drone

        bg_x = rng.integers(0, self.W, size=n_bg)
        bg_y = rng.integers(0, self.H, size=n_bg)
        bg_t = rng.integers(0, self.T, size=n_bg)
        bg_p = rng.choice([-1, 1], size=n_bg).astype(np.float32)

        cx = int(rng.integers(30, self.W - 30))
        cy = int(rng.integers(30, self.H - 30))
        t0 = int(rng.integers(0, max(self.T - 100, 1)))
        drone_x = np.clip(cx + rng.integers(-8, 9, size=n_drone), 0, self.W - 1)
        drone_y = np.clip(cy + rng.integers(-8, 9, size=n_drone), 0, self.H - 1)
        drone_t = np.clip(
            t0 + rng.integers(0, 100, size=n_drone), 0, self.T - 1
        )
        drone_p = rng.choice([-1, 1], size=n_drone).astype(np.float32)

        x = np.concatenate([bg_x, drone_x]).astype(np.int64)
        y = np.concatenate([bg_y, drone_y]).astype(np.int64)
        t = np.concatenate([bg_t, drone_t]).astype(np.int64)
        p = np.concatenate([bg_p, drone_p]).astype(np.float32)

        coords = np.stack([x, y, t], axis=1)
        features = np.stack(
            [
                x.astype(np.float32) / self.W,
                y.astype(np.float32) / self.H,
                t.astype(np.float32) / self.T,
                p,
            ],
            axis=1,
        ).astype(np.float32)
        labels = np.concatenate(
            [np.zeros(n_bg, dtype=np.float32), np.ones(n_drone, dtype=np.float32)]
        )

        return {
            "features": torch.from_numpy(features),
            "coords": torch.from_numpy(coords),
            "labels": torch.from_numpy(labels),
        }
