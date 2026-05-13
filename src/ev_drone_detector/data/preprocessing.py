"""Generic event preprocessor: any (x, y, t, p) source -> model input.

DroneDetector accepts `(features, coords)` from any source — a DAVIS346 .npz,
a Prophesee EVK4 .hdf5 stream, a live event camera, synthetic events, etc.
This module is the bridge: it handles the three adaptations that always need
to happen before the model regardless of where the events came from.

  1. Polarity normalization: {0, 1} or {-1, +1} -> {-1, +1}
  2. Temporal windowing: slice the stream into fixed-length windows that
     match the temporal field the model was trained on.
  3. Spatial adaptation via sliding-window tiling: carve a high-resolution
     source image (e.g. 1280x720 EVK4) into target-resolution tiles
     (e.g. 346x260 DAVIS346) the model was trained on, so a small drone
     in the source stays roughly the same pixel size when fed to the model.

Outputs are per-tile, per-time-window `TileWindow` slices that drop straight
into `DroneDetector.detect(features, coords)`. Tile origin and time window
are returned alongside so detections can be mapped back to source coords.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class TileWindow:
    """One (spatial tile, time window) preprocessing slice ready for the model."""

    features: np.ndarray
    """(N, 4) float32 = [x_norm, y_norm, t_norm, polarity]."""

    coords: np.ndarray
    """(N, 3) int64 = [x, y, t_voxel] in TILE-LOCAL coordinates."""

    tile_origin: tuple[int, int]
    """(x0, y0) of the tile's top-left corner, in SOURCE pixel coords."""

    t_window_us: tuple[int, int]
    """(t_start, t_end) of the time window, in SOURCE timestamps (microseconds)."""


class EventPreprocessor:
    """Adapt a raw event stream to the model's (features, coords) input.

    Args:
        source_resolution: (W, H) of the input sensor.
            e.g. (1280, 720) for Prophesee EVK4, (346, 260) for DAVIS346.
        target_resolution: (W, H) the model was trained on. DAVIS346 by default.
        target_t_us: Length of one time window in microseconds (default 8 s,
            matching the EV-UAV `whole_t=8000` ms config).
        target_t_bins: Number of temporal voxel bins (must match the model's
            `spatial_shape[2]`; 8192 by default).
        tile_overlap: Fraction of overlap between adjacent tiles, in [0, 1).
            0.0 = non-overlapping tiles; 0.5 = 50% overlap so a drone on a
            tile boundary still falls inside one full tile.
        polarity_mode: "auto" (detect 0/1 vs -1/+1 from the data), "01",
            or "pm1".
        max_events_per_tile: Hard cap; tiles with more events are random-
            subsampled. None = no cap. Useful to stay under the model's
            training-time `max_events` budget.
    """

    def __init__(
        self,
        source_resolution: tuple[int, int],
        target_resolution: tuple[int, int] = (346, 260),
        target_t_us: int = 8_000_000,
        target_t_bins: int = 8192,
        tile_overlap: float = 0.0,
        polarity_mode: str = "auto",
        max_events_per_tile: int | None = None,
    ):
        if not (0.0 <= tile_overlap < 1.0):
            raise ValueError(f"tile_overlap must be in [0, 1), got {tile_overlap}")
        if polarity_mode not in ("auto", "01", "pm1"):
            raise ValueError(
                f"polarity_mode must be 'auto', '01', or 'pm1', got {polarity_mode!r}"
            )
        self.src_W, self.src_H = int(source_resolution[0]), int(source_resolution[1])
        self.tgt_W, self.tgt_H = int(target_resolution[0]), int(target_resolution[1])
        self.target_t_us = int(target_t_us)
        self.target_t_bins = int(target_t_bins)
        self.tile_overlap = float(tile_overlap)
        self.polarity_mode = polarity_mode
        self.max_events_per_tile = max_events_per_tile

    @classmethod
    def from_config(
        cls,
        config,
        source_resolution: tuple[int, int],
        **overrides,
    ) -> EventPreprocessor:
        """Build a preprocessor matching the model's config.

        Reads `sensor.resolution`, `sensor.whole_t` (ms), and
        `sensor.spatial_shape[2]` from a loaded Config.
        """
        kwargs = dict(
            source_resolution=source_resolution,
            target_resolution=tuple(config.sensor.resolution),
            target_t_us=int(config.sensor.whole_t) * 1000,
            target_t_bins=int(config.sensor.spatial_shape[2]),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def tile_grid(self) -> list[tuple[int, int]]:
        """Top-left corners (x0, y0) of every tile in source pixel coords.

        Guarantees that the union of tiles covers the full source image —
        if the source is not an integer multiple of the tile size, the
        rightmost and bottommost tiles are shifted in to reach the edge.
        """
        stride_x = max(1, int(self.tgt_W * (1.0 - self.tile_overlap)))
        stride_y = max(1, int(self.tgt_H * (1.0 - self.tile_overlap)))
        if self.src_W <= self.tgt_W:
            xs = [0]
        else:
            xs = list(range(0, self.src_W - self.tgt_W + 1, stride_x))
            if xs[-1] + self.tgt_W < self.src_W:
                xs.append(self.src_W - self.tgt_W)
        if self.src_H <= self.tgt_H:
            ys = [0]
        else:
            ys = list(range(0, self.src_H - self.tgt_H + 1, stride_y))
            if ys[-1] + self.tgt_H < self.src_H:
                ys.append(self.src_H - self.tgt_H)
        return [(x, y) for y in ys for x in xs]

    def normalize_polarity(self, p: np.ndarray) -> np.ndarray:
        """Map polarity to {-1.0, +1.0} regardless of input convention."""
        p = np.asarray(p)
        if self.polarity_mode == "pm1":
            return p.astype(np.float32)
        if self.polarity_mode == "01":
            return (p.astype(np.float32) * 2.0) - 1.0
        if (p < 0).any():
            return np.where(p > 0, 1.0, -1.0).astype(np.float32)
        return (p.astype(np.float32) * 2.0) - 1.0

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> Iterator[TileWindow]:
        """Stream `TileWindow` slices from a raw event stream.

        Args:
            x: (N,) integer pixel x coords in source resolution.
            y: (N,) integer pixel y coords in source resolution.
            t: (N,) int64 timestamps (microseconds). Reference frame doesn't
                matter — only differences are used.
            p: (N,) polarity. Encoding is auto-detected by default.

        Yields:
            One `TileWindow` per (tile, time-window) that actually contains
            events. Empty combinations are skipped.
        """
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        if not (len(x) == len(y) == len(t) == len(p)):
            raise ValueError(
                f"x, y, t, p must have the same length, got "
                f"{len(x)}, {len(y)}, {len(t)}, {len(p)}"
            )
        if len(x) == 0:
            return
        p = self.normalize_polarity(p)

        t = t - int(t.min())
        t_max = int(t.max())
        n_windows = max(1, t_max // self.target_t_us + 1)

        tiles = self.tile_grid()

        for w in range(n_windows):
            t_start = w * self.target_t_us
            t_end = t_start + self.target_t_us
            wmask = (t >= t_start) & (t < t_end)
            if not wmask.any():
                continue
            wx, wy, wt, wp = x[wmask], y[wmask], t[wmask], p[wmask]
            wt_rel = wt - t_start

            for (x0, y0) in tiles:
                tmask = (
                    (wx >= x0) & (wx < x0 + self.tgt_W)
                    & (wy >= y0) & (wy < y0 + self.tgt_H)
                )
                if not tmask.any():
                    continue
                tx = wx[tmask] - x0
                ty = wy[tmask] - y0
                tt = wt_rel[tmask]
                tp = wp[tmask]

                if (
                    self.max_events_per_tile is not None
                    and len(tx) > self.max_events_per_tile
                ):
                    sel = np.random.choice(
                        len(tx), self.max_events_per_tile, replace=False,
                    )
                    tx, ty, tt, tp = tx[sel], ty[sel], tt[sel], tp[sel]

                tt_vox = (tt.astype(np.int64) * self.target_t_bins) // max(
                    self.target_t_us, 1
                )
                tt_vox = np.clip(tt_vox, 0, self.target_t_bins - 1)

                features = np.stack(
                    [
                        tx.astype(np.float32) / max(self.tgt_W, 1),
                        ty.astype(np.float32) / max(self.tgt_H, 1),
                        tt.astype(np.float32) / max(self.target_t_us, 1),
                        tp.astype(np.float32),
                    ],
                    axis=1,
                )
                coords = np.stack([tx, ty, tt_vox], axis=1).astype(np.int64)

                yield TileWindow(
                    features=features,
                    coords=coords,
                    tile_origin=(int(x0), int(y0)),
                    t_window_us=(int(t_start), int(t_end)),
                )
