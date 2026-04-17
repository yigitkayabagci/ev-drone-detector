"""Convert EV-UAV .npz event windows into 2D event frames.

FRED and most frame-based event detectors expect a 2D image (or a small
time-sliced stack) rather than a raw event stream. This script bins the
events in each .npz file along the temporal axis and produces one image
per bin. Output can be PNG (for visual inspection / standard image
pipelines) or .npy (for direct tensor loading).

Two modes:
    single   — one frame per .npz file (all events collapsed). Fast,
               gives one image per sample. Good starting point.
    windowed — split each .npz by --window_ms into N sub-frames.

Two channel layouts:
    polarity_rgb (default) — H x W x 3 uint8. Positive events written
        to the blue channel, negative to the red channel. Matches the
        existing scripts/detect.py visualization and is directly
        consumable by any RGB-input network.
    counts_2ch            — 2 x H x W float32. Channel 0 = positive
        event count, channel 1 = negative event count. This is the
        standard input for event-frame CNN detectors.

Expected .npz layout (EV-UAV):
    ev_loc    (N, 3)  int    [x, y, t]
    evs_norm  (N, >=4) float  [x_norm, y_norm, t_norm, polarity, ...]

Usage:
    uv run python frame_detector/events_to_frames.py \\
        --input data/test \\
        --output frames/test \\
        --mode single --format png --channels polarity_rgb

    uv run python frame_detector/events_to_frames.py \\
        --input data/test \\
        --output frames/test \\
        --mode windowed --window_ms 33.3 --format npy --channels counts_2ch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def accumulate_polarity_rgb(
    x: np.ndarray, y: np.ndarray, polarity: np.ndarray, resolution: tuple[int, int],
) -> np.ndarray:
    """Render events as an H x W x 3 uint8 image (gray bg, pos=blue, neg=red)."""
    W, H = resolution
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    xi = np.clip(x.astype(np.int64), 0, W - 1)
    yi = np.clip(y.astype(np.int64), 0, H - 1)
    pos = polarity > 0
    neg = ~pos
    frame[yi[pos], xi[pos]] = [255, 200, 200]
    frame[yi[neg], xi[neg]] = [200, 200, 255]
    return frame


def accumulate_counts_2ch(
    x: np.ndarray, y: np.ndarray, polarity: np.ndarray, resolution: tuple[int, int],
) -> np.ndarray:
    """Render events as a 2 x H x W float32 count map (pos, neg)."""
    W, H = resolution
    frame = np.zeros((2, H, W), dtype=np.float32)
    xi = np.clip(x.astype(np.int64), 0, W - 1)
    yi = np.clip(y.astype(np.int64), 0, H - 1)
    pos = polarity > 0
    neg = ~pos
    np.add.at(frame[0], (yi[pos], xi[pos]), 1.0)
    np.add.at(frame[1], (yi[neg], xi[neg]), 1.0)
    return frame


def render_frame(
    x: np.ndarray, y: np.ndarray, polarity: np.ndarray,
    resolution: tuple[int, int], channels: str,
) -> np.ndarray:
    if channels == "polarity_rgb":
        return accumulate_polarity_rgb(x, y, polarity, resolution)
    if channels == "counts_2ch":
        return accumulate_counts_2ch(x, y, polarity, resolution)
    raise ValueError(f"Unknown channels layout: {channels}")


def write_frame(frame: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(str(path.with_suffix(".npy")), frame)
        return
    if fmt == "png":
        if frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8:
            img = frame
        elif frame.ndim == 3 and frame.shape[0] == 2:
            a = np.clip(frame[0], 0, 255).astype(np.uint8)
            b = np.clip(frame[1], 0, 255).astype(np.uint8)
            img = np.stack([b, np.zeros_like(a), a], axis=-1)
        else:
            raise ValueError(f"Cannot PNG-encode frame with shape {frame.shape}")
        try:
            import cv2
            cv2.imwrite(str(path.with_suffix(".png")), img)
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(str(path.with_suffix(".png")), img)
        return
    raise ValueError(f"Unknown format: {fmt}")


def process_npz(
    npz_path: Path, out_dir: Path, *,
    mode: str, window_ms: float, resolution: tuple[int, int],
    channels: str, fmt: str,
) -> int:
    data = np.load(str(npz_path), allow_pickle=True)
    ev_loc = data["ev_loc"]
    evs_norm = data["evs_norm"]
    x = ev_loc[:, 0]
    y = ev_loc[:, 1]
    t = ev_loc[:, 2].astype(np.float64)
    polarity = evs_norm[:, 3]

    stem = npz_path.stem

    if mode == "single":
        frame = render_frame(x, y, polarity, resolution, channels)
        write_frame(frame, out_dir / stem, fmt)
        return 1

    if mode == "windowed":
        if t.size == 0:
            return 0
        t_min = float(t.min())
        t_max = float(t.max())
        if t_max <= t_min:
            return 0
        n_bins = max(int(np.ceil((t_max - t_min) / window_ms)), 1)
        edges = t_min + np.arange(n_bins + 1) * window_ms
        bin_idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
        written = 0
        for i in range(n_bins):
            mask = bin_idx == i
            if not mask.any():
                continue
            frame = render_frame(x[mask], y[mask], polarity[mask], resolution, channels)
            write_frame(frame, out_dir / f"{stem}_f{i:04d}", fmt)
            written += 1
        return written

    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert EV-UAV .npz to event frames")
    p.add_argument("--input", required=True, help=".npz file or directory")
    p.add_argument("--output", required=True, help="Output directory for frames")
    p.add_argument("--mode", choices=["single", "windowed"], default="single")
    p.add_argument("--window_ms", type=float, default=33.3,
                   help="Window size for --mode windowed (default 33.3ms = ~30 FPS)")
    p.add_argument("--resolution", nargs=2, type=int, default=[346, 260],
                   metavar=("W", "H"), help="Sensor resolution (default DAVIS346)")
    p.add_argument("--channels", choices=["polarity_rgb", "counts_2ch"],
                   default="polarity_rgb")
    p.add_argument("--format", choices=["png", "npy"], default="png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = (int(args.resolution[0]), int(args.resolution[1]))

    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(in_path.glob("*.npz"))
        if not files:
            files = sorted(in_path.rglob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found under {in_path}")

    total_frames = 0
    for f in files:
        sub_dir = out_dir if args.mode == "single" else out_dir / f.stem
        n = process_npz(
            f, sub_dir,
            mode=args.mode, window_ms=args.window_ms,
            resolution=resolution, channels=args.channels, fmt=args.format,
        )
        total_frames += n
        print(f"{f.name}: {n} frame(s) -> {sub_dir}")

    print(f"Done. {len(files)} file(s), {total_frames} frame(s) total.")


if __name__ == "__main__":
    main()
