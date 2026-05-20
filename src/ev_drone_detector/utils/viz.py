"""Visualization utilities for event-based drone detection."""

from __future__ import annotations

import numpy as np


def events_to_frame(
    events_x: np.ndarray,
    events_y: np.ndarray,
    polarity: np.ndarray | None = None,
    resolution: tuple[int, int] = (346, 260),
) -> np.ndarray:
    """Convert events to a 2D frame image for visualization.

    Args:
        events_x: (N,) x coordinates.
        events_y: (N,) y coordinates.
        polarity: (N,) event polarities (+1/-1). If None, all white.
        resolution: (W, H) image size.

    Returns:
        (H, W, 3) uint8 image. Gray background, positive=blue, negative=red.
    """
    W, H = resolution
    frame = np.full((H, W, 3), 128, dtype=np.uint8)

    x = np.clip(events_x.astype(int), 0, W - 1)
    y = np.clip(events_y.astype(int), 0, H - 1)

    if polarity is not None:
        pos_mask = polarity > 0
        neg_mask = polarity <= 0
        frame[y[pos_mask], x[pos_mask]] = [255, 200, 200]  # Blue-ish
        frame[y[neg_mask], x[neg_mask]] = [200, 200, 255]  # Red-ish
    else:
        frame[y, x] = [255, 255, 255]

    return frame


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a frame.

    Args:
        frame: (H, W, 3) image.
        detections: List of dicts with 'bbox' key [x_min, y_min, x_max, y_max].
        color: BGR color for bounding box.
        thickness: Line thickness.

    Returns:
        Frame with drawn bounding boxes.
    """
    try:
        import cv2
    except ImportError:
        # Fallback: draw boxes manually without OpenCV
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Top and bottom edges
            out[y1:y1 + thickness, x1:x2] = color
            out[y2 - thickness:y2, x1:x2] = color
            # Left and right edges
            out[y1:y2, x1:x1 + thickness] = color
            out[y1:y2, x2 - thickness:x2] = color
        return out

    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        score_text = f"{det['score']:.2f}"
        cv2.putText(out, score_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
    return out


# Fixed BGR palette — distinct hues, cycles for >8 tracks. Indexed by
# (track_id - 1) so the first track gets the first color.
_TRACK_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),       # green
    (0, 165, 255),     # orange
    (255, 0, 255),     # magenta
    (255, 255, 0),     # cyan
    (0, 0, 255),       # red
    (255, 0, 0),       # blue
    (0, 255, 255),     # yellow
    (128, 0, 128),     # purple
)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color for a track ID (cycles through 8 hues)."""
    return _TRACK_PALETTE[(int(track_id) - 1) % len(_TRACK_PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    active_tracks: list[dict],
    trails: dict[int, list[tuple[float, float]]] | None = None,
    thickness: int = 2,
    show_velocity: bool = False,
) -> np.ndarray:
    """Render tracked drones on a frame, one color per track ID.

    Args:
        frame: (H, W, 3) BGR image.
        active_tracks: Per-window output of `SlidingWindowTracker.update`.
            Each dict needs at least 'track_id', 'bbox', 'score'.
        trails: Optional {track_id: [(cx, cy), ...]} — recent centers per
            track, oldest first. Drawn as a polyline that brightens toward
            the present.
        thickness: Bbox line thickness.
        show_velocity: If True and an active track carries a 'velocity'
            field (set by the extrapolating detector), draw a short arrow
            from the bbox center along the motion direction.

    Returns:
        New (H, W, 3) image with overlays drawn.
    """
    try:
        import cv2
    except ImportError:
        cv2 = None

    out = frame.copy()
    active_ids = {tr["track_id"] for tr in active_tracks}

    if trails is not None and cv2 is not None:
        for tid, pts in trails.items():
            if tid not in active_ids or len(pts) < 2:
                continue
            base_color = color_for_track(tid)
            n = len(pts)
            for i in range(1, n):
                alpha = (i + 1) / n  # newer = brighter
                dim = tuple(int(c * alpha) for c in base_color)
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(out, p1, p2, dim, max(1, int(thickness * alpha)))

    for tr in active_tracks:
        tid = tr["track_id"]
        x1, y1, x2, y2 = (int(v) for v in tr["bbox"])
        color = color_for_track(tid)

        if cv2 is not None:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{tid} s={tr.get('score', 0.0):.2f}"
            cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if show_velocity and "velocity" in tr:
                vx, vy = tr["velocity"]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # Scale so a typical velocity reads as ~20 px arrow.
                scale = 200.0
                ex = int(cx + vx * scale)
                ey = int(cy + vy * scale)
                cv2.arrowedLine(out, (cx, cy), (ex, ey), color, 1, tipLength=0.3)
        else:
            out[y1:y1 + thickness, x1:x2] = color
            out[y2 - thickness:y2, x1:x2] = color
            out[y1:y2, x1:x1 + thickness] = color
            out[y1:y2, x2 - thickness:x2] = color

    return out


class TrailBuffer:
    """Per-track-id ring buffer of recent bbox centers for trail rendering.

    The tracker only reports the *current* bbox per window; the visualizer
    needs the recent history to draw a trajectory line. Wrap the per-window
    update in `TrailBuffer.update(active_tracks)` and pass `buffer.trails`
    to `draw_tracks`.
    """

    def __init__(self, max_length: int = 20):
        self.max_length = int(max_length)
        self.trails: dict[int, list[tuple[float, float]]] = {}

    def update(self, active_tracks: list[dict]) -> None:
        for tr in active_tracks:
            tid = int(tr["track_id"])
            x1, y1, x2, y2 = tr["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            buf = self.trails.setdefault(tid, [])
            buf.append((cx, cy))
            if len(buf) > self.max_length:
                del buf[: len(buf) - self.max_length]
