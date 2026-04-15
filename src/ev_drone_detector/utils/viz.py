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
