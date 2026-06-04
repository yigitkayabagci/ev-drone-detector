"""Visualization utilities for event-based drone detection."""

from __future__ import annotations

import numpy as np


def events_to_frame(
    events_x: np.ndarray,
    events_y: np.ndarray,
    resolution: tuple[int, int] = (346, 260),
    target_xy: np.ndarray | None = None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    bg_event_color: tuple[int, int, int] = (110, 110, 110),
    target_color: tuple[int, int, int] = (0, 0, 200),
) -> np.ndarray:
    """Render events onto a frame, paper-style (EV-UAV Fig. 1).

    White background, all events in dark gray, and the detected drone events
    (`target_xy` — the model's positive predictions, NOT a whole bbox) painted in
    red on top. Colors are **BGR** to match OpenCV's imwrite / VideoWriter, the
    primary output path in detect.py.

    Args:
        events_x: (N,) x pixel coordinates of ALL events.
        events_y: (N,) y pixel coordinates of ALL events.
        resolution: (W, H) image size.
        target_xy: optional (M, 2) pixel coords [x, y] of the drone events to
            paint in `target_color` (red). Only these points turn red.
        bg_color: background fill (BGR). Default white.
        bg_event_color: color of all (background) events (BGR). Default dark gray.
        target_color: color of the drone events (BGR). Default red.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    W, H = resolution
    frame = np.full((H, W, 3), bg_color, dtype=np.uint8)

    x = np.clip(np.asarray(events_x).astype(int), 0, W - 1)
    y = np.clip(np.asarray(events_y).astype(int), 0, H - 1)

    # All events: dark gray background points.
    frame[y, x] = bg_event_color

    # The detected drone events: red, painted on top of the gray.
    if target_xy is not None and len(target_xy) > 0:
        txy = np.asarray(target_xy)
        tx = np.clip(txy[:, 0].astype(int), 0, W - 1)
        ty = np.clip(txy[:, 1].astype(int), 0, H - 1)
        frame[ty, tx] = target_color

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
