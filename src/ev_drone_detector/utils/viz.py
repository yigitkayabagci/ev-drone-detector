"""Visualization utilities for event-based drone detection."""

from __future__ import annotations

import numpy as np


def events_to_frame(
    events_x: np.ndarray,
    events_y: np.ndarray,
    resolution: tuple[int, int] = (346, 260),
    bboxes: list | None = None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    bg_event_color: tuple[int, int, int] = (70, 70, 70),
    target_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Render events onto a frame, paper-style (EV-UAV Fig. 1).

    White background, background events in dark gray, and events that fall inside
    any detection bbox drawn in red (the target). Colors are **BGR** to match
    OpenCV's imwrite / VideoWriter, which is the primary output path in detect.py.

    Args:
        events_x: (N,) x pixel coordinates.
        events_y: (N,) y pixel coordinates.
        resolution: (W, H) image size.
        bboxes: optional list of [x_min, y_min, x_max, y_max]. Events inside any
            of these are drawn in `target_color` (the detected drone, red).
        bg_color: background fill (BGR). Default white.
        bg_event_color: color of background events (BGR). Default dark gray.
        target_color: color of events inside a bbox (BGR). Default red.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    W, H = resolution
    frame = np.full((H, W, 3), bg_color, dtype=np.uint8)

    x = np.clip(np.asarray(events_x).astype(int), 0, W - 1)
    y = np.clip(np.asarray(events_y).astype(int), 0, H - 1)

    # All events: dark gray background points.
    frame[y, x] = bg_event_color

    # Events inside a detected bbox: red target, painted on top.
    if bboxes:
        red = np.zeros(x.shape[0], dtype=bool)
        for x1, y1, x2, y2 in bboxes:
            red |= (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
        if red.any():
            frame[y[red], x[red]] = target_color

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
