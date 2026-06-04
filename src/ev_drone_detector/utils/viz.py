"""Visualization utilities for event-based drone detection."""

from __future__ import annotations

import numpy as np


def events_to_frame(
    events_x: np.ndarray,
    events_y: np.ndarray,
    resolution: tuple[int, int] = (346, 260),
    target_xy: np.ndarray | None = None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    bg_event_color: tuple[int, int, int] = (175, 175, 175),
    target_color: tuple[int, int, int] = (0, 0, 210),
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
    color: tuple[int, int, int] = (0, 160, 0),
    thickness: int = 2,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw bounding boxes + readable score labels on a frame.

    The box is a fairly dark, high-contrast green (reads on the white background),
    and the score sits in a FILLED label (green fill, white text) so it stays
    legible over events of any color. Colors are BGR.

    Args:
        frame: (H, W, 3) image.
        detections: List of dicts with 'bbox' [x_min,y_min,x_max,y_max] and 'score'.
        color: BGR box/label color.
        thickness: Box line thickness.
        text_color: BGR color of the score text inside the filled label.

    Returns:
        Frame with drawn boxes and labels.
    """
    H, W = frame.shape[:2]
    try:
        import cv2
    except ImportError:
        # Fallback: draw boxes manually without OpenCV (no text labels)
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            out[y1:y1 + thickness, x1:x2] = color
            out[y2 - thickness:y2, x1:x2] = color
            out[y1:y2, x1:x1 + thickness] = color
            out[y1:y2, x2 - thickness:x2] = color
        return out

    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['score']:.2f}"
        (tw, th), base = cv2.getTextSize(label, font, scale, 1)
        # Put the label above the box; if there's no room, drop it just below the top edge.
        if y1 - th - 6 >= 0:
            ly1, ly2 = y1 - th - 6, y1
        else:
            ly1, ly2 = y1, y1 + th + 6
        lx1 = x1
        lx2 = min(x1 + tw + 6, W - 1)
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)  # filled label bg
        cv2.putText(out, label, (lx1 + 3, ly2 - 4), font, scale,
                    text_color, 1, cv2.LINE_AA)
    return out
