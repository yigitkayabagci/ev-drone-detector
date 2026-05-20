"""Detection-to-track association via IoU + Hungarian assignment.

Pure NumPy/SciPy — no torch or spconv dependency, so this module can be
unit-tested without the full model pipeline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(
    boxes_a: Sequence[Sequence[float]],
    boxes_b: Sequence[Sequence[float]],
) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes.

    Args:
        boxes_a: (Na, 4) iterable of axis-aligned boxes.
        boxes_b: (Nb, 4) iterable of axis-aligned boxes.

    Returns:
        (Na, Nb) IoU matrix. Empty input yields a zero-shaped array.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_by_iou(
    track_boxes: Sequence[Sequence[float]],
    det_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy-optimal track <-> detection assignment via Hungarian on IoU.

    Args:
        track_boxes: List of current track bboxes (one entry per active track).
        det_boxes:   List of new detection bboxes for this window.
        iou_threshold: Pairs with IoU below this are rejected even if Hungarian
            picks them — keeps spurious cross-window matches out.

    Returns:
        matches:           list of (track_idx, det_idx) pairs that passed the IoU gate.
        unmatched_tracks:  track indices with no matching detection this window.
        unmatched_dets:    detection indices that did not match any existing track.
    """
    nt, nd = len(track_boxes), len(det_boxes)
    if nt == 0:
        return [], [], list(range(nd))
    if nd == 0:
        return [], list(range(nt)), []

    iou = iou_matrix(track_boxes, det_boxes)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_threshold:
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

    unmatched_tracks = [i for i in range(nt) if i not in matched_tracks]
    unmatched_dets = [i for i in range(nd) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets
