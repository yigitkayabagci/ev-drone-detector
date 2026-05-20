"""Event clustering for bounding box extraction.

After SPGNet segments events into target/background, this module
clusters the positive (drone) events spatially to find bounding boxes.
Uses DBSCAN clustering projected onto the (x, y) image plane.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def cluster_events_to_bbox(
    events_xy: np.ndarray,
    scores: np.ndarray | None = None,
    events_t: np.ndarray | None = None,
    extrapolate_to_t: float | None = None,
    eps: float = 10.0,
    min_samples: int = 3,
    min_cluster_size: int = 5,
    bbox_padding: int = 5,
    max_detections: int = 10,
    image_size: tuple[int, int] = (346, 260),
) -> list[dict]:
    """Cluster positive events and extract bounding boxes.

    By default emits a "trail" bbox covering the spatial extent of each
    cluster across the entire window. If both `events_t` and
    `extrapolate_to_t` are provided, fits a per-cluster linear motion
    model and returns a bbox centered at the drone's estimated position
    at `extrapolate_to_t` (e.g. the end of a sliding window). Useful for
    sliding-window tracking where the drone may have moved several
    pixels across the 8 s window.

    Args:
        events_xy: (N, 2) array of event (x, y) pixel coordinates.
        scores: (N,) confidence scores for each event (optional).
        events_t: (N,) per-event time coords (same units as
            `extrapolate_to_t`). Required for motion extrapolation.
        extrapolate_to_t: Target time at which the bbox should be predicted.
            None disables motion fitting and emits the legacy trail bbox.
        eps: DBSCAN epsilon (max distance between points in a cluster).
        min_samples: DBSCAN minimum samples to form a core point.
        min_cluster_size: Minimum events to form a valid detection.
        bbox_padding: Padding around detected bbox in pixels.
        max_detections: Maximum number of detections to return.
        image_size: (W, H) sensor resolution for clamping.

    Returns:
        List of detection dicts with keys:
        - 'bbox': [x_min, y_min, x_max, y_max]
        - 'score': mean confidence score
        - 'num_events': number of events in cluster
        - 'center': [cx, cy] center of bounding box
        - 'velocity': [vx, vy] pixels per t-unit (only if extrapolated)
    """
    if len(events_xy) == 0:
        return []

    extrapolate = extrapolate_to_t is not None and events_t is not None

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(events_xy)
    cluster_labels = clustering.labels_

    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)

    detections = []
    W, H = image_size

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_points = events_xy[mask]

        if len(cluster_points) < min_cluster_size:
            continue

        velocity: tuple[float, float] | None = None
        if extrapolate:
            bbox, velocity = _extrapolate_cluster_bbox(
                cluster_points,
                events_t[mask],
                float(extrapolate_to_t),
                bbox_padding=bbox_padding,
                image_size=(W, H),
            )
        else:
            x_min = int(cluster_points[:, 0].min()) - bbox_padding
            y_min = int(cluster_points[:, 1].min()) - bbox_padding
            x_max = int(cluster_points[:, 0].max()) + bbox_padding
            y_max = int(cluster_points[:, 1].max()) + bbox_padding

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(W - 1, x_max)
            y_max = min(H - 1, y_max)
            bbox = [x_min, y_min, x_max, y_max]

        if scores is not None:
            cluster_scores = scores[mask]
            score = float(cluster_scores.mean())
        else:
            score = 1.0

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        det = {
            "bbox": bbox,
            "score": score,
            "num_events": int(mask.sum()),
            "center": [cx, cy],
        }
        if velocity is not None:
            det["velocity"] = list(velocity)
        detections.append(det)

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections[:max_detections]


def _extrapolate_cluster_bbox(
    cluster_xy: np.ndarray,
    cluster_t: np.ndarray,
    target_t: float,
    bbox_padding: int,
    image_size: tuple[int, int],
    min_half_size: float = 4.0,
) -> tuple[list[int], tuple[float, float]]:
    """Linear motion fit on a cluster → predicted bbox at `target_t`.

    The center comes from a least-squares fit x(t)=ax+bx*t (and likewise
    for y). The half-extent uses 2σ of residuals about the fitted line,
    floored at `min_half_size` so a stationary drone still gets a
    drone-sized box.

    Returns:
        bbox: [x_min, y_min, x_max, y_max] integer pixel box.
        velocity: (vx, vy) pixels per unit time.
    """
    W, H = image_size
    n = cluster_xy.shape[0]

    t = cluster_t.astype(np.float64)
    x = cluster_xy[:, 0].astype(np.float64)
    y = cluster_xy[:, 1].astype(np.float64)

    t_mean = float(t.mean())
    dt = t - t_mean
    var_t = float((dt * dt).sum())

    if n < 4 or var_t < 1e-9:
        # Not enough temporal spread for a motion fit → trail bbox at centroid.
        cx = float(x.mean())
        cy = float(y.mean())
        half_w = max((float(x.max()) - float(x.min())) / 2.0, min_half_size)
        half_h = max((float(y.max()) - float(y.min())) / 2.0, min_half_size)
        vx = vy = 0.0
    else:
        bx = float((dt * x).sum() / var_t)
        by = float((dt * y).sum() / var_t)
        ax = float(x.mean())
        ay = float(y.mean())
        cx = ax + bx * (target_t - t_mean)
        cy = ay + by * (target_t - t_mean)

        # Residuals around the fitted trajectory line.
        res_x = x - (ax + bx * dt)
        res_y = y - (ay + by * dt)
        half_w = max(2.0 * float(res_x.std()), min_half_size)
        half_h = max(2.0 * float(res_y.std()), min_half_size)
        vx, vy = bx, by

    x_min = int(round(cx - half_w)) - bbox_padding
    y_min = int(round(cy - half_h)) - bbox_padding
    x_max = int(round(cx + half_w)) + bbox_padding
    y_max = int(round(cy + half_h)) + bbox_padding

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W - 1, x_max)
    y_max = min(H - 1, y_max)
    return [x_min, y_min, x_max, y_max], (vx, vy)


def segmentation_to_detections(
    predictions: torch.Tensor,
    coords: torch.Tensor,
    p2v_map: torch.Tensor,
    threshold: float = 0.5,
    extrapolate_to_t: float | None = None,
    eps: float = 10.0,
    min_samples: int = 3,
    min_cluster_size: int = 5,
    bbox_padding: int = 5,
    max_detections: int = 10,
    image_size: tuple[int, int] = (346, 260),
) -> list[dict]:
    """Convert per-voxel predictions to bounding box detections.

    Full pipeline: threshold → map to events → project to (x,y) → cluster → bbox.
    When `extrapolate_to_t` is set, fits a linear motion model per cluster
    and outputs a bbox at the predicted position at `extrapolate_to_t`
    (in voxel-t units — see `cluster_events_to_bbox`).

    Args:
        predictions: (N_voxels, 1) sigmoid probabilities.
        coords: (N_events, 3) event coordinates [x, y, t].
        p2v_map: (N_events,) point-to-voxel mapping.
        threshold: Segmentation threshold.
        extrapolate_to_t: If set, predict bbox at this voxel-t (e.g.
            whole_t - 1 for "end of window").
        eps, min_samples, min_cluster_size, bbox_padding, max_detections:
            Clustering parameters (see cluster_events_to_bbox).
        image_size: (W, H) sensor resolution.

    Returns:
        List of detection dicts.
    """
    # Map voxel predictions to event-level
    event_scores = predictions.squeeze(-1)[p2v_map].detach().cpu().numpy()

    # Threshold
    positive_mask = event_scores >= threshold
    if not positive_mask.any():
        return []

    # Get (x, y) coordinates of positive events
    coords_np = coords.cpu().numpy()
    positive_xy = coords_np[positive_mask, :2]  # (x, y) only
    positive_scores = event_scores[positive_mask]
    positive_t = coords_np[positive_mask, 2] if extrapolate_to_t is not None else None

    return cluster_events_to_bbox(
        positive_xy,
        scores=positive_scores,
        events_t=positive_t,
        extrapolate_to_t=extrapolate_to_t,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        bbox_padding=bbox_padding,
        max_detections=max_detections,
        image_size=image_size,
    )
