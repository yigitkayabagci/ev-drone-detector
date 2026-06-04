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
    eps: float = 10.0,
    min_samples: int = 3,
    min_cluster_size: int = 5,
    bbox_padding: int = 5,
    max_detections: int = 10,
    image_size: tuple[int, int] = (346, 260),
) -> list[dict]:
    """Cluster positive events and extract bounding boxes.

    Args:
        events_xy: (N, 2) array of event (x, y) pixel coordinates.
        scores: (N,) confidence scores for each event (optional).
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
    """
    if len(events_xy) == 0:
        return []

    # Run DBSCAN clustering on (x, y) coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(events_xy)
    cluster_labels = clustering.labels_

    # Extract unique cluster IDs (ignore noise label -1)
    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)

    detections = []
    W, H = image_size

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_points = events_xy[mask]

        if len(cluster_points) < min_cluster_size:
            continue

        # Compute bounding box
        x_min = int(cluster_points[:, 0].min()) - bbox_padding
        y_min = int(cluster_points[:, 1].min()) - bbox_padding
        x_max = int(cluster_points[:, 0].max()) + bbox_padding
        y_max = int(cluster_points[:, 1].max()) + bbox_padding

        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W - 1, x_max)
        y_max = min(H - 1, y_max)

        # Compute confidence score
        if scores is not None:
            cluster_scores = scores[mask]
            score = float(cluster_scores.mean())
        else:
            score = 1.0

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        detections.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "score": score,
            "num_events": int(mask.sum()),
            "center": [cx, cy],
        })

    # Sort by score descending, take top-k
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections[:max_detections]


def segmentation_to_detections(
    predictions: torch.Tensor,
    coords: torch.Tensor,
    p2v_map: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 10.0,
    min_samples: int = 3,
    min_cluster_size: int = 5,
    bbox_padding: int = 5,
    max_detections: int = 10,
    image_size: tuple[int, int] = (346, 260),
) -> list[dict]:
    """Convert per-voxel predictions to bounding box detections.

    Full pipeline: threshold → map to events → project to (x,y) → cluster → bbox.

    Args:
        predictions: (N_voxels, 1) sigmoid probabilities.
        coords: (N_events, 3) event coordinates [x, y, t].
        p2v_map: (N_events,) point-to-voxel mapping.
        threshold: Segmentation threshold.
        eps, min_samples, min_cluster_size, bbox_padding, max_detections:
            Clustering parameters (see cluster_events_to_bbox).
        image_size: (W, H) sensor resolution.

    Returns:
        List of detection dicts.
    """
    # Map voxel predictions to event-level.
    # Move predictions to CPU *before* indexing so this works regardless of
    # whether `predictions` lives on CUDA and `p2v_map` on CPU (advanced
    # indexing requires both operands on the same device).
    preds_cpu = predictions.squeeze(-1).detach().cpu()
    p2v_cpu = p2v_map.detach().cpu()
    event_scores = preds_cpu[p2v_cpu].numpy()

    # Threshold
    positive_mask = event_scores >= threshold
    if not positive_mask.any():
        return []

    # Get (x, y) coordinates of positive events
    coords_np = coords.cpu().numpy()
    positive_xy = coords_np[positive_mask, :2]  # (x, y) only
    positive_scores = event_scores[positive_mask]

    return cluster_events_to_bbox(
        positive_xy,
        scores=positive_scores,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        bbox_padding=bbox_padding,
        max_detections=max_detections,
        image_size=image_size,
    )
