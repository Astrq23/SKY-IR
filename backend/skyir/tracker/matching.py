"""Association costs and linear assignment."""

from __future__ import annotations

import numpy as np

try:
    import lap

    def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Min-cost matching with a rejection threshold. Returns (matches[K,2], unmatched_rows, unmatched_cols)."""
        if cost.size == 0:
            return np.empty((0, 2), int), np.arange(cost.shape[0]), np.arange(cost.shape[1])
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
        matches = np.array([[i, j] for i, j in enumerate(x) if j >= 0], dtype=int).reshape(-1, 2)
        return matches, np.where(x < 0)[0], np.where(y < 0)[0]

except ImportError:  # pragma: no cover
    from scipy.optimize import linear_sum_assignment

    def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cost.size == 0:
            return np.empty((0, 2), int), np.arange(cost.shape[0]), np.arange(cost.shape[1])
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] <= thresh
        matches = np.stack([r[ok], c[ok]], axis=1)
        return matches, np.setdiff1d(np.arange(cost.shape[0]), matches[:, 0]), np.setdiff1d(np.arange(cost.shape[1]), matches[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def buffer_boxes(b: np.ndarray, ratio: float, min_px: float) -> np.ndarray:
    """Expand each box by ``ratio`` of its size on every side (at least ``min_px``)."""
    wh = b[:, 2:] - b[:, :2]
    pad = np.maximum(wh * ratio, min_px)
    return np.concatenate([b[:, :2] - pad, b[:, 2:] + pad], axis=1)


def buffered_iou(tracks: np.ndarray, dets: np.ndarray, ratio: float = 0.3, min_px: float = 3.0) -> np.ndarray:
    """C-BIoU (Yang et al., WACV 2023): IoU of enlarged boxes.

    Small, fast targets often have zero overlap between consecutive frames; enlarging both
    boxes restores a usable similarity signal while still preferring the closest candidate.
    """
    return iou_matrix(buffer_boxes(tracks, ratio, min_px), buffer_boxes(dets, ratio, min_px))


def center_similarity(tracks: np.ndarray, dets: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """exp(-d^2 / s^2) where d is centre distance normalised by the track's diagonal."""
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))
    ct = (tracks[:, :2] + tracks[:, 2:]) / 2
    cd = (dets[:, :2] + dets[:, 2:]) / 2
    diag = np.maximum(np.hypot(*(tracks[:, 2:] - tracks[:, :2]).T), 4.0)
    d = np.linalg.norm(ct[:, None] - cd[None], axis=2) / diag[:, None]
    return np.exp(-(d**2) / scale**2)


def association_cost(
    track_boxes: np.ndarray,
    track_cls: np.ndarray,
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    det_cls: np.ndarray,
    buffer: float,
    fuse_score: bool = True,
    class_penalty: float = 0.15,
) -> np.ndarray:
    """1 - similarity, where similarity mixes buffered IoU with centre proximity.

    Class disagreement adds a soft penalty instead of a hard gate, because the detector often
    flickers between e.g. Person/Bicycle on 10-pixel targets.
    """
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(track_boxes), len(det_boxes)))
    biou = buffered_iou(track_boxes, det_boxes, buffer)
    sim = 0.75 * biou + 0.25 * center_similarity(track_boxes, det_boxes) * (biou > 0)
    if fuse_score:
        sim = sim * (0.5 + 0.5 * det_scores[None, :])
    cost = 1.0 - sim
    cost += class_penalty * (track_cls[:, None] != det_cls[None, :])
    return cost
