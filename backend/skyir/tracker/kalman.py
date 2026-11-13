"""Batched constant-velocity Kalman filter in (cx, cy, w, h) space.

State: [cx, cy, w, h, vcx, vcy, vw, vh]. Noise is proportional to box size (BoT-SORT) with an
absolute pixel floor so the filter does not become over-confident on tiny (5-15 px) thermal
targets, and the measurement noise is scaled by detection confidence (NSA Kalman, GIAOTracker).
All operations are vectorised over N tracks: means (N, 8), covariances (N, 8, 8).
"""

from __future__ import annotations

import numpy as np

NDIM = 4
_F = np.eye(2 * NDIM)
_F[:NDIM, NDIM:] = np.eye(NDIM)
_H = np.eye(NDIM, 2 * NDIM)

STD_POS = 1.0 / 20
STD_VEL = 1.0 / 160
MIN_POS_PX = 1.0  # floors for tiny boxes
MIN_VEL_PX = 0.25


def _wh4(wh: np.ndarray) -> np.ndarray:
    """(N, 2) -> (N, 4) [w, h, w, h]."""
    return np.concatenate([wh, wh], axis=1)


def initiate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z: (N, 4) xywh measurements -> means (N, 8), covs (N, 8, 8)."""
    n = len(z)
    mean = np.concatenate([z, np.zeros((n, NDIM))], axis=1)
    wh = _wh4(z[:, 2:4])
    std = np.concatenate([np.maximum(2 * STD_POS * wh, 2 * MIN_POS_PX), np.maximum(10 * STD_VEL * wh, 10 * MIN_VEL_PX)], axis=1)
    cov = np.einsum("ni,ij->nij", std**2, np.eye(2 * NDIM))
    return mean, cov


def predict(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(mean) == 0:
        return mean, cov
    wh = _wh4(mean[:, 2:4])
    std = np.concatenate([np.maximum(STD_POS * wh, MIN_POS_PX), np.maximum(STD_VEL * wh, MIN_VEL_PX)], axis=1)
    Q = np.einsum("ni,ij->nij", std**2, np.eye(2 * NDIM))
    mean = mean @ _F.T
    cov = _F @ cov @ _F.T + Q
    mean[:, 2:4] = np.maximum(mean[:, 2:4], 1.0)  # keep boxes valid
    return mean, cov


def update(mean: np.ndarray, cov: np.ndarray, z: np.ndarray, score: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Kalman correction for N (track, measurement) pairs."""
    if len(mean) == 0:
        return mean, cov
    wh = _wh4(mean[:, 2:4])
    std = np.maximum(STD_POS * wh, MIN_POS_PX)
    R = np.einsum("ni,ij->nij", std**2, np.eye(NDIM))
    if score is not None:  # NSA: confident detections are trusted more
        R = R * (1.0 - np.clip(score, 0.0, 0.95))[:, None, None] * 2.0
    S = _H @ cov @ _H.T + R  # (N, 4, 4)
    PHt = cov @ _H.T  # (N, 8, 4)
    K = np.linalg.solve(S, PHt.transpose(0, 2, 1)).transpose(0, 2, 1)  # (N, 8, 4)
    innov = z - mean[:, :NDIM]
    mean = mean + np.einsum("nij,nj->ni", K, innov)
    cov = cov - K @ S @ K.transpose(0, 2, 1)
    return mean, cov


def apply_affine(mean: np.ndarray, cov: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp states by a 2x3 camera-motion matrix (applied to position, size and velocity)."""
    if len(mean) == 0:
        return mean, cov
    R = M[:, :2]
    R8 = np.kron(np.eye(4), R)
    mean = mean @ R8.T
    mean[:, :2] += M[:, 2]
    cov = R8 @ cov @ R8.T
    mean[:, 2:4] = np.abs(mean[:, 2:4])
    return mean, cov


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
