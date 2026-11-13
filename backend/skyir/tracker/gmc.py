"""Global motion compensation (camera ego-motion) via sparse optical flow.

Shi-Tomasi corners are found on the previous (downscaled) frame, masked away from detected
objects so moving targets do not bias the estimate, tracked with pyramidal Lucas-Kanade and fitted
with a RANSAC similarity transform. Returns a 2x3 matrix mapping previous-frame pixels to current.
"""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.eye(2, 3, dtype=np.float64)


class GMC:
    def __init__(self, downscale: int = 2, max_corners: int = 400, min_inliers: int = 12):
        self.downscale = max(1, downscale)
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.prev: np.ndarray | None = None
        self.prev_boxes: np.ndarray | None = None
        self.last = IDENTITY.copy()

    def reset(self) -> None:
        self.prev = self.prev_boxes = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            g = cv2.resize(g, (g.shape[1] // self.downscale, g.shape[0] // self.downscale), interpolation=cv2.INTER_AREA)
        return g

    def _corners(self, gray: np.ndarray, boxes: np.ndarray | None) -> np.ndarray | None:
        mask = np.zeros_like(gray)
        h, w = gray.shape
        mask[int(0.02 * h) : int(0.98 * h), int(0.02 * w) : int(0.98 * w)] = 255
        if boxes is not None and len(boxes):
            for x1, y1, x2, y2 in (boxes / self.downscale).astype(int):
                mask[max(y1 - 2, 0) : y2 + 2, max(x1 - 2, 0) : x2 + 2] = 0
        return cv2.goodFeaturesToTrack(gray, self.max_corners, 0.01, 5, mask=mask, blockSize=3)

    def set_boxes(self, boxes: np.ndarray) -> None:
        """Detections of the frame last passed to ``estimate`` (masked out of the next corner search)."""
        self.prev_boxes = boxes

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Motion from the previous call's frame to ``frame``.

        Needs only pixels, so it can run on a CPU thread while the GPU detects the same frame.
        """
        gray = self._gray(frame)
        M = IDENTITY.copy()
        if self.prev is not None:
            pts = self._corners(self.prev, self.prev_boxes)
            if pts is not None and len(pts) >= self.min_inliers:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= self.min_inliers:
                    A, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                    if A is not None and inl is not None and inl.sum() >= self.min_inliers:
                        M = A.astype(np.float64)
                        M[:, 2] *= self.downscale
        self.prev = gray
        self.prev_boxes = None
        self.last = M
        return M
