"""Flat-ground range estimate from camera pitch, altitude and the target's foot point."""

from __future__ import annotations

import math


class DistanceEstimator:
    def __init__(self, focal_px: float, image_height: int):
        self.f = max(float(focal_px), 1e-6)
        self.cy = image_height / 2

    def estimate(self, y_bottom: float, altitude_m: float, pitch_deg: float) -> tuple[float | None, float | None]:
        """Return (ground, slant) distance in metres; None when the ray does not hit the ground.

        pitch_deg: camera depression angle, 0 = horizon, 90 = nadir (sign is ignored).
        """
        angle = math.radians(abs(pitch_deg)) + math.atan((y_bottom - self.cy) / self.f)
        if angle <= 0.01:
            return None, None
        if angle >= math.pi / 2 - 1e-3:
            return 0.0, float(altitude_m)
        return altitude_m / math.tan(angle), altitude_m / math.sin(angle)
