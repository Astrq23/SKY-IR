"""Flat-ground camera geometry: pixel -> ground position, range, metric size and ground velocity.

Model: pinhole camera (principal point at the image centre, square pixels, no roll) at height
`altitude_m` above a flat ground plane, tilted down by `pitch_deg` (0 = horizon, 90 = nadir).

Ground frame (metres), origin at the point directly below the camera:
    right   = along the image x axis
    forward = horizontal direction the camera looks at
A pixel's viewing ray in camera coordinates is (u - cx, v - cy, f); rotating it by the pitch gives
    down    = (v - cy) cos(p) + f sin(p)
    forward = f cos(p) - (v - cy) sin(p)
and the ray reaches the ground after scaling by altitude / down.
"""

from __future__ import annotations

import math

import numpy as np

# rays shallower than this (radians below the horizon) are treated as never hitting the ground
MIN_DEPRESSION = 0.01

# Plausible largest visible extent (metres) per class name, generous on purpose: the box includes
# blur/halo around small thermal blobs and oblique views mix height with footprint.
SIZE_RANGES: dict[str, tuple[float, float]] = {
    "person": (0.15, 2.6),
    "bicycle": (0.3, 3.0),
    "car": (1.0, 9.0),
    "othervehicle": (1.2, 22.0),
}


class GroundModel:
    def __init__(self, focal_px: float, width: int, height: int):
        self.f = max(float(focal_px), 1e-6)
        self.cx = width / 2
        self.cy = height / 2
        self.width, self.height = width, height

    def project(self, u, v, altitude_m: float, pitch_deg: float):
        """Pixels -> (right, forward, slant, valid), each an array shaped like `u`.

        right/forward are ground coordinates (m), slant the 3-D distance camera -> point (m).
        Entries whose ray misses the ground have valid=False and NaN coordinates.
        """
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        p = math.radians(min(abs(pitch_deg), 90.0))
        xu, yv = u - self.cx, v - self.cy
        down = yv * math.cos(p) + self.f * math.sin(p)
        fwd = self.f * math.cos(p) - yv * math.sin(p)
        norm = np.sqrt(xu**2 + yv**2 + self.f**2)
        valid = down > math.sin(MIN_DEPRESSION) * norm
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(valid, altitude_m / down, np.nan)
        return xu * t, fwd * t, norm * t, valid

    def metres_per_pixel(self, slant_m):
        """Size of one pixel, perpendicular to the viewing ray, at the given slant distance."""
        return np.asarray(slant_m, float) / self.f

    def ground_velocity(self, u, v, du, dv, altitude_m: float, pitch_deg: float, fps: float):
        """Image velocity (px/frame) at pixel (u, v) -> ground velocity (right, forward) in m/s."""
        r0, f0, _, ok0 = self.project(u, v, altitude_m, pitch_deg)
        r1, f1, _, ok1 = self.project(np.asarray(u) + du, np.asarray(v) + dv, altitude_m, pitch_deg)
        ok = ok0 & ok1
        return np.where(ok, (r1 - r0) * fps, np.nan), np.where(ok, (f1 - f0) * fps, np.nan)

    def footprint(self, altitude_m: float, pitch_deg: float, max_range_m: float, samples: int = 12):
        """Ground polygon [[right, forward], ...] seen by the camera, clipped at `max_range_m`.

        Walks the image border; rows above the horizon are replaced by the farthest visible row.
        """
        w, h = self.width, self.height
        # lowest row (largest v) is always closest; find the first row whose ray lands within range
        rows = np.linspace(0, h, 4 * samples + 1)
        _, fwd, _, ok = self.project(np.full_like(rows, self.cx), rows, altitude_m, pitch_deg)
        good = ok & (fwd <= max_range_m)
        if not good.any():
            return []
        top = rows[np.argmax(good)]
        xs = np.linspace(0, w, samples + 1)
        ys = np.linspace(top, h, samples + 1)
        border_u = np.concatenate([xs, np.full_like(ys, w), xs[::-1], np.zeros_like(ys)])
        border_v = np.concatenate([np.full_like(xs, top), ys, np.full_like(xs, h), ys[::-1]])
        r, fw, _, ok = self.project(border_u, border_v, altitude_m, pitch_deg)
        return [[round(float(a), 1), round(float(b), 1)] for a, b, k in zip(r, fw, ok) if k]


def size_range(name: str) -> tuple[float, float] | None:
    return SIZE_RANGES.get(name.lower().replace(" ", "").replace("_", ""))


def plausible_size(name: str, size_m: float, slack_m: float = 0.0) -> bool:
    """True when `size_m` fits the class (unknown classes always fit). `slack_m` widens both ends."""
    r = size_range(name)
    return bool(r is None or r[0] - slack_m <= size_m <= r[1] + slack_m)
