"""Server-side rendering, used only for exported videos (the live UI draws overlays in the browser)."""

from __future__ import annotations

import cv2
import numpy as np

PALETTES = {
    "white-hot": None,
    "black-hot": "invert",
    "ironbow": cv2.COLORMAP_INFERNO,
    "arctic": cv2.COLORMAP_OCEAN,
    "rainbow": cv2.COLORMAP_TURBO,
}

# BGR, chosen to stay readable on grey thermal imagery and under any palette
CLASS_COLORS = [(80, 220, 255), (255, 190, 60), (120, 255, 120), (255, 120, 220), (180, 180, 180)]
LOCK_COLOR = (60, 60, 255)


def apply_palette(frame: np.ndarray, palette: str) -> np.ndarray:
    mode = PALETTES.get(palette)
    if mode is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "invert":
        return cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, mode)


def _corners(img, x1, y1, x2, y2, color, t):
    L = int(max(4, min(x2 - x1, y2 - y1) * 0.3))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), color, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, t, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks, style: str = "corners", show_locked_only: bool = False) -> np.ndarray:
    img = frame
    any_locked = any(t.locked for t in tracks)
    for t in tracks:
        if show_locked_only and any_locked and not t.locked:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in t.box)
        # small targets: draw the box a little larger than the object so it stays visible
        pad = 3 if max(x2 - x1, y2 - y1) < 24 else 1
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        color = LOCK_COLOR if t.locked else CLASS_COLORS[t.cls % len(CLASS_COLORS)]
        th = 2 if t.locked else 1
        if t.coasting:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)
        elif style == "box":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)
        else:
            _corners(img, x1, y1, x2, y2, color, th)
        label = f"{t.id} {t.name}"
        if t.ground_m is not None:
            label += f" {t.ground_m:.0f}m"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ly = y1 - 4 if y1 - tht - 6 > 0 else y2 + tht + 4
        cv2.rectangle(img, (x1, ly - tht - 3), (x1 + tw + 4, ly + 3), (20, 20, 20), -1)
        cv2.putText(img, label, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return img


def draw_hud(img: np.ndarray, index: int, n_tracks: int, fps: float) -> np.ndarray:
    txt = f"F{index:05d}  TRK {n_tracks:02d}  {fps:5.1f} FPS"
    cv2.rectangle(img, (6, 6), (12 + 8 * len(txt), 26), (20, 20, 20), -1)
    cv2.putText(img, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return img
