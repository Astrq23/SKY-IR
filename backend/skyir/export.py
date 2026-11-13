"""Offline processing: render an annotated H.264 video, MOT tracks file and (optionally) metrics."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .pipeline import Params, Session
from .render import apply_palette, draw_hud, draw_tracks


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


class H264Writer:
    """Pipe raw BGR frames straight into ffmpeg (no intermediate mp4v file / re-encode)."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.proc = subprocess.Popen(
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def load_mot(path: str | Path) -> dict[int, list[tuple[int, list[float]]]]:
    """MOT txt (frame,id,x,y,w,h,...) -> {frame(1-based): [(id, [x,y,w,h])]}. Comma or whitespace separated."""
    out: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 6:
            continue
        try:
            f, i, x, y, w, h = (float(v) for v in parts[:6])
        except ValueError:
            continue
        out.setdefault(int(f), []).append((int(i), [x, y, w, h]))
    return out


def _iou_distance(gt_xywh: list, hyp_xywh: list, max_iou: float = 0.5) -> np.ndarray:
    """motmetrics-style distance (1 - IoU, NaN if IoU < max_iou); avoids motmetrics' NumPy-2-incompatible helper."""
    if not gt_xywh or not hyp_xywh:
        return np.empty((len(gt_xywh), len(hyp_xywh)))
    from .tracker.matching import iou_matrix

    g, h = np.asarray(gt_xywh, float), np.asarray(hyp_xywh, float)
    g[:, 2:] += g[:, :2]
    h[:, 2:] += h[:, :2]
    d = 1.0 - iou_matrix(g, h)
    d[d > max_iou] = np.nan
    return d


@dataclass
class ExportResult:
    video: Path
    tracks: Path
    frames: int
    seconds: float
    metrics: dict | None = None
    summary: dict = field(default_factory=dict)


def export_video(
    source: str, detector, out_dir: Path, params: Params, palette: str = "white-hot", style: str = "corners",
    locked_ids: list[int] | None = None, gt_path: str | None = None,
    progress: Callable[[int, int], None] | None = None, cancelled: Callable[[], bool] | None = None,
) -> ExportResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Session(source, detector, params)
    for tid in locked_ids or []:
        s.tracker.lock(tid)
    vpath, tpath = out_dir / "result.mp4", out_dir / "tracks.txt"
    writer = H264Writer(vpath, s.width, s.height, s.fps)
    gt = load_mot(gt_path) if gt_path else None
    acc = None
    if gt is not None:
        import motmetrics as mm

        acc = mm.MOTAccumulator(auto_id=True)
    ids: set[int] = set()
    t0 = time.perf_counter()
    n = 0
    try:
        with open(tpath, "w") as ft:
            while (r := s.step()) is not None:
                if cancelled and cancelled():
                    break
                n += 1
                live = [t for t in r.tracks if not t.coasting]
                for t in live:
                    x1, y1, x2, y2 = t.box
                    ft.write(f"{r.index + 1},{t.id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{t.score:.3f},{t.cls},-1,-1\n")
                    ids.add(t.id)
                if acc is not None:
                    import motmetrics as mm

                    g = gt.get(r.index + 1, [])
                    hyp = [[t.box[0], t.box[1], t.box[2] - t.box[0], t.box[3] - t.box[1]] for t in live]
                    dist = _iou_distance([b for _, b in g], hyp)
                    acc.update([i for i, _ in g], [t.id for t in live], dist)
                img = apply_palette(r.frame, palette)
                draw_tracks(img, r.tracks, style)
                draw_hud(img, r.index, len(live), n / max(time.perf_counter() - t0, 1e-6))
                writer.write(img)
                if progress and n % 5 == 0:
                    progress(n, s.total)
    finally:
        writer.close()
        s.close()
    secs = time.perf_counter() - t0
    metrics = None
    if acc is not None:
        import motmetrics as mm

        mh = mm.metrics.create()
        names = ["num_frames", "mota", "motp", "idf1", "idp", "idr", "mostly_tracked", "mostly_lost", "num_switches", "num_fragmentations"]
        df = mh.compute(acc, metrics=names, name="seq")
        metrics = {k: (None if np.isnan(v) else float(v)) for k, v in df.iloc[0].to_dict().items()}
    return ExportResult(vpath, tpath, n, secs, metrics, {"unique_ids": len(ids), "fps": n / max(secs, 1e-6)})
