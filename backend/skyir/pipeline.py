"""Per-video processing session: decode -> (detect || GMC) -> track -> range estimate.

Stages overlap: a reader thread prefetches frames, and camera-motion estimation (CPU, OpenCV
releases the GIL) runs concurrently with detection on the GPU. Each processed frame becomes a
`FrameResult` that the server streams to the browser or the exporter renders to disk.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .distance import DistanceEstimator
from .tracker import GMC, SkyTracker, TrackerConfig


@dataclass
class Params:
    """Everything the UI can change. Detection/track/camera fields apply live."""

    conf: float = 0.1  # detector floor (low-score dets still feed ByteTrack stage 2)
    nms_iou: float = 0.45
    track_high: float = 0.5
    new_track: float = 0.55
    match_thresh: float = 0.85
    track_buffer_s: float = 4.0
    altitude_m: float = 120.0
    pitch_deg: float = 35.0
    focal_px: float = 1470.0
    gmc: bool = True
    classes: list[int] | None = None  # None = all

    def update(self, d: dict) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class TrackInfo:
    id: int
    box: list[float]
    score: float
    cls: int
    name: str
    locked: bool
    coasting: bool
    age: int
    ground_m: float | None
    slant_m: float | None
    speed_px: float


@dataclass
class FrameResult:
    index: int  # 0-based
    time_s: float
    frame: np.ndarray  # BGR, original resolution
    tracks: list[TrackInfo]
    n_dets: int
    motion: list[float]  # camera translation dx, dy (px/frame)
    warp: list[float]  # full 2x3 camera motion prev->current, row-major [a, b, tx, c, d, ty]
    timings: dict[str, float] = field(default_factory=dict)  # ms

    def header(self) -> dict:
        return {
            "index": self.index,
            "time": round(self.time_s, 3),
            "tracks": [asdict(t) for t in self.tracks],
            "dets": self.n_dets,
            "motion": self.motion,
            "warp": self.warp,
            "timings": self.timings,
        }


class FrameReader:
    """Background decoder with a small prefetch queue."""

    def __init__(self, source: str, prefetch: int = 8):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps < 1 or self.fps > 240:
            self.fps = 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            ok, frame = self.cap.read()
            item = (frame, (time.perf_counter() - t) * 1000) if ok else None
            while not self._stop.is_set():
                try:
                    self.q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if item is None:
                break
        self.cap.release()

    def read(self) -> tuple[np.ndarray, float] | None:
        return self.q.get()

    def close(self) -> None:
        self._stop.set()


class Session:
    """Stateful tracker over one video source. Not thread-safe; drive it from one thread."""

    def __init__(self, source: str, detector, params: Params | None = None):
        self.det = detector
        self.params = params or Params()
        self.source = source
        self.names = detector.names
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gmc")
        self._open()

    def _open(self) -> None:
        self.reader = FrameReader(self.source)
        self.fps = self.reader.fps
        self.width, self.height, self.total = self.reader.width, self.reader.height, self.reader.total
        self.tracker = SkyTracker(len(self.names), self._tracker_cfg())
        self.gmc = GMC(downscale=2 if max(self.width, self.height) > 400 else 1)
        self.dist = DistanceEstimator(self.params.focal_px, self.height)
        self.index = -1

    def _tracker_cfg(self) -> TrackerConfig:
        p = self.params
        return TrackerConfig(
            track_high=p.track_high, new_track=max(p.new_track, p.track_high), match_thresh=p.match_thresh,
            track_buffer_s=p.track_buffer_s, track_low=p.conf, fps=self.fps,
        )

    def set_params(self, d: dict) -> None:
        self.params.update(d)
        self.tracker.cfg = self._tracker_cfg()
        self.dist = DistanceEstimator(self.params.focal_px, self.height)

    def restart(self) -> None:
        """Rewind to frame 0 with a fresh tracker (IDs restart, so locks are cleared)."""
        self.reader.close()
        self._open()

    def toggle_lock(self, track_id: int) -> bool:
        on = track_id not in self.tracker.locked_ids
        self.tracker.lock(track_id, on)
        return on

    def clear_locks(self) -> None:
        for tid in list(self.tracker.locked_ids):
            self.tracker.lock(tid, False)

    def close(self) -> None:
        self.reader.close()
        self._pool.shutdown(wait=False)

    def step(self) -> FrameResult | None:
        t0 = time.perf_counter()
        item = self.reader.read()
        if item is None:
            return None
        frame, decode_ms = item
        self.index += 1
        p = self.params

        t1 = time.perf_counter()
        fut = self._pool.submit(self.gmc.estimate, frame) if p.gmc else None
        dets = self.det.detect(frame, conf=p.conf, iou=p.nms_iou)
        t2 = time.perf_counter()
        M = fut.result() if fut else None
        t3 = time.perf_counter()

        if p.classes is not None and len(dets):
            keep = np.isin(dets.classes, p.classes)
            dets.xyxy, dets.scores, dets.classes = dets.xyxy[keep], dets.scores[keep], dets.classes[keep]
        if p.gmc:
            self.gmc.set_boxes(dets.xyxy)
        outs = self.tracker.update(dets.xyxy, dets.scores, dets.classes, M)
        t4 = time.perf_counter()

        tracks = []
        for o in outs:
            g, s = self.dist.estimate(float(o.xyxy[3]), p.altitude_m, p.pitch_deg)
            tracks.append(TrackInfo(
                id=o.id, box=[round(float(v), 1) for v in o.xyxy], score=round(o.score, 3), cls=o.cls,
                name=self.names.get(o.cls, str(o.cls)), locked=o.locked, coasting=o.coasting, age=o.age,
                ground_m=None if g is None else round(g, 1), slant_m=None if s is None else round(s, 1),
                speed_px=round(float(np.hypot(*o.velocity)), 2),
            ))
        motion = [0.0, 0.0] if M is None else [round(float(M[0, 2]), 2), round(float(M[1, 2]), 2)]
        warp = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] if M is None else [round(float(v), 6) for v in M.ravel()]
        t5 = time.perf_counter()
        timings = {
            "decode": round(decode_ms, 2),
            "wait": round((t1 - t0) * 1000, 2),
            "detect": round((t2 - t1) * 1000, 2),
            "gmc_extra": round((t3 - t2) * 1000, 2),  # GMC time NOT hidden behind detection
            "track": round((t4 - t3) * 1000, 2),
            "total": round((t5 - t1) * 1000, 2),
        }
        return FrameResult(self.index, self.index / self.fps, frame, tracks, len(dets), motion, warp, timings)
