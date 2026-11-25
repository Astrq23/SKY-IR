"""Per-video processing session: decode -> (detect || GMC) -> size gate -> track -> ground geometry.

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

from .distance import GroundModel, plausible_size, size_range
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
    size_check: bool = True  # demote/relabel detections whose metric size does not fit their class
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
    ground_m: float | None  # horizontal range camera -> target foot point
    slant_m: float | None
    speed_px: float
    ground_xy: list[float] | None  # [right, forward] metres from the point below the camera
    speed_ms: float | None  # ground speed (needs GMC on to be relative to the ground)
    heading_deg: float | None  # 0 = away from camera, 90 = to the right; None when ~stationary
    vel_ms: list[float] | None  # [right, forward] m/s
    size_m: float | None  # largest visible extent
    size_ok: bool


@dataclass
class FrameResult:
    index: int  # 0-based
    time_s: float
    frame: np.ndarray  # BGR, original resolution
    tracks: list[TrackInfo]
    n_dets: int
    motion: list[float]  # camera translation dx, dy (px/frame)
    warp: list[float]  # full 2x3 camera motion prev->current, row-major [a, b, tx, c, d, ty]
    footprint: list[list[float]] = field(default_factory=list)  # ground polygon seen by the camera
    size_rejects: int = 0  # detections demoted or relabelled by the size gate this frame
    timings: dict[str, float] = field(default_factory=dict)  # ms

    def header(self) -> dict:
        return {
            "index": self.index,
            "time": round(self.time_s, 3),
            "tracks": [asdict(t) for t in self.tracks],
            "dets": self.n_dets,
            "motion": self.motion,
            "warp": self.warp,
            "footprint": self.footprint,
            "size_rejects": self.size_rejects,
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


# class confusions the size gate may fix: a "bicycle" the size of a car is almost always a car
RELABEL_IF_TOO_LARGE = {"bicycle": "car"}
STATIONARY_MS = 0.4  # below this the heading is noise
# Box-edge jitter grows with box size: a 100 px parked car wobbles ~0.5 px/frame. A target only
# counts as moving when its foot point travels more than this fraction of its box size per second.
STATIONARY_BOX_FRAC = 0.15
SPEED_TAU_S = 0.6  # time constant of the speed smoothing
MAX_MAP_RANGE_M = 1500.0
SIZE_SLACK_PX = 4.0  # thermal halo / box padding allowance


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
        self.geo = GroundModel(self.params.focal_px, self.width, self.height)
        self._vel: dict[int, np.ndarray] = {}  # smoothed ground velocity per track
        self._footprint: tuple[tuple, list] = ((), [])
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
        self.geo = GroundModel(self.params.focal_px, self.width, self.height)

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

    def _size_gate(self, dets) -> int:
        """Demote detections too small/large for their class below `track_high` (they may still
        extend an existing track but cannot start one); relabel known class confusions instead."""
        if not len(dets):
            return 0
        p = self.params
        b = dets.xyxy
        _, _, slant, ok = self.geo.project((b[:, 0] + b[:, 2]) / 2, b[:, 3], p.altitude_m, p.pitch_deg)
        mpp = self.geo.metres_per_pixel(slant)
        size = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * mpp
        by_name = {n.lower(): k for k, n in self.names.items()}
        changed = 0
        for i in np.where(ok)[0]:
            name = self.names.get(int(dets.classes[i]), "").lower()
            slack = SIZE_SLACK_PX * mpp[i]
            if plausible_size(name, size[i], slack):
                continue
            changed += 1
            alt = RELABEL_IF_TOO_LARGE.get(name)
            if alt in by_name and size[i] > size_range(name)[1] and plausible_size(alt, size[i], slack):
                dets.classes[i] = by_name[alt]
            else:
                dets.scores[i] = min(dets.scores[i], p.track_high - 0.01) * 0.5
        return changed

    def _geometry(self, outs) -> list[TrackInfo]:
        p = self.params
        if not outs:
            self._vel.clear()
            return []
        box = np.array([o.xyxy for o in outs], float)
        vel = np.array([o.velocity for o in outs], float)  # vcx, vcy, vw, vh
        fu, fv = (box[:, 0] + box[:, 2]) / 2, box[:, 3]  # foot point
        right, fwd, slant, ok = self.geo.project(fu, fv, p.altitude_m, p.pitch_deg)
        mpp = self.geo.metres_per_pixel(slant)
        size = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]) * mpp
        vr, vf = self.geo.ground_velocity(fu, fv, vel[:, 0], vel[:, 1] + vel[:, 3] / 2, p.altitude_m, p.pitch_deg, self.fps)
        a = 1.0 - np.exp(-1.0 / (self.fps * SPEED_TAU_S))
        box_px = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
        px_s = np.hypot(vel[:, 0], vel[:, 1] + vel[:, 3] / 2) * self.fps  # foot-point image speed
        alive, tracks = {}, []
        for i, o in enumerate(outs):
            name = self.names.get(o.cls, str(o.cls))
            g = speed = heading = v_ms = sz = None
            if ok[i]:
                g = round(float(np.hypot(right[i], fwd[i])), 1)
                sz = round(float(size[i]), 2)
                if np.isfinite(vr[i]):
                    raw = np.array([vr[i], vf[i], px_s[i]])
                    prev = self._vel.get(o.id)
                    sm = raw if prev is None else prev + a * (raw - prev)
                    alive[o.id] = sm
                    speed = round(float(np.hypot(sm[0], sm[1])), 2)
                    v_ms = [round(float(sm[0]), 2), round(float(sm[1]), 2)]
                    if speed >= STATIONARY_MS and sm[2] >= STATIONARY_BOX_FRAC * box_px[i]:
                        heading = round(float(np.degrees(np.arctan2(sm[0], sm[1]))) % 360, 1)
            tracks.append(TrackInfo(
                id=o.id, box=[round(float(v), 1) for v in o.xyxy], score=round(o.score, 3), cls=o.cls,
                name=name, locked=o.locked, coasting=o.coasting, age=o.age,
                ground_m=g, slant_m=round(float(slant[i]), 1) if ok[i] else None,
                speed_px=round(float(np.hypot(vel[i, 0], vel[i, 1])), 2),
                ground_xy=[round(float(right[i]), 1), round(float(fwd[i]), 1)] if ok[i] else None,
                speed_ms=speed, heading_deg=heading, vel_ms=v_ms, size_m=sz,
                size_ok=bool(sz is None or plausible_size(name, sz, SIZE_SLACK_PX * mpp[i])),
            ))
        # keep smoothing state for tracks that are only hidden for a moment
        live = {t.id for t in self.tracker.tracks}
        self._vel = {k: v for k, v in self._vel.items() if k in live} | alive
        return tracks

    def footprint(self) -> list[list[float]]:
        p = self.params
        key = (p.altitude_m, p.pitch_deg, p.focal_px)
        if self._footprint[0] != key:
            self._footprint = (key, self.geo.footprint(p.altitude_m, p.pitch_deg, min(MAX_MAP_RANGE_M, 12 * max(p.altitude_m, 1))))
        return self._footprint[1]

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
        rejects = self._size_gate(dets) if p.size_check else 0
        if p.gmc:
            self.gmc.set_boxes(dets.xyxy)
        outs = self.tracker.update(dets.xyxy, dets.scores, dets.classes, M)
        t4 = time.perf_counter()

        tracks = self._geometry(outs)
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
        return FrameResult(self.index, self.index / self.fps, frame, tracks, len(dets), motion, warp,
                           self.footprint(), rejects, timings)
