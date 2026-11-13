"""SkyTrack: multi-object tracker tuned for small targets seen from a moving UAV.

Pipeline per frame (ByteTrack / BoT-SORT family):
  1. Predict all tracks with a batched Kalman filter, then warp them with camera motion (GMC).
  2. High-score detections <-> confirmed + lost tracks, cost = buffered IoU + centre proximity.
  3. Low-score detections <-> remaining *tracked* tracks (recovers occluded / faint targets).
  4. Remaining high detections <-> unconfirmed tracks.
  5. Unmatched tracks become lost (kept `track_buffer` seconds, longer for locked targets);
     unmatched high detections spawn new tracks.

Differences from the previous SORT-style tracker:
  * GMC warps position, size, velocity *and* covariance (was: centre only, after the fact).
  * Lost tracks can be re-matched, so a missed frame no longer creates a new ID.
  * Buffered IoU keeps similarity non-zero for 8-px targets that move a few pixels per frame.
  * Class is a confidence-weighted vote, robust to per-frame class flicker.
  * IDs are per-session (not a global class counter) and selected targets can be locked live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kalman as kf
from .matching import association_cost, iou_matrix, linear_assignment

TRACKED, LOST, REMOVED = 0, 1, 2


@dataclass
class TrackerConfig:
    track_high: float = 0.5  # detections above this are "high" (stage 1, can start tracks)
    track_low: float = 0.1  # detections in [low, high) only extend existing tracks
    new_track: float = 0.55  # min score to start a new track
    match_thresh: float = 0.85  # max cost for stage 1
    low_match_thresh: float = 0.6
    unconfirmed_thresh: float = 0.75
    buffer_high: float = 0.3  # C-BIoU buffer ratios
    buffer_low: float = 0.15
    track_buffer_s: float = 4.0  # how long lost tracks are kept (seconds)
    locked_buffer_factor: float = 3.0
    n_init: int = 2  # consecutive hits before a track is confirmed
    lost_velocity_decay: float = 0.9
    min_box_px: float = 2.0
    fps: float = 30.0


@dataclass
class Track:
    id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    cls_votes: np.ndarray
    state: int = TRACKED
    confirmed: bool = False
    hits: int = 1
    age: int = 1
    frames_lost: int = 0
    start_frame: int = 0
    locked: bool = False
    history: list = field(default_factory=list)

    @property
    def cls(self) -> int:
        return int(np.argmax(self.cls_votes))

    @property
    def xyxy(self) -> np.ndarray:
        return kf.xywh_to_xyxy(self.mean[None, :4])[0]

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[4:6]


@dataclass
class TrackOutput:
    id: int
    xyxy: np.ndarray
    score: float
    cls: int
    age: int
    locked: bool
    coasting: bool  # predicted only (no detection this frame)
    velocity: np.ndarray


class SkyTracker:
    def __init__(self, num_classes: int, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.nc = num_classes
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1
        self.locked_ids: set[int] = set()

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        self.tracks.clear()
        self.frame_id = 0
        self._next_id = 1
        self.locked_ids.clear()

    def lock(self, track_id: int, on: bool = True) -> None:
        (self.locked_ids.add if on else self.locked_ids.discard)(track_id)
        for t in self.tracks:
            if t.id == track_id:
                t.locked = on

    def update(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, warp: np.ndarray | None = None) -> list[TrackOutput]:
        cfg = self.cfg
        self.frame_id += 1
        wh = boxes[:, 2:] - boxes[:, :2] if len(boxes) else np.zeros((0, 2))
        valid = (scores >= cfg.track_low) & (wh.min(1) >= cfg.min_box_px) if len(boxes) else np.zeros(0, bool)
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        high = scores >= cfg.track_high
        low = ~high

        active = [t for t in self.tracks if t.state != REMOVED]
        self._predict(active, warp)

        tracked = [t for t in active if t.state == TRACKED and t.confirmed]
        unconfirmed = [t for t in active if t.state == TRACKED and not t.confirmed]
        lost = [t for t in active if t.state == LOST]

        hi_idx = np.where(high)[0]
        lo_idx = np.where(low)[0]
        matched_tracks: set[int] = set()

        # stage 1: high dets vs tracked + lost
        pool = tracked + lost
        m, u_trk, u_det = self._match(pool, hi_idx, boxes, scores, classes, cfg.buffer_high, cfg.match_thresh)
        updates = [(pool[i], hi_idx[j]) for i, j in m]
        remaining_hi = hi_idx[u_det]
        remain_tracked = [pool[i] for i in u_trk if pool[i].state == TRACKED]
        remain_lost = [pool[i] for i in u_trk if pool[i].state == LOST]

        # stage 2: low dets vs still-unmatched tracked tracks
        m2, u_trk2, _ = self._match(remain_tracked, lo_idx, boxes, scores, classes, cfg.buffer_low, cfg.low_match_thresh, fuse=False)
        updates += [(remain_tracked[i], lo_idx[j]) for i, j in m2]
        missed = [remain_tracked[i] for i in u_trk2]

        # stage 3: unconfirmed vs remaining high dets
        m3, u_unc, u_det3 = self._match(unconfirmed, remaining_hi, boxes, scores, classes, cfg.buffer_high, cfg.unconfirmed_thresh)
        updates += [(unconfirmed[i], remaining_hi[j]) for i, j in m3]
        remaining_hi = remaining_hi[u_det3]

        self._correct(updates, boxes, scores, classes)
        matched_tracks.update(t.id for t, _ in updates)

        for t in missed + remain_lost:
            t.state = LOST
            t.frames_lost += 1
        for i in u_unc:  # an unconfirmed track that misses once is dropped
            unconfirmed[i].state = REMOVED

        # new tracks (suppress spawns that duplicate an existing track)
        if len(remaining_hi):
            cand = remaining_hi[scores[remaining_hi] >= cfg.new_track]
            live = np.array([t.xyxy for t in self.tracks if t.state == TRACKED]).reshape(-1, 4)
            if len(cand) and len(live):
                cand = cand[iou_matrix(boxes[cand], live).max(1) < 0.5]
            self._spawn(cand, boxes, scores, classes)

        buf = cfg.track_buffer_s * cfg.fps
        for t in self.tracks:
            if t.state == LOST and t.frames_lost > buf * (cfg.locked_buffer_factor if t.locked else 1.0):
                t.state = REMOVED
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
        return self._outputs()

    # ------------------------------------------------------------------ internals
    def _predict(self, tracks: list[Track], warp: np.ndarray | None) -> None:
        if not tracks:
            return
        mean = np.stack([t.mean for t in tracks])
        cov = np.stack([t.cov for t in tracks])
        lost = np.array([t.state == LOST for t in tracks])
        mean[lost, 6:8] = 0.0  # don't let size drift while lost
        mean[lost, 4:6] *= self.cfg.lost_velocity_decay  # unseen targets slow down instead of flying off
        mean, cov = kf.predict(mean, cov)
        if warp is not None:
            mean, cov = kf.apply_affine(mean, cov, warp)
        for t, mu, c in zip(tracks, mean, cov):
            t.mean, t.cov = mu, c
            t.age += 1

    def _match(self, tracks, det_idx, boxes, scores, classes, buffer, thresh, fuse=True):
        if not tracks or len(det_idx) == 0:
            return np.empty((0, 2), int), np.arange(len(tracks)), np.arange(len(det_idx))
        tb = np.array([t.xyxy for t in tracks])
        tc = np.array([t.cls for t in tracks])
        cost = association_cost(tb, tc, boxes[det_idx], scores[det_idx], classes[det_idx], buffer, fuse_score=fuse)
        # lost tracks: widen the gate with time lost (uncertainty grows), but slightly prefer tracked ones
        lost_frames = np.array([t.frames_lost if t.state == LOST else 0 for t in tracks])
        if lost_frames.any():
            cost += 0.02 * np.minimum(lost_frames, 10)[:, None]
        return linear_assignment(cost, thresh)

    def _correct(self, pairs: list[tuple[Track, int]], boxes, scores, classes) -> None:
        if not pairs:
            return
        tracks = [t for t, _ in pairs]
        di = np.array([j for _, j in pairs])
        mean = np.stack([t.mean for t in tracks])
        cov = np.stack([t.cov for t in tracks])
        mean, cov = kf.update(mean, cov, kf.xyxy_to_xywh(boxes[di]), scores[di])
        for t, mu, c, j in zip(tracks, mean, cov, di):
            t.mean, t.cov = mu, c
            t.score = float(scores[j])
            t.cls_votes[classes[j]] += scores[j]
            t.cls_votes *= 0.97  # slowly forget old votes
            t.hits += 1
            t.frames_lost = 0
            t.state = TRACKED
            if not t.confirmed and t.hits >= self.cfg.n_init:
                t.confirmed = True

    def _spawn(self, idx: np.ndarray, boxes, scores, classes) -> None:
        if len(idx) == 0:
            return
        mean, cov = kf.initiate(kf.xyxy_to_xywh(boxes[idx]))
        for mu, c, j in zip(mean, cov, idx):
            votes = np.zeros(self.nc)
            votes[classes[j]] = scores[j]
            t = Track(self._next_id, mu, c, float(scores[j]), votes, start_frame=self.frame_id)
            t.confirmed = self.frame_id == 1 or self.cfg.n_init <= 1
            self._next_id += 1
            self.tracks.append(t)

    def _outputs(self) -> list[TrackOutput]:
        out = []
        for t in self.tracks:
            if not t.confirmed:
                continue
            visible = t.state == TRACKED and t.frames_lost == 0
            coasting = t.state == LOST and t.locked and t.frames_lost <= self.cfg.fps  # show locked target ~1s predicted
            if visible or coasting:
                out.append(TrackOutput(t.id, t.xyxy.copy(), t.score, t.cls, t.age, t.locked, coasting, t.velocity.copy()))
        return out
