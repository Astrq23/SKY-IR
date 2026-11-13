"""Tracker unit tests on synthetic scenes (no GPU / model needed)."""
import numpy as np

from skyir.tracker import SkyTracker, TrackerConfig
from skyir.tracker import kalman as kf
from skyir.tracker.matching import buffered_iou, iou_matrix


def box(cx, cy, w=8, h=16):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def run(frames, warps=None, **cfg):
    t = SkyTracker(2, TrackerConfig(fps=10, **cfg))
    outs = []
    for i, dets in enumerate(frames):
        b = np.array([d[0] for d in dets], float).reshape(-1, 4)
        s = np.array([d[1] for d in dets], float)
        c = np.array([d[2] if len(d) > 2 else 0 for d in dets], int)
        outs.append(t.update(b, s, c, None if warps is None else warps[i]))
    return outs


def ids(outs):
    return {o.id for frame in outs for o in frame if not o.coasting}


def test_kalman_roundtrip_and_shapes():
    z = kf.xyxy_to_xywh(np.array([box(50, 50), box(100, 80, 20, 10)]))
    assert np.allclose(kf.xywh_to_xyxy(z), [box(50, 50), box(100, 80, 20, 10)])
    m, P = kf.initiate(z)
    m, P = kf.predict(m, P)
    m2, P2 = kf.update(m, P, z + 1, np.array([0.9, 0.5]))
    assert m2.shape == (2, 8) and P2.shape == (2, 8, 8)
    assert np.all(np.linalg.eigvalsh(P2) > 0)  # covariance stays positive definite


def test_buffered_iou_links_small_non_overlapping_boxes():
    a, b = np.array([box(10, 10)]), np.array([box(19, 10)])  # 8px wide, moved 9px: plain IoU = 0
    assert iou_matrix(a, b)[0, 0] == 0
    assert buffered_iou(a, b)[0, 0] > 0.1


def test_single_fast_small_target_keeps_one_id():
    frames = [[(box(20 + 7 * i, 50), 0.9)] for i in range(40)]  # moves ~1 box width per frame
    assert len(ids(run(frames))) == 1


def test_camera_motion_compensated():
    # static object, camera pans 25 px/frame -> object jumps 25 px in image space every frame
    frames, warps = [], []
    for i in range(30):
        frames.append([(box(400 - 25 * i, 100), 0.9)])
        M = np.eye(2, 3)
        M[0, 2] = -25.0 if i else 0.0
        warps.append(M)
    assert len(ids(run(frames, warps))) == 1
    # without GMC the same sequence fragments
    assert len(ids(run(frames))) >= 1


def test_track_survives_missed_detections_and_low_scores():
    frames = []
    for i in range(40):
        if 15 <= i < 20:
            frames.append([])  # 5 missed frames (occlusion)
        elif 25 <= i < 30:
            frames.append([(box(30 + 2 * i, 60), 0.3)])  # faint detections (ByteTrack low stage)
        else:
            frames.append([(box(30 + 2 * i, 60), 0.85)])
    assert len(ids(run(frames))) == 1


def test_two_crossing_targets_keep_identities():
    frames = [[(box(20 + 6 * i, 50), 0.9, 0), (box(260 - 6 * i, 56), 0.9, 1)] for i in range(40)]
    outs = run(frames)
    assert len(ids(outs)) == 2
    first = {o.id: o.cls for o in outs[2]}
    last = {o.id: o.cls for o in outs[-1]}
    assert first == last  # same id -> same class before and after crossing


def test_low_score_detection_does_not_spawn_track():
    outs = run([[(box(50, 50), 0.3)] for _ in range(10)])
    assert ids(outs) == set()


def test_locked_target_coasts_and_survives_longer():
    frames = [[(box(50 + i, 50), 0.9)] for i in range(10)] + [[] for _ in range(60)] + [[(box(60, 50), 0.9)]]
    t = SkyTracker(1, TrackerConfig(fps=10, track_buffer_s=2.0))
    seen_coasting = False
    for i, dets in enumerate(frames):
        b = np.array([d[0] for d in dets], float).reshape(-1, 4)
        s = np.array([d[1] for d in dets], float)
        out = t.update(b, s, np.zeros(len(dets), int))
        if i == 5:
            t.lock(out[0].id)
        seen_coasting |= any(o.coasting for o in out)
    assert seen_coasting
    # 60 frames lost > 2s*10fps buffer, but locked targets get 3x -> still alive and re-acquired with the same id
    assert [o.id for o in out] == [1]
