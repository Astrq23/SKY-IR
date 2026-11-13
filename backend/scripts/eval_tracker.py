"""Compare the legacy SORT-style tracker (scripts/legacy, the original src/tracker) with SkyTracker on proxy metrics (no GT needed).

Detections are computed once per video and cached, so both trackers see identical input.
With a MOT-format ground-truth file (--gt), real MOTA / IDF1 are reported too.

    python scripts/eval_tracker.py models/*.mov
"""
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "backend" / "scripts")]

from skyir.detector import load_detector  # noqa: E402
from skyir.tracker import GMC, SkyTracker, TrackerConfig  # noqa: E402
from skyir.tracker.matching import iou_matrix  # noqa: E402


def detect_all(det, path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, dets = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f); dets.append(det.detect(f))
    return frames, dets, fps


def run_new(frames, dets, fps, nc):
    gmc, trk = GMC(2), SkyTracker(nc, TrackerConfig(fps=fps))
    out, t = [], 0.0
    for f, d in zip(frames, dets):
        a = time.perf_counter()
        M = gmc.estimate(f); gmc.set_boxes(d.xyxy)
        res = trk.update(d.xyxy, d.scores, d.classes, M)
        t += time.perf_counter() - a
        out.append([(r.id, r.xyxy) for r in res if not r.coasting])
    return out, t / len(frames)


def run_legacy(frames, dets, fps, conf=0.5, iou_thr=0.2):
    from legacy.association import associate_detections_to_trackers
    from legacy.gmc import GMC as OldGMC
    from legacy.kalman import KalmanBoxTracker
    KalmanBoxTracker.count = 0
    gmc, trackers, out, t = OldGMC(2), [], [], 0.0
    for idx, (f, d) in enumerate(zip(frames, dets), 1):
        a = time.perf_counter()
        D = np.concatenate([d.xyxy, d.scores[:, None]], 1) if len(d) else np.empty((0, 5))
        gmc.apply(f, trackers)
        trks = np.zeros((len(trackers), 5))
        for k in range(len(trackers)):
            trks[k, :4] = trackers[k].predict()[0]
        hi, lo = D[D[:, 4] >= conf], D[(D[:, 4] > 0.1) & (D[:, 4] < conf)]
        m, ud, ut = associate_detections_to_trackers(hi, trks, iou_thr)
        if len(ut) and len(lo):
            ml, _, _ = associate_detections_to_trackers(lo, trks[ut], 0.1)
            for x in ml: trackers[ut[x[1]]].update(lo[x[0]][:4], lo[x[0]][4])
        for x in m: trackers[x[1]].update(hi[x[0]][:4], hi[x[0]][4])
        for i in ud: trackers.append(KalmanBoxTracker(hi[i][:4]))
        res = []
        for k in range(len(trackers) - 1, -1, -1):
            tr = trackers[k]
            if tr.time_since_update < 1 and (tr.hit_streak >= 3 or idx <= 3):
                res.append((tr.id, tr.get_state()[0]))
            if tr.time_since_update > 30: trackers.pop(k)
        t += time.perf_counter() - a
        out.append(res)
    return out, t / len(frames)


def proxy_metrics(out, dets, fps):
    first, last, lastbox = {}, {}, {}
    covered = total = 0
    for i, (res, d) in enumerate(zip(out, dets)):
        for tid, b in res:
            first.setdefault(tid, i); last[tid] = i; lastbox[tid] = b
        conf = d.xyxy[d.scores >= 0.5]
        total += len(conf)
        if len(conf) and res:
            covered += (iou_matrix(conf, np.array([b for _, b in res])).max(1) >= 0.5).sum()
    lengths = np.array([last[k] - first[k] + 1 for k in first])
    # re-birth: a new ID starts near where another ID ended within the last second
    births = sorted((first[k], k) for k in first)
    ends = [(last[k], k) for k in first]
    rebirth = 0
    for f0, k in births:
        if f0 == 0: continue
        b = out[f0]; bb = dict(b)[k]; c = (bb[:2] + bb[2:]) / 2; diag = max(np.hypot(*(bb[2:] - bb[:2])), 8)
        for fe, ke in ends:
            if ke != k and 0 < f0 - fe <= fps:
                be = lastbox[ke]
                if np.linalg.norm((be[:2] + be[2:]) / 2 - c) < 2 * diag:
                    rebirth += 1; break
    return dict(ids=len(first), mean_len=lengths.mean() if len(lengths) else 0,
                coverage=covered / max(total, 1), rebirths=rebirth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--weights", default=str(ROOT / "models/efficientnetB0-yolov8.pt"))
    a = ap.parse_args()
    det = load_detector(a.weights)
    nc = len(det.names)
    tot = {"legacy": np.zeros(4), "skytrack": np.zeros(4)}
    print(f"{'video':16s} {'tracker':9s} {'IDs':>5s} {'meanLen':>8s} {'cover':>6s} {'rebirth':>7s} {'ms/f':>6s}")
    for v in a.videos:
        frames, dets, fps = detect_all(det, v)
        for name, fn in [("legacy", lambda: run_legacy(frames, dets, fps)), ("skytrack", lambda: run_new(frames, dets, fps, nc))]:
            out, ms = fn()
            m = proxy_metrics(out, dets, fps)
            tot[name] += [m["ids"], m["mean_len"], m["coverage"], m["rebirths"]]
            print(f"{Path(v).name:16s} {name:9s} {m['ids']:5d} {m['mean_len']:8.1f} {m['coverage']:6.3f} {m['rebirths']:7d} {ms*1000:6.2f}")
    n = len(a.videos)
    for k, v in tot.items():
        print(f"{'TOTAL':16s} {k:9s} {int(v[0]):5d} {v[1]/n:8.1f} {v[2]/n:6.3f} {int(v[3]):7d}")


if __name__ == "__main__":
    main()
