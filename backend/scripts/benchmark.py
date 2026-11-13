"""Benchmark detector backends and the full pipeline on a video.

    python backend/scripts/benchmark.py models/60m-30_1.mov [--frames 300]

Reports ms/frame for: Ultralytics PyTorch FP32 (the original app), PyTorch FP16, the lean TensorRT
detector, and the complete SKY-IR pipeline (decode + detect || GMC + track).
"""
import argparse, sys, time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from skyir.detector import TensorRTDetector, TorchDetector, load_detector  # noqa: E402
from skyir.pipeline import Session  # noqa: E402


def timeit(fn, frames, warmup=10):
    for f in frames[:warmup]:
        fn(f)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for f in frames:
        fn(f)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / len(frames) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(ROOT / "models/efficientnetB0-yolov8.pt"))
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    cap, frames = cv2.VideoCapture(a.video), []
    while len(frames) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    print(f"{len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]} on {torch.cuda.get_device_name(0)}\n")
    from ultralytics import YOLO

    y = YOLO(a.weights)
    rows = [("ultralytics fp32 (original app)", timeit(lambda f: y(f, verbose=False, conf=0.1, device=0), frames))]
    td = TorchDetector(a.weights)
    rows.append(("pytorch fp16", timeit(td.detect, frames)))
    eng = Path(a.weights).with_suffix(".engine")
    if eng.exists():
        rows.append(("tensorrt fp16 (lean runner)", timeit(TensorRTDetector(eng).detect, frames)))
    for name, ms in rows:
        print(f"  detector  {name:34s} {ms:6.2f} ms  {1000 / ms:6.1f} FPS")
    s = Session(a.video, load_detector(a.weights))
    n, t = 0, time.perf_counter()
    while n < len(frames) and s.step() is not None:
        n += 1
    ms = (time.perf_counter() - t) / n * 1000
    print(f"  pipeline  {'end-to-end (' + s.det.backend + ')':34s} {ms:6.2f} ms  {1000 / ms:6.1f} FPS")
    s.close()


if __name__ == "__main__":
    main()
