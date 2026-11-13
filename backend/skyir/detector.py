"""Object detectors.

Two backends share one interface (`Detector.detect(frame_bgr) -> Detections`):

* `TensorRTDetector` – runs an Ultralytics-exported ``.engine`` directly through the TensorRT
  runtime. Pre-processing (BGR->RGB, letterbox, normalise) and post-processing (decode + NMS)
  stay on the GPU, so only the raw frame is uploaded and only the final boxes are downloaded.
* `TorchDetector` – Ultralytics ``.pt`` fallback (FP16 on CUDA) for machines without TensorRT.

`load_detector()` picks the fastest available backend and builds the engine on first use.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

log = logging.getLogger(__name__)


@dataclass
class Detections:
    """N detections as float32 arrays in original-frame pixel coordinates."""

    xyxy: np.ndarray  # (N, 4)
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) int

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))

    def __len__(self) -> int:
        return len(self.scores)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _GpuPostprocess:
    """Shared YOLOv8 decode + class-aware NMS on (1, 4+nc, A) raw head output."""

    def __init__(self, conf: float = 0.1, iou: float = 0.45, max_det: int = 300):
        self.conf, self.iou, self.max_det = conf, iou, max_det

    def __call__(self, raw: torch.Tensor, scale: float, pad: tuple[float, float], shape: tuple[int, int],
                 conf: float | None = None, iou: float | None = None) -> Detections:
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        p = raw[0].float().T  # (A, 4+nc)
        scores, classes = p[:, 4:].max(1)
        keep = scores > conf
        if not keep.any():
            return Detections.empty()
        p, scores, classes = p[keep], scores[keep], classes[keep]
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1)
        idx = torchvision.ops.batched_nms(boxes, scores, classes, iou)[: self.max_det]
        boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
        # undo letterbox
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, shape[0])
        out = torch.cat((boxes, scores[:, None], classes[:, None].float()), 1).cpu().numpy()
        return Detections(out[:, :4].astype(np.float32), out[:, 4].astype(np.float32), out[:, 5].astype(np.int32))


class TensorRTDetector:
    backend = "tensorrt"

    def __init__(self, engine_path: str | Path, conf: float = 0.1, iou: float = 0.45):
        import tensorrt as trt

        self.dev = _device()
        with open(engine_path, "rb") as f:
            n = int.from_bytes(f.read(4), "little")
            meta = json.loads(f.read(n))  # Ultralytics header: <int32 len><json><engine bytes>
            blob = f.read()
        self.names = {int(k): v for k, v in meta["names"].items()}
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        _, _, self.H, self.W = tuple(self.engine.get_tensor_shape(self.in_name))
        self.inp = torch.zeros((1, 3, self.H, self.W), dtype=torch.float32, device=self.dev)
        self.out = torch.zeros(tuple(self.engine.get_tensor_shape(self.out_name)), dtype=torch.float32, device=self.dev)
        self.ctx.set_tensor_address(self.in_name, self.inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.stream = torch.cuda.Stream(self.dev)
        self.post = _GpuPostprocess(conf, iou)
        self._lock = threading.Lock()  # one execution context -> serialise callers
        self._pinned: torch.Tensor | None = None

    @property
    def input_size(self) -> tuple[int, int]:
        return self.H, self.W

    def _upload(self, frame: np.ndarray) -> torch.Tensor:
        if self._pinned is None or self._pinned.shape != frame.shape:
            self._pinned = torch.empty(frame.shape, dtype=torch.uint8).pin_memory()
        self._pinned.numpy()[...] = frame
        return self._pinned.to(self.dev, non_blocking=True)

    @torch.inference_mode()
    def detect(self, frame: np.ndarray, conf: float | None = None, iou: float | None = None) -> Detections:
        h0, w0 = frame.shape[:2]
        with self._lock, torch.cuda.stream(self.stream):
            img = self._upload(frame).permute(2, 0, 1).flip(0)[None].float().div_(255.0)  # BGR->RGB, CHW
            scale = min(self.H / h0, self.W / w0)
            nh, nw = round(h0 * scale), round(w0 * scale)
            if (nh, nw) != (h0, w0):
                img = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)
            px, py = (self.W - nw) / 2, (self.H - nh) / 2
            self.inp.fill_(114 / 255.0)
            top, left = int(round(py - 0.1)), int(round(px - 0.1))
            self.inp[:, :, top : top + nh, left : left + nw] = img
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            dets = self.post(self.out, scale, (left, top), (h0, w0), conf, iou)
        return dets


class TorchDetector:
    backend = "torch"

    def __init__(self, weights: str | Path, conf: float = 0.1, iou: float = 0.45, imgsz: tuple[int, int] = (512, 640)):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = dict(self.model.names)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.half = torch.cuda.is_available()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        from ultralytics.cfg import DEFAULT_CFG_DICT

        # newer Ultralytics renamed `half=True` to `quantize=16`
        self._fp16 = ({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}) if self.half else {}
        self._lock = threading.Lock()

    @property
    def input_size(self) -> tuple[int, int]:
        return self.imgsz

    def detect(self, frame: np.ndarray, conf: float | None = None, iou: float | None = None) -> Detections:
        with self._lock:
            r = self.model(frame, verbose=False, conf=self.conf if conf is None else conf, iou=self.iou if iou is None else iou, imgsz=self.imgsz, device=self.device, **self._fp16)[0]
        d = r.boxes.data.cpu().numpy()
        if len(d) == 0:
            return Detections.empty()
        return Detections(d[:, :4].astype(np.float32), d[:, 4].astype(np.float32), d[:, 5].astype(np.int32))


def build_engine(weights: Path, imgsz: tuple[int, int]) -> Path:
    """Export ``weights`` to an FP16 TensorRT engine next to it (takes ~2-3 minutes once)."""
    from ultralytics import YOLO

    log.info("Building TensorRT FP16 engine for %s at %s (one-off, ~2-3 min)…", weights.name, imgsz)
    out = YOLO(str(weights)).export(format="engine", half=True, imgsz=imgsz, device=0, workspace=2, simplify=True, verbose=False)
    return Path(out)


def load_detector(weights: str | Path, backend: str = "auto", imgsz: tuple[int, int] = (512, 640)):
    """Return the fastest detector available for ``weights``.

    backend: ``auto`` (TensorRT if possible, building the engine on first run), ``tensorrt`` or ``torch``.
    """
    weights = Path(weights)
    engine = weights.with_suffix(".engine")
    if backend in ("auto", "tensorrt") and torch.cuda.is_available():
        try:
            if not engine.exists() and weights.suffix == ".pt":
                engine = build_engine(weights, imgsz)
            det = TensorRTDetector(engine if weights.suffix != ".engine" else weights)
            log.info("Detector: TensorRT engine %s (input %sx%s)", engine.name, det.W, det.H)
            return det
        except Exception as e:  # noqa: BLE001 - fall back to torch on any TRT problem
            if backend == "tensorrt":
                raise
            log.warning("TensorRT unavailable (%s); falling back to PyTorch", e)
    det = TorchDetector(weights, imgsz=imgsz)
    log.info("Detector: PyTorch %s (fp16=%s)", weights.name, det.half)
    return det
