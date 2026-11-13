"""FastAPI backend.

REST:  /api/system, /api/models, /api/videos (+upload, thumbnails), /api/exports (batch render jobs)
WS:    /api/stream – live tracking. Client -> server: JSON control messages. Server -> client:
       text JSON for events, binary frames = <uint32 LE header length><JSON header><JPEG bytes>.

Run:   python -m skyir   (or: uvicorn skyir.server:app)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .detector import load_detector
from .export import export_video
from .pipeline import Params, Session
from .render import PALETTES, apply_palette

log = logging.getLogger("skyir")

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
SAMPLE_DIRS = [MODELS_DIR, DATA_DIR / "samples"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="SKY-IR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- models / detectors
_detectors: dict[tuple[str, str], object] = {}
_det_lock = threading.Lock()


def list_models() -> list[dict]:
    out = []
    for p in sorted(MODELS_DIR.glob("*.pt")):
        if p.stat().st_size == 0:
            continue
        out.append({"id": p.stem, "name": p.stem, "path": str(p), "engine": p.with_suffix(".engine").exists()})
    return out


def get_detector(model_id: str | None, backend: str = "auto"):
    models = {m["id"]: m for m in list_models()}
    if not models:
        raise HTTPException(500, f"no .pt weights in {MODELS_DIR}")
    m = models.get(model_id or "", next(iter(models.values())))
    key = (m["id"], backend)
    with _det_lock:  # first load may build a TensorRT engine; never do it twice
        if key not in _detectors:
            _detectors[key] = load_detector(m["path"], backend)
    return _detectors[key]


# ---------------------------------------------------------------- videos
def _vid(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def _probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 3), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def _all_videos() -> dict[str, dict]:
    vids = {}
    for kind, dirs in (("sample", SAMPLE_DIRS), ("upload", [UPLOAD_DIR])):
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in VIDEO_EXT:
                    name = p.name.split("__", 1)[-1] if kind == "upload" else p.name
                    vids[_vid(p)] = {"id": _vid(p), "name": name, "kind": kind, "path": str(p)}
    return vids


def _video_path(video_id: str) -> str:
    v = _all_videos().get(video_id)
    if not v:
        raise HTTPException(404, "unknown video")
    return v["path"]


@app.get("/api/system")
def system():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        import tensorrt  # noqa: F401

        trt = tensorrt.__version__
    except Exception:  # noqa: BLE001
        trt = None
    return {"gpu": gpu, "torch": torch.__version__, "tensorrt": trt, "palettes": list(PALETTES)}


@app.get("/api/models")
def models():
    return list_models()


@app.get("/api/videos")
def videos():
    out = []
    for v in _all_videos().values():
        out.append({k: v[k] for k in ("id", "name", "kind")} | _probe(Path(v["path"])))
    return out


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported video type {suffix}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}__{Path(file.filename).name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    meta = _probe(dest)
    if meta["frames"] <= 0 and meta["width"] == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "could not decode video")
    return {"id": _vid(dest), "name": Path(file.filename).name, "kind": "upload"} | meta


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    v = _all_videos().get(video_id)
    if not v or v["kind"] != "upload":
        raise HTTPException(404, "not an uploaded video")
    Path(v["path"]).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/videos/{video_id}/thumb.jpg")
def thumbnail(video_id: str, palette: str = "white-hot"):
    cap = cv2.VideoCapture(_video_path(video_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "cannot read frame")
    frame = apply_palette(frame, palette)
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (320, int(320 * h / w)), interpolation=cv2.INTER_AREA)
    return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------- live stream
@app.websocket("/api/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session")  # one thread owns the Session
    session: Session | None = None
    state = {"playing": False, "step": False, "speed": 1.0, "palette": "white-hot", "quality": 82}
    wake = asyncio.Event()
    clock = {"t0": 0.0, "i0": 0}
    ema = {"fps": 0.0, "last": 0.0}

    def next_packet():
        r = session.step()
        if r is None:
            return None
        t = time.perf_counter()
        img = apply_palette(r.frame, state["palette"])
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, state["quality"]])[1].tobytes()
        hdr = r.header()
        hdr["timings"]["encode"] = round((time.perf_counter() - t) * 1000, 2)
        now = time.perf_counter()
        if ema["last"]:
            inst = 1.0 / max(now - ema["last"], 1e-6)
            ema["fps"] = inst if not ema["fps"] else 0.9 * ema["fps"] + 0.1 * inst
        ema["last"] = now
        hdr["fps"] = round(ema["fps"], 1)
        hdr["locked"] = sorted(session.tracker.locked_ids)
        h = json.dumps(hdr, separators=(",", ":")).encode()
        return struct.pack("<I", len(h)) + h + jpg, r.index

    def reset_clock(index: int):
        clock["t0"], clock["i0"] = time.perf_counter(), index
        ema["last"] = 0.0

    async def producer():
        while True:
            if session is None or not (state["playing"] or state["step"]):
                wake.clear()
                await wake.wait()
                continue
            stepping = state["step"]
            state["step"] = False
            res = await loop.run_in_executor(ex, next_packet)
            if res is None:
                state["playing"] = False
                await ws.send_text(json.dumps({"type": "end"}))
                continue
            pkt, idx = res
            await ws.send_bytes(pkt)
            if state["playing"] and not stepping and state["speed"] > 0:
                target = clock["t0"] + (idx - clock["i0"] + 1) / (session.fps * state["speed"])
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -0.5:  # fell behind (slow client/GPU): resync instead of bursting
                    reset_clock(idx + 1)

    prod = asyncio.create_task(producer())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "open":
                if session is not None:
                    await loop.run_in_executor(ex, session.close)
                await ws.send_text(json.dumps({"type": "loading", "message": "Loading model…"}))
                try:
                    path = _video_path(msg["video"])
                    det = await loop.run_in_executor(None, get_detector, msg.get("model"), msg.get("backend", "auto"))
                    params = Params()
                    params.update(msg.get("params", {}))
                    session = await loop.run_in_executor(ex, Session, path, det, params)
                except Exception as e:  # noqa: BLE001
                    session = None
                    await ws.send_text(json.dumps({"type": "error", "message": str(getattr(e, "detail", e))}))
                    continue
                state.update(playing=bool(msg.get("autoplay", True)), step=False)
                state["palette"] = msg.get("palette", state["palette"])
                reset_clock(0)
                await ws.send_text(json.dumps({"type": "ready", "meta": {
                    "width": session.width, "height": session.height, "fps": session.fps, "frames": session.total,
                    "names": session.names, "backend": det.backend, "input": list(det.input_size),
                }}))
                if not state["playing"]:
                    state["step"] = True  # show the first frame
                wake.set()
            elif session is None:
                continue
            elif kind == "play":
                state["playing"] = True
                reset_clock(session.index + 1)
                wake.set()
            elif kind == "pause":
                state["playing"] = False
            elif kind == "step":
                state["step"] = True
                wake.set()
            elif kind == "restart":
                await loop.run_in_executor(ex, session.restart)
                reset_clock(0)
                state["step"] = not state["playing"]
                wake.set()
            elif kind == "speed":
                state["speed"] = float(msg.get("value", 1.0))
                reset_clock(session.index + 1)
            elif kind == "palette":
                state["palette"] = msg.get("value", "white-hot")
            elif kind == "params":
                await loop.run_in_executor(ex, session.set_params, msg.get("value", {}))
            elif kind == "lock":
                on = await loop.run_in_executor(ex, session.toggle_lock, int(msg["id"]))
                await ws.send_text(json.dumps({"type": "locked", "id": int(msg["id"]), "on": on}))
            elif kind == "clear_locks":
                await loop.run_in_executor(ex, session.clear_locks)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        prod.cancel()
        if session is not None:
            ex.submit(session.close)
        ex.shutdown(wait=False)


# ---------------------------------------------------------------- export jobs
_jobs: dict[str, dict] = {}


@app.post("/api/exports")
async def create_export(config: str = Form(...), gt: UploadFile | None = File(None)):
    cfg = json.loads(config)
    path = _video_path(cfg["video"])
    job_id = uuid.uuid4().hex[:10]
    out_dir = EXPORT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = None
    if gt is not None and gt.filename:
        gt_path = out_dir / "gt.txt"
        gt_path.write_bytes(await gt.read())
    job = {"id": job_id, "status": "queued", "progress": 0.0, "frames": 0, "total": 0, "result": None, "error": None, "cancel": False}
    _jobs[job_id] = job

    def run():
        try:
            job["status"] = "loading"
            det = get_detector(cfg.get("model"), cfg.get("backend", "auto"))
            params = Params()
            params.update(cfg.get("params", {}))
            job["status"] = "running"

            def progress(n, total):
                job.update(frames=n, total=total, progress=n / total if total else 0.0)

            r = export_video(path, det, out_dir, params, cfg.get("palette", "white-hot"), cfg.get("style", "corners"),
                             cfg.get("locked", []), str(gt_path) if gt_path else None, progress, lambda: job["cancel"])
            job["result"] = {"frames": r.frames, "seconds": round(r.seconds, 2), "metrics": r.metrics, **r.summary}
            job.update(status="cancelled" if job["cancel"] else "done", progress=1.0)
        except Exception as e:  # noqa: BLE001
            log.exception("export failed")
            job.update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return _public(job)


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "cancel"}


@app.get("/api/exports/{job_id}")
def export_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _public(_jobs[job_id])


@app.delete("/api/exports/{job_id}")
def cancel_export(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    _jobs[job_id]["cancel"] = True
    return {"ok": True}


@app.get("/api/exports/{job_id}/{name}")
def export_file(job_id: str, name: str):
    files = {"video": ("result.mp4", "video/mp4"), "tracks": ("tracks.txt", "text/plain")}
    if name not in files:
        raise HTTPException(404)
    p = EXPORT_DIR / job_id / files[name][0]
    if not p.exists():
        raise HTTPException(404, "not ready")
    return FileResponse(p, media_type=files[name][1], filename=f"skyir_{job_id}_{files[name][0]}")


@app.get("/api/defaults")
def defaults():
    return asdict(Params())


# ---------------------------------------------------------------- frontend (mounted last: catches "/")
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
