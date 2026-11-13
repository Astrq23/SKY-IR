import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type ModelInfo, type Params, type SystemInfo, type VideoInfo } from "./api";
import { ExportPanel } from "./components/ExportPanel";
import { PerfPanel } from "./components/PerfPanel";
import { SourcePanel } from "./components/SourcePanel";
import { TrackList } from "./components/TrackList";
import { NumberField, Section, Segmented, Slider, Toggle } from "./components/ui";
import { Viewer, type Overlay } from "./components/Viewer";
import { PALETTE_LABELS } from "./theme";
import { useStream } from "./useStream";

const STORE_KEY = "skyir.settings.v1";

interface Saved {
  params?: Partial<Params>;
  overlay?: Partial<Overlay>;
  palette?: string;
  speed?: number;
  model?: string;
}

function loadSaved(): Saved {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

const DEFAULT_OVERLAY: Overlay = { style: "corners", labels: true, distance: true, trails: true, lockedOnly: false, inset: true };
const SPEEDS = [
  { value: 0.5, label: "½×" },
  { value: 1, label: "1×" },
  { value: 2, label: "2×" },
  { value: 4, label: "4×" },
  { value: 0, label: "Max" },
];

export default function App() {
  const saved = useMemo(loadSaved, []);
  const stream = useStream();
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [video, setVideo] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(saved.model ?? null);
  const [params, setParamsState] = useState<Params | null>(null);
  const [overlay, setOverlay] = useState<Overlay>({ ...DEFAULT_OVERLAY, ...saved.overlay });
  const [palette, setPaletteState] = useState(saved.palette ?? "white-hot");
  const [speed, setSpeedState] = useState(saved.speed ?? 1);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [bootError, setBootError] = useState("");
  const [, forceTick] = useState(0);

  useEffect(() => {
    Promise.all([api.system(), api.models(), api.videos(), api.defaults()])
      .then(([s, m, v, d]) => {
        setSystem(s);
        setModels(m);
        setVideos(v);
        setParamsState({ ...d, ...saved.params });
        setModel((cur) => (cur && m.some((x) => x.id === cur) ? cur : m[0]?.id ?? null));
      })
      .catch((e) => setBootError(`Cannot reach the backend: ${e.message}`));
  }, [saved.params]);

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ params: params ?? undefined, overlay, palette, speed, model: model ?? undefined } satisfies Saved));
    } catch {
      /* storage unavailable */
    }
  }, [params, overlay, palette, speed, model]);

  // live parameter updates, debounced
  const pending = useRef<Partial<Params>>({});
  const timer = useRef<number>();
  const setParam = useCallback(<K extends keyof Params>(k: K, v: Params[K]) => {
    setParamsState((p) => (p ? { ...p, [k]: v } : p));
    pending.current[k] = v;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      stream.setParams(pending.current);
      pending.current = {};
    }, 120);
  }, [stream]);

  const openVideo = useCallback((id: string, mdl = model) => {
    if (!params) return;
    setVideo(id);
    // HIT-UAV style names encode flight data: "<altitude>m-<pitch>_<n>" e.g. 60m-40_3.mov
    const v = videos.find((x) => x.id === id);
    const m = v?.name.match(/(\d+(?:\.\d+)?)m[-_](\d+(?:\.\d+)?)/);
    let p = params;
    if (m && mdl === model) {
      p = { ...params, altitude_m: parseFloat(m[1]), pitch_deg: Math.min(90, parseFloat(m[2])) };
      setParamsState(p);
    }
    stream.open({ video: id, model: mdl, params: p, palette });
  }, [model, params, palette, stream, videos]);

  // speed must be sent after the session is ready
  useEffect(() => {
    if (stream.meta) stream.setSpeed(speed);
  }, [stream.meta]);

  const setPalette = (p: string) => {
    setPaletteState(p);
    stream.setPalette(p);
  };
  const setSpeed = (s: number) => {
    setSpeedState(s);
    stream.setSpeed(s);
  };

  const playing = stream.status === "playing";
  const hasSession = !!stream.meta;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, select, textarea")) return;
      if (!hasSession) return;
      if (e.code === "Space") {
        e.preventDefault();
        playing ? stream.pause() : stream.play();
      } else if (e.code === "ArrowRight" && !playing) stream.step();
      else if (e.key === "r") stream.restart();
      else if (e.key === "Escape") stream.clearLocks();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hasSession, playing, stream]);

  // re-render perf history every 500 ms
  useEffect(() => {
    const id = window.setInterval(() => forceTick((x) => x + 1), 500);
    return () => window.clearInterval(id);
  }, []);

  const h = stream.header;
  const meta = stream.meta;
  const progress = meta && h ? (h.index + 1) / Math.max(meta.frames, 1) : 0;
  const names = meta?.names ?? {};
  const currentVideo = videos.find((v) => v.id === video);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="22" height="22" aria-hidden>
            <path d="M6 6h6M6 6v6M26 6h-6M26 6v6M6 26h6M6 26v-6M26 26h-6M26 26v-6" stroke="currentColor" strokeWidth="2.5" fill="none" />
            <circle cx="16" cy="16" r="3" fill="currentColor" />
          </svg>
          <span>SKY-IR</span>
          <small>UAV thermal tracking</small>
        </div>
        <div className="topbar-info">
          {currentVideo && <span className="chip">{currentVideo.name}</span>}
          <span className={`status-dot ${stream.status}`} />
          <span className="muted">{{ idle: "Ready", connecting: "Connecting", loading: stream.message || "Loading", playing: "Running", paused: "Paused", ended: "Finished", error: "Error" }[stream.status]}</span>
          {system && <span className="chip mono" title={`torch ${system.torch} · TensorRT ${system.tensorrt ?? "—"}`}>{system.gpu ?? "CPU"}</span>}
        </div>
      </header>

      <aside className="panel left">
        <SourcePanel videos={videos} selected={video} palette={palette}
          onSelect={(id) => openVideo(id)}
          onUploaded={(v) => { setVideos((vs) => [...vs.filter((x) => x.id !== v.id), v]); openVideo(v.id); }}
          onDeleted={(id) => setVideos((vs) => vs.filter((x) => x.id !== id))} />

        <Section title="Model">
          <select className="select" value={model ?? ""} onChange={(e) => { setModel(e.target.value); if (video) openVideo(video, e.target.value); }}>
            {models.map((m) => <option key={m.id} value={m.id}>{m.name}{m.engine ? "  · TRT" : ""}</option>)}
          </select>
          {meta && <p className="muted small">Input {meta.input[1]}×{meta.input[0]} · {Object.keys(names).length} classes · {meta.backend}</p>}
        </Section>

        {params && (
          <>
            <Section title="Detection & Tracking">
              <Slider label="High confidence threshold" value={params.track_high} min={0.2} max={0.9} step={0.05}
                hint="Detections above this are matched first and can start new tracks" onChange={(v) => setParam("track_high", v)} />
              <Slider label="New track threshold" value={params.new_track} min={0.2} max={0.95} step={0.05}
                hint="Raise to suppress false IDs, lower to pick up faint targets sooner" onChange={(v) => setParam("new_track", v)} />
              <Slider label="Low threshold (floor)" value={params.conf} min={0.05} max={0.5} step={0.01}
                hint="Detections below are discarded; those between floor and high only extend existing tracks (ByteTrack)" onChange={(v) => setParam("conf", v)} />
              <Slider label="NMS IoU" value={params.nms_iou} min={0.2} max={0.9} step={0.05} onChange={(v) => setParam("nms_iou", v)} />
              <Slider label="Match threshold (cost)" value={params.match_thresh} min={0.5} max={0.95} step={0.01}
                hint="Higher = easier matching (fewer lost tracks, more ID mix-ups)" onChange={(v) => setParam("match_thresh", v)} />
              <Slider label="Keep lost tracks for" value={params.track_buffer_s} min={0.5} max={15} step={0.5} unit="s" onChange={(v) => setParam("track_buffer_s", v)} />
              <Toggle label="Camera motion compensation (GMC)" checked={params.gmc} onChange={(v) => setParam("gmc", v)} />
              {Object.keys(names).length > 0 && (
                <div className="class-filter">
                  {Object.entries(names).map(([k, n]) => {
                    const id = Number(k);
                    const on = params.classes == null || params.classes.includes(id);
                    return (
                      <button key={k} className={`chip-btn ${on ? "on" : ""}`} onClick={() => {
                        const all = Object.keys(names).map(Number);
                        const cur = params.classes ?? all;
                        const next = on ? cur.filter((c) => c !== id) : [...cur, id];
                        setParam("classes", next.length === all.length ? null : next);
                      }}>{n}</button>
                    );
                  })}
                </div>
              )}
            </Section>

            <Section title="Camera / UAV">
              <div className="grid3">
                <NumberField label="Altitude" unit="m" value={params.altitude_m} min={0} onChange={(v) => setParam("altitude_m", v)} />
                <NumberField label="Pitch" unit="°" value={params.pitch_deg} min={0} max={90} onChange={(v) => setParam("pitch_deg", v)} />
                <NumberField label="Focal length" unit="px" value={params.focal_px} min={1} step={10} onChange={(v) => setParam("focal_px", v)} />
              </div>
              <p className="muted small">Auto-filled from file names like <span className="mono">60m-40_*.mov</span> (altitude 60 m, pitch 40°). Focal length in pixels: f<sub>px</sub> = f<sub>mm</sub> × width<sub>px</sub> / sensor<sub>mm</sub>.</p>
            </Section>
          </>
        )}

        <Section title="Display">
          <Segmented options={(system?.palettes ?? ["white-hot"]).map((p) => ({ value: p, label: PALETTE_LABELS[p] ?? p }))} value={palette} onChange={setPalette} />
          <Segmented options={[{ value: "corners", label: "Corners" }, { value: "box", label: "Box" }]} value={overlay.style}
            onChange={(v) => setOverlay((o) => ({ ...o, style: v as Overlay["style"] }))} />
          <div className="toggles">
            <Toggle label="Labels" checked={overlay.labels} onChange={(v) => setOverlay((o) => ({ ...o, labels: v }))} />
            <Toggle label="Distance" checked={overlay.distance} onChange={(v) => setOverlay((o) => ({ ...o, distance: v }))} />
            <Toggle label="Motion trails" checked={overlay.trails} onChange={(v) => setOverlay((o) => ({ ...o, trails: v }))} />
            <Toggle label="Target zoom inset" checked={overlay.inset} onChange={(v) => setOverlay((o) => ({ ...o, inset: v }))} />
            <Toggle label="Show locked targets only" checked={overlay.lockedOnly} onChange={(v) => setOverlay((o) => ({ ...o, lockedOnly: v }))} />
          </div>
        </Section>
      </aside>

      <main className="stage">
        <div className="viewer-wrap">
          {!hasSession && (
            <div className="placeholder">
              {bootError ? <p className="error-text">{bootError}</p> : stream.status === "loading" || stream.status === "connecting" ? (
                <><div className="spinner" /><p>{stream.message || "Initializing…"}</p><p className="muted small">The first run may take 2–3 minutes to build the TensorRT engine.</p></>
              ) : stream.status === "error" ? <p className="error-text">{stream.message}</p> : (
                <><p>Select a video in the left panel to start.</p><p className="muted small">Click an object to lock it · Space play/pause · → next frame · R restart · Esc unlock all</p></>
              )}
            </div>
          )}
          <Viewer stream={stream} overlay={overlay} hoverId={hoverId} onHover={setHoverId} />
        </div>
        <div className="transport">
          <button className="btn icon" disabled={!hasSession} onClick={() => (playing ? stream.pause() : stream.play())} title="Space">
            {playing ? "❚❚" : "▶"}
          </button>
          <button className="btn icon" disabled={!hasSession || playing} onClick={stream.step} title="Next frame (→)">⏭</button>
          <button className="btn icon" disabled={!hasSession} onClick={stream.restart} title="Restart (R)">↺</button>
          <div className="timeline">
            <div className="progress"><span style={{ width: `${progress * 100}%` }} /></div>
            <div className="row between small mono">
              <span>{h ? `${h.time.toFixed(2)}s · F${h.index + 1}` : "—"}</span>
              <span>{meta ? `${meta.frames} frames · ${meta.fps.toFixed(1)} fps` : ""}</span>
            </div>
          </div>
          <Segmented options={SPEEDS} value={speed} onChange={setSpeed} />
        </div>
      </main>

      <aside className="panel right">
        <TrackList header={h} hoverId={hoverId} onHover={setHoverId} onToggleLock={stream.toggleLock} onClearLocks={stream.clearLocks} />
        <PerfPanel header={h} history={stream.historyRef.current} meta={meta} />
        {params && <ExportPanel video={video} model={model} params={params} palette={palette} style={overlay.style} locked={h?.locked ?? []} />}
      </aside>
    </div>
  );
}
