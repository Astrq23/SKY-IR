import { useEffect, useRef, useState } from "react";
import { api, type ExportJob, type Params } from "../api";
import { Section } from "./ui";

interface Props {
  video: string | null;
  model: string | null;
  params: Params;
  palette: string;
  style: string;
  locked: number[];
}

const METRIC_LABELS: [string, string, (v: number) => string][] = [
  ["mota", "MOTA", (v) => `${(v * 100).toFixed(1)}%`],
  ["idf1", "IDF1", (v) => `${(v * 100).toFixed(1)}%`],
  ["motp", "MOTP", (v) => v.toFixed(3)],
  ["num_switches", "ID Sw", (v) => v.toFixed(0)],
  ["mostly_tracked", "MT", (v) => v.toFixed(0)],
  ["mostly_lost", "ML", (v) => v.toFixed(0)],
  ["num_fragmentations", "Frag", (v) => v.toFixed(0)],
  ["num_frames", "Frames", (v) => v.toFixed(0)],
];

export function ExportPanel({ video, model, params, palette, style, locked }: Props) {
  const [job, setJob] = useState<ExportJob | null>(null);
  const [gt, setGt] = useState<File | null>(null);
  const [err, setErr] = useState("");
  const gtInput = useRef<HTMLInputElement>(null);
  const running = job && ["queued", "loading", "running"].includes(job.status);

  useEffect(() => {
    if (!job || !running) return;
    const id = window.setInterval(async () => {
      try {
        setJob(await api.exportStatus(job.id));
      } catch (e) {
        setErr((e as Error).message);
      }
    }, 400);
    return () => window.clearInterval(id);
  }, [job?.id, running]);

  const start = async () => {
    if (!video) return;
    setErr("");
    try {
      setJob(await api.startExport({ video, model, params, palette, style, locked }, gt));
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const m = job?.result?.metrics;
  return (
    <Section title="Export">
      <p className="muted small">Processes the whole video at full speed and exports an annotated MP4 (H.264) plus a MOT-format tracks file.</p>
      <div className="row gap-s">
        <button className="btn ghost" onClick={() => gtInput.current?.click()} title="MOT-format ground truth (frame,id,x,y,w,h,…)">
          {gt ? `GT: ${gt.name}` : "+ Ground truth"}
        </button>
        {gt && <button className="link-btn" onClick={() => setGt(null)}>remove</button>}
        <input ref={gtInput} type="file" accept=".txt,.csv" hidden onChange={(e) => setGt(e.target.files?.[0] ?? null)} />
        <span className="spacer" />
        {running ? (
          <button className="btn" onClick={() => api.cancelExport(job!.id)}>Cancel</button>
        ) : (
          <button className="btn primary" disabled={!video} onClick={start}>Export video</button>
        )}
      </div>
      {err && <p className="error-text">{err}</p>}
      {job && (
        <div className="job">
          <div className="progress"><span style={{ width: `${(job.progress * 100).toFixed(1)}%` }} /></div>
          <div className="row between small">
            <span>{{ queued: "Queued", loading: "Loading model", running: `Processing ${job.frames}/${job.total}`, done: "Done", error: "Error", cancelled: "Cancelled" }[job.status]}</span>
            {job.result && <span className="mono">{job.result.frames} frame · {job.result.seconds}s · {job.result.fps.toFixed(0)} fps · {job.result.unique_ids} ID</span>}
          </div>
          {job.error && <p className="error-text">{job.error}</p>}
          {job.status === "done" && (
            <div className="row gap-s">
              <a className="btn" href={api.exportFile(job.id, "video")} download>↓ MP4</a>
              <a className="btn" href={api.exportFile(job.id, "tracks")} download>↓ tracks.txt</a>
            </div>
          )}
          {m && (
            <div className="metrics">
              {METRIC_LABELS.map(([k, label, fmt]) => (
                <div key={k}><span>{label}</span><strong>{m[k] == null ? "—" : fmt(m[k] as number)}</strong></div>
              ))}
            </div>
          )}
        </div>
      )}
    </Section>
  );
}
