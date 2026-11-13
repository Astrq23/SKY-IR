import { useRef, useState } from "react";
import { api, type VideoInfo } from "../api";
import { Section } from "./ui";

interface Props {
  videos: VideoInfo[];
  selected: string | null;
  palette: string;
  onSelect: (id: string) => void;
  onUploaded: (v: VideoInfo) => void;
  onDeleted: (id: string) => void;
}

const fmtDur = (v: VideoInfo) => {
  const s = v.fps ? v.frames / v.fps : 0;
  return `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
};

export function SourcePanel({ videos, selected, palette, onSelect, onUploaded, onDeleted }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const upload = async (file?: File) => {
    if (!file) return;
    setBusy(true);
    setErr("");
    try {
      onUploaded(await api.upload(file));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Video source" right={<span className="muted">{videos.length}</span>}>
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}
      >
        <input ref={input} type="file" accept="video/*,.mov,.mkv" hidden onChange={(e) => upload(e.target.files?.[0])} />
        {busy ? "Uploading…" : <>Drop a video here or <u>browse</u></>}
      </div>
      {err && <p className="error-text">{err}</p>}
      <ul className="video-list">
        {videos.map((v) => (
          <li key={v.id} className={v.id === selected ? "active" : ""} onClick={() => onSelect(v.id)}>
            <img src={api.thumb(v.id, palette)} alt="" loading="lazy" />
            <div className="video-meta">
              <strong title={v.name}>{v.name}</strong>
              <span>{v.width}×{v.height} · {v.fps.toFixed(1)} fps · {fmtDur(v)}</span>
            </div>
            {v.kind === "upload" && (
              <button className="icon-btn" title="Delete uploaded video"
                onClick={async (e) => { e.stopPropagation(); await api.deleteVideo(v.id); onDeleted(v.id); }}>×</button>
            )}
          </li>
        ))}
      </ul>
    </Section>
  );
}
