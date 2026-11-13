import type { FrameHeader, StreamMeta } from "../api";
import { Section } from "./ui";

interface Props {
  header: FrameHeader | null;
  history: FrameHeader[];
  meta: StreamMeta | null;
}

const STAGES: { key: keyof FrameHeader["timings"]; label: string; color: string }[] = [
  { key: "detect", label: "Detect", color: "#ff7a1a" },
  { key: "track", label: "Track", color: "#4cc9f0" },
  { key: "gmc_extra", label: "GMC*", color: "#7bf1a8" },
  { key: "encode", label: "Encode", color: "#c084fc" },
  { key: "decode", label: "Decode", color: "#9aa4b2" },
];

function avg(h: FrameHeader[], key: keyof FrameHeader["timings"]) {
  if (!h.length) return 0;
  return h.reduce((s, x) => s + (x.timings[key] ?? 0), 0) / h.length;
}

function Sparkline({ values, max }: { values: number[]; max: number }) {
  const W = 260, H = 44;
  if (values.length < 2) return <svg className="spark" viewBox={`0 0 ${W} ${H}`} />;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${H - (Math.min(v, max) / max) * (H - 4) - 2}`).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline points={`0,${H} ${pts} ${W},${H}`} fill="rgba(255,122,26,0.12)" stroke="none" />
      <polyline points={pts} fill="none" stroke="#ff7a1a" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function PerfPanel({ header, history, meta }: Props) {
  const recent = history.slice(-30);
  const total = avg(recent, "total") + avg(recent, "encode");
  const capacity = total > 0 ? 1000 / total : 0;
  const lat = history.map((h) => h.timings.total + (h.timings.encode ?? 0));
  const maxLat = Math.max(20, ...lat);
  return (
    <Section title="Performance" right={meta && <span className="badge">{meta.backend === "tensorrt" ? "TensorRT FP16" : "PyTorch"}</span>}>
      <div className="kpis">
        <div><span>Display</span><strong>{header?.fps.toFixed(1) ?? "—"}<small> fps</small></strong></div>
        <div><span>Capacity</span><strong>{capacity ? capacity.toFixed(0) : "—"}<small> fps</small></strong></div>
        <div><span>Latency</span><strong>{total ? total.toFixed(1) : "—"}<small> ms</small></strong></div>
      </div>
      <Sparkline values={lat} max={maxLat} />
      <div className="stage-bar">
        {STAGES.map((s) => {
          const v = avg(recent, s.key);
          return total > 0 && v > 0 ? <span key={s.key} style={{ flex: v, background: s.color }} title={`${s.label} ${v.toFixed(2)} ms`} /> : null;
        })}
      </div>
      <ul className="stage-legend">
        {STAGES.map((s) => (
          <li key={s.key}><i style={{ background: s.color }} />{s.label}<b>{avg(recent, s.key).toFixed(2)} ms</b></li>
        ))}
      </ul>
      <p className="footnote">* GMC runs on the CPU in parallel with detection; only time exceeding detection is counted. Camera: Δ {header ? `${header.motion[0].toFixed(1)}, ${header.motion[1].toFixed(1)}` : "—"} px/frame.</p>
    </Section>
  );
}
