import { useEffect, useRef, useState } from "react";
import type { FrameHeader, Track } from "../api";
import { classColor, LOCK_COLOR } from "../theme";
import { Section } from "./ui";

interface Props {
  header: FrameHeader | null;
  hoverId: number | null;
  onHover: (id: number | null) => void;
  onToggleLock: (id: number) => void;
}

const HEIGHT = 230;
const ARROW_S = 3; // velocity arrows show where the target will be in this many seconds
const STEPS = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];

interface Frame {
  ox: number;
  oy: number;
  k: number; // css px per metre
}

/**
 * Bird's-eye view of the ground in front of the UAV: camera at the bottom centre looking up,
 * range rings, the camera footprint, and each track with a velocity arrow.
 */
export function GroundMap({ header, hoverId, onHover, onToggleLock }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<Frame>({ ox: 0, oy: 0, k: 1 });
  const [width, setWidth] = useState(300);

  useEffect(() => {
    const ro = new ResizeObserver(() => setWidth(wrapRef.current?.clientWidth ?? 300));
    ro.observe(wrapRef.current!);
    return () => ro.disconnect();
  }, []);

  const placed = (header?.tracks ?? []).filter((t): t is Track & { ground_xy: [number, number] } => t.ground_xy != null);

  useEffect(() => {
    const c = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(HEIGHT * dpr);
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#06080b";
    ctx.fillRect(0, 0, width, HEIGHT);
    if (!header) return;

    // fit the footprint and every target
    const pts: [number, number][] = [...(header.footprint ?? []), ...placed.map((t) => t.ground_xy)];
    let maxR = 10, maxF = 10;
    for (const [r, f] of pts) {
      maxR = Math.max(maxR, Math.abs(r));
      maxF = Math.max(maxF, f);
    }
    const pad = 14;
    const k = Math.min((width / 2 - pad) / maxR, (HEIGHT - 2 * pad) / maxF);
    const ox = width / 2, oy = HEIGHT - pad;
    frameRef.current = { ox, oy, k };
    const P = (r: number, f: number): [number, number] => [ox + r * k, oy - f * k];

    // footprint
    const fp = header.footprint ?? [];
    if (fp.length > 2) {
      ctx.beginPath();
      fp.forEach(([r, f], i) => (i ? ctx.lineTo(...P(r, f)) : ctx.moveTo(...P(r, f))));
      ctx.closePath();
      ctx.fillStyle = "rgba(255,122,26,0.06)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,122,26,0.35)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // range rings
    const far = Math.hypot(maxR, maxF);
    const step = STEPS.find((s) => far / s <= 5) ?? 2000;
    ctx.font = `10px "JetBrains Mono", ui-monospace, monospace`;
    ctx.textBaseline = "bottom";
    for (let d = step; d <= far * 1.05; d += step) {
      ctx.beginPath();
      ctx.arc(ox, oy, d * k, Math.PI, 2 * Math.PI);
      ctx.strokeStyle = "rgba(169,180,192,0.14)";
      ctx.stroke();
      const [lx, ly] = P(0, d);
      if (ly > 10) {
        ctx.fillStyle = "rgba(169,180,192,0.55)";
        ctx.fillText(`${d} m`, lx + 3, ly - 1);
      }
    }
    // camera marker
    ctx.fillStyle = "#ff7a1a";
    ctx.beginPath();
    ctx.moveTo(ox, oy - 7);
    ctx.lineTo(ox - 5, oy + 3);
    ctx.lineTo(ox + 5, oy + 3);
    ctx.closePath();
    ctx.fill();

    // tracks
    ctx.textBaseline = "middle";
    for (const t of placed) {
      const col = t.locked ? LOCK_COLOR : classColor(t.cls);
      const [x, y] = P(...t.ground_xy);
      const hovered = t.id === hoverId;
      if (t.vel_ms && t.heading_deg != null) {
        const [ex, ey] = P(t.ground_xy[0] + t.vel_ms[0] * ARROW_S, t.ground_xy[1] + t.vel_ms[1] * ARROW_S);
        const len = Math.hypot(ex - x, ey - y);
        if (len > 3) {
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.25;
          ctx.globalAlpha = t.coasting ? 0.5 : 0.9;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(ex, ey);
          const a = Math.atan2(ey - y, ex - x);
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - 5 * Math.cos(a - 0.45), ey - 5 * Math.sin(a - 0.45));
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - 5 * Math.cos(a + 0.45), ey - 5 * Math.sin(a + 0.45));
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
      ctx.fillStyle = col;
      ctx.globalAlpha = t.coasting ? 0.5 : 1;
      ctx.beginPath();
      ctx.arc(x, y, t.locked || hovered ? 4.5 : 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (t.locked || hovered) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.stroke();
      }
      if (t.locked || hovered || placed.length <= 12) {
        ctx.fillStyle = "rgba(230,237,243,0.8)";
        ctx.fillText(String(t.id), x + 6, y - 6);
      }
    }
  }, [header, hoverId, width]);

  const pick = (e: React.MouseEvent): Track | null => {
    const r = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const { ox, oy, k } = frameRef.current;
    let best: Track | null = null, bestD = 10;
    for (const t of placed) {
      const d = Math.hypot(ox + t.ground_xy[0] * k - mx, oy - t.ground_xy[1] * k - my);
      if (d < bestD) (best = t), (bestD = d);
    }
    return best;
  };

  const moving = placed.filter((t) => t.heading_deg != null).length;
  return (
    <Section title="Ground map" right={<span className="muted">{placed.length} on ground · {moving} moving</span>}>
      <div ref={wrapRef} className="ground-map">
        <canvas ref={canvasRef} style={{ width: "100%", height: HEIGHT }}
          onMouseMove={(e) => { const t = pick(e); if ((t?.id ?? null) !== hoverId) onHover(t?.id ?? null); }}
          onMouseLeave={() => onHover(null)}
          onClick={(e) => { const t = pick(e); if (t) onToggleLock(t.id); }} />
      </div>
      <p className="muted small">Camera at the bottom, looking up. The shaded area is what the camera sees. Arrows show where each target will be in {ARROW_S} s.</p>
    </Section>
  );
}
