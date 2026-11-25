import { useEffect, useRef, useState } from "react";
import type { FrameHeader, Track } from "../api";
import { compass } from "../compass";
import { classColor, LOCK_COLOR } from "../theme";
import { Section } from "./ui";

interface Props {
  /** compass bearing the camera faces (0 = north) */
  yaw: number;
  header: FrameHeader | null;
  hoverId: number | null;
  onHover: (id: number | null) => void;
  onToggleLock: (id: number) => void;
}

const HEIGHT = 230;
const ARROW_S = 3; // velocity arrows show where the target will be in this many seconds
const STEPS = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];

interface Frame {
  /** ground (right, forward) metres -> canvas css px, as drawn last */
  P: (r: number, f: number) => [number, number];
}

const NORTH_UP_KEY = "skyir.map.northUp";

/**
 * Bird's-eye view of the ground around the UAV: range rings, the camera footprint, a compass and
 * each track with a velocity arrow. "Camera up" puts the viewing direction at the top; "north up"
 * rotates the map by the camera's compass heading (yaw) so north is at the top.
 */
export function GroundMap({ yaw, header, hoverId, onHover, onToggleLock }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<Frame>({ P: () => [-1e9, -1e9] });
  const [width, setWidth] = useState(300);
  const [northUp, setNorthUp] = useState(() => localStorage.getItem(NORTH_UP_KEY) === "1");
  useEffect(() => {
    try {
      localStorage.setItem(NORTH_UP_KEY, northUp ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  }, [northUp]);

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

    // map rotation: screen angle (clockwise from up) of the camera's forward axis
    const phi = northUp ? (yaw * Math.PI) / 180 : 0;
    const cs = Math.cos(phi), sn = Math.sin(phi);
    const rot = (r: number, f: number): [number, number] => [r * cs + f * sn, -r * sn + f * cs];

    // fit the camera footprint (plus the camera itself) after rotation, not the targets: the
    // footprint only changes with altitude/pitch/focal/heading, so the scale stays put while targets
    // appear and disappear (targets outside it are clamped to the edge below)
    const area = header.footprint?.length ? header.footprint : placed.map((t) => t.ground_xy);
    const pts: [number, number][] = [[0, 0], ...area];
    let x0 = -5, x1 = 5, y0 = -5, y1 = 5;
    for (const [r, f] of pts) {
      const [x, y] = rot(r, f);
      x0 = Math.min(x0, x), x1 = Math.max(x1, x), y0 = Math.min(y0, y), y1 = Math.max(y1, y);
    }
    const pad = 14;
    const k = Math.min((width - 2 * pad) / (x1 - x0), (HEIGHT - 2 * pad) / (y1 - y0));
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    const P = (r: number, f: number): [number, number] => {
      const [x, y] = rot(r, f);
      return [width / 2 + (x - mx) * k, HEIGHT / 2 - (y - my) * k];
    };
    frameRef.current = { P };
    const [ox, oy] = P(0, 0);
    const maxR = Math.max(Math.abs(x0), Math.abs(x1)), maxF = Math.max(Math.abs(y0), Math.abs(y1));

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
      ctx.arc(ox, oy, d * k, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(169,180,192,0.14)";
      ctx.stroke();
      const [lx, ly] = P(0, d);
      if (ly > 10 && ly < HEIGHT - 4 && lx > 0 && lx < width - 30) {
        ctx.fillStyle = "rgba(169,180,192,0.55)";
        ctx.fillText(`${d} m`, lx + 3, ly - 1);
      }
    }
    // camera marker, pointing where the camera looks
    ctx.save();
    ctx.translate(ox, oy);
    ctx.rotate(phi);
    ctx.fillStyle = "#ff7a1a";
    ctx.beginPath();
    ctx.moveTo(0, -7);
    ctx.lineTo(-5, 3);
    ctx.lineTo(5, 3);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // compass rose (top right): N is at screen angle phi - yaw
    const north = phi - (yaw * Math.PI) / 180;
    const R = 15, cx0 = width - R - 9, cy0 = R + 9;
    ctx.fillStyle = "rgba(8,11,15,0.8)";
    ctx.strokeStyle = "rgba(169,180,192,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx0, cy0, R + 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.font = `700 8.5px "Inter", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ["N", "E", "S", "W"].forEach((lbl, i) => {
      const ang = north + (i * Math.PI) / 2;
      ctx.fillStyle = lbl === "N" ? "#ff4d4d" : "rgba(230,237,243,0.7)";
      ctx.fillText(lbl, cx0 + Math.sin(ang) * (R - 1), cy0 - Math.cos(ang) * (R - 1));
    });
    ctx.save();
    ctx.translate(cx0, cy0);
    ctx.rotate(north);
    for (const [tip, col] of [[-1, "#ff4d4d"], [1, "rgba(230,237,243,0.55)"]] as const) {
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(0, tip * (R - 7));
      ctx.lineTo(-3, 0);
      ctx.lineTo(3, 0);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
    ctx.textAlign = "start";

    // tracks
    ctx.textBaseline = "middle";
    for (const t of placed) {
      const col = t.locked ? LOCK_COLOR : classColor(t.cls);
      const [px, py] = P(...t.ground_xy);
      const x = Math.min(Math.max(px, 4), width - 4), y = Math.min(Math.max(py, 4), HEIGHT - 4);
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
  }, [header, hoverId, width, northUp, yaw]);

  const pick = (e: React.MouseEvent): Track | null => {
    const r = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const { P } = frameRef.current;
    let best: Track | null = null, bestD = 10;
    for (const t of placed) {
      const [x, y] = P(...t.ground_xy);
      const d = Math.hypot(x - mx, y - my);
      if (d < bestD) (best = t), (bestD = d);
    }
    return best;
  };

  const moving = placed.filter((t) => t.heading_deg != null).length;
  return (
    <Section title="Ground map" right={
      <span className="row gap-s">
        <span className="muted">{placed.length} on ground · {moving} moving</span>
        <button className={`chip-btn small-btn ${northUp ? "on" : ""}`} title="Switch between north up and camera direction up"
          onClick={() => setNorthUp(!northUp)}>{northUp ? "N ↑" : "Cam ↑"}</button>
      </span>
    }>
      <div ref={wrapRef} className="ground-map">
        <canvas ref={canvasRef} style={{ width: "100%", height: HEIGHT }}
          onMouseMove={(e) => { const t = pick(e); if ((t?.id ?? null) !== hoverId) onHover(t?.id ?? null); }}
          onMouseLeave={() => onHover(null)}
          onClick={(e) => { const t = pick(e); if (t) onToggleLock(t.id); }} />
      </div>
      <p className="muted small">
        {northUp ? <>North up · camera facing {compass(yaw)}. </> : <>Camera direction up (facing {compass(yaw)}). </>}
        The shaded area is what the camera sees. Arrows show where each target will be in {ARROW_S} s.
      </p>
    </Section>
  );
}
