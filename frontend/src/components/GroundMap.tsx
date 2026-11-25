import { useEffect, useRef, useState } from "react";
import type { FrameHeader, Track } from "../api";
import { compass, toBearing } from "../compass";
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
const STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];
const CARD_W = 180;

/** Info card for a target (hovered on the map or in the Objects table). */
function cardState(t: Track): string {
  return t.locked ? "locked" : t.coasting ? "predicted" : `conf ${t.score.toFixed(2)}`;
}

function cardLines(t: Track, yaw: number): string[] {
  const motion = t.speed_ms == null ? "speed —"
    : t.heading_deg == null ? "stationary"
    : `${(t.speed_ms * 3.6).toFixed(1)} km/h ${compass(toBearing(t.heading_deg, yaw))}`;
  return [
    `${t.ground_m == null ? "range —" : `${t.ground_m.toFixed(0)} m away`} · ${t.size_m == null ? "size —" : `≈ ${t.size_m.toFixed(1)} m`}${t.size_ok ? "" : " (!)"}`,
    motion,
  ];
}

/** Largest "nice" length (m) that is at most `max`. */
const niceAtMost = (max: number) => [...STEPS].reverse().find((s) => s <= max) ?? STEPS[0];

/**
 * What the map shows, in rotated map coordinates (x right, y up on screen, metres from the camera):
 * point (cx, cy) at the canvas centre, k css px per metre.
 */
interface View {
  cx: number;
  cy: number;
  k: number;
}

type Mode = "fit" | "manual" | "follow";

const NORTH_UP_KEY = "skyir.map.northUp";

/** Camera-relative ground (right, forward) -> map coordinates, rotated by `phi` (radians, clockwise). */
function rotate(phi: number, r: number, f: number): [number, number] {
  const c = Math.cos(phi), s = Math.sin(phi);
  return [r * c + f * s, -r * s + f * c];
}

/**
 * Bird's-eye view of the ground around the UAV: a square metre grid with a scale bar, the camera
 * footprint, a compass and each track with a velocity arrow. "Camera up" puts the viewing direction
 * at the top; "north up" rotates the map by the camera's compass heading (yaw) so north is at the top.
 * Scroll to zoom, drag to pan, Fit / Follow / Expand.
 */
export function GroundMap({ yaw, header, hoverId, onHover, onToggleLock }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 300, height: HEIGHT });
  const { width, height } = size;
  // "fit": frame the camera footprint (stable: changes only with altitude/pitch/focal),
  // "manual": after the user zoomed or panned, "follow": keep the locked target centred
  const [mode, setMode] = useState<Mode>("fit");
  const [view, setView] = useState<View>({ cx: 0, cy: 0, k: 0 });
  const [expanded, setExpanded] = useState(false);
  const viewRef = useRef(view);
  viewRef.current = view;
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const [northUp, setNorthUp] = useState(() => localStorage.getItem(NORTH_UP_KEY) === "1");
  useEffect(() => {
    try {
      localStorage.setItem(NORTH_UP_KEY, northUp ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  }, [northUp]);
  // map rotation: screen angle (clockwise from up) of the camera's forward axis
  const phi = northUp ? (yaw * Math.PI) / 180 : 0;

  useEffect(() => {
    const wrap = wrapRef.current!;
    const ro = new ResizeObserver(() => setSize({ width: wrap.clientWidth || 300, height: wrap.clientHeight || HEIGHT }));
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // fit / follow: derive the view from the data; manual: keep what the user set
  const footprintKey = JSON.stringify(header?.footprint ?? []);
  const locked = header?.tracks.find((t) => t.locked && t.ground_xy);
  useEffect(() => {
    if (mode === "manual") return;
    // footprint + camera after rotation: changes only with altitude/pitch/focal/heading
    const fp: [number, number][] = [[0, 0], ...(header?.footprint ?? [])];
    let x0 = -5, x1 = 5, y0 = -5, y1 = 5;
    for (const [r, f] of fp) {
      const [x, y] = rotate(phi, r, f);
      (x0 = Math.min(x0, x)), (x1 = Math.max(x1, x)), (y0 = Math.min(y0, y)), (y1 = Math.max(y1, y));
    }
    const pad = 16;
    const fitK = Math.min((width - 2 * pad) / (x1 - x0), (height - 2 * pad) / (y1 - y0));
    if (mode === "fit") setView({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, k: fitK });
    else if (locked?.ground_xy) {
      const [x, y] = rotate(phi, ...locked.ground_xy);
      setView((v) => ({ cx: x, cy: y, k: v.k || fitK }));
    }
  }, [mode, footprintKey, width, height, phi, locked?.ground_xy?.[0], locked?.ground_xy?.[1]]);
  useEffect(() => {
    if (mode === "follow" && !locked) setMode("fit");
  }, [mode, locked]);

  // wheel zoom around the cursor; a native listener because React's onWheel is passive
  useEffect(() => {
    const c = canvasRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const v = viewRef.current;
      if (!v.k) return;
      const W = rect.width, H = rect.height;
      const gr = v.cx + (mx - W / 2) / v.k, gf = v.cy - (my - H / 2) / v.k; // ground point under the cursor
      const k = Math.min(Math.max(v.k * Math.exp(-e.deltaY * 0.0015), 0.02), 100);
      setView({ k, cx: gr - (mx - W / 2) / k, cy: gf + (my - H / 2) / k });
      setMode((m) => (m === "follow" ? "follow" : "manual"));
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, []);

  const placed = (header?.tracks ?? []).filter((t): t is Track & { ground_xy: [number, number] } => t.ground_xy != null);

  useEffect(() => {
    const c = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#06080b";
    ctx.fillRect(0, 0, width, height);
    if (!header || !view.k) return;

    const { k } = view;
    const P = (r: number, f: number): [number, number] => toScreen(view, phi, width, height, r, f);
    const [ox, oy] = P(0, 0); // camera

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

    // square metre grid, aligned with the screen (so it rotates with the map in north-up mode) and
    // passing through the camera; ~30-60 px per cell. Lines are at map coordinates x, y = n * step.
    const step = STEPS.find((s) => s * k >= 30) ?? STEPS[STEPS.length - 1];
    const xMin = view.cx - width / 2 / k, xMax = view.cx + width / 2 / k;
    const yMin = view.cy - height / 2 / k, yMax = view.cy + height / 2 / k;
    const sx = (x: number) => Math.round(width / 2 + (x - view.cx) * k) + 0.5;
    const sy = (y: number) => Math.round(height / 2 - (y - view.cy) * k) + 0.5;
    ctx.lineWidth = 1;
    for (let x = Math.ceil(xMin / step) * step; x <= xMax; x += step) {
      ctx.strokeStyle = x === 0 ? "rgba(169,180,192,0.22)" : "rgba(169,180,192,0.09)";
      ctx.beginPath();
      ctx.moveTo(sx(x), 0);
      ctx.lineTo(sx(x), height);
      ctx.stroke();
    }
    ctx.font = `10px "JetBrains Mono", ui-monospace, monospace`;
    ctx.textBaseline = "bottom";
    for (let y = Math.ceil(yMin / step) * step; y <= yMax; y += step) {
      ctx.strokeStyle = y === 0 ? "rgba(169,180,192,0.22)" : "rgba(169,180,192,0.09)";
      ctx.beginPath();
      ctx.moveTo(0, sy(y));
      ctx.lineTo(width, sy(y));
      ctx.stroke();
      // distance ahead of the camera (camera up) or north of it (north up), on the left edge
      if (y > 0 && sy(y) > 12 && sy(y) < height - 18) {
        ctx.fillStyle = "rgba(169,180,192,0.6)";
        ctx.fillText(`${y} m`, 4, sy(y) - 1);
      }
    }

    // scale bar (bottom right)
    const bar = niceAtMost(90 / k);
    const bx1 = width - 10, bx0 = bx1 - bar * k, by = height - 9;
    ctx.fillStyle = "rgba(8,11,15,0.7)";
    ctx.fillRect(bx0 - 6, by - 16, bx1 - bx0 + 12, 21);
    ctx.strokeStyle = "rgba(230,237,243,0.85)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(bx0, by - 4);
    ctx.lineTo(bx0, by);
    ctx.lineTo(bx1, by);
    ctx.lineTo(bx1, by - 4);
    ctx.stroke();
    ctx.fillStyle = "rgba(230,237,243,0.9)";
    ctx.textAlign = "center";
    ctx.fillText(`${bar} m`, (bx0 + bx1) / 2, by - 3);
    ctx.textAlign = "start";
    ctx.lineWidth = 1;

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
      // in Fit mode keep targets outside the footprint visible at the edge; when zoomed/panned, let
      // them leave the view instead of piling up along the border
      const clamp = mode === "fit";
      const x = clamp ? Math.min(Math.max(px, 4), width - 4) : px, y = clamp ? Math.min(Math.max(py, 4), height - 4) : py;
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
  }, [header, hoverId, width, height, view, mode, phi, yaw]);

  const pick = (e: React.MouseEvent): Track | null => {
    const r = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best: Track | null = null, bestD = 10;
    for (const t of placed) {
      const [x, y] = toScreen(view, phi, width, height, ...t.ground_xy);
      const d = Math.hypot(x - mx, y - my);
      if (d < bestD) (best = t), (bestD = d);
    }
    return best;
  };

  // drag to pan; a press without movement is a click (lock / unlock the target under it)
  const onDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
  };
  const onMove = (e: React.MouseEvent) => {
    const d = drag.current;
    if (d && (d.moved || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 3)) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      setView((v) => ({ ...v, cx: v.cx - dx / v.k, cy: v.cy + dy / v.k }));
      setMode("manual");
      Object.assign(d, { x: e.clientX, y: e.clientY, moved: true });
      return;
    }
    const t = pick(e);
    if ((t?.id ?? null) !== hoverId) onHover(t?.id ?? null);
  };
  const onUp = (e: React.MouseEvent) => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) {
      const t = pick(e);
      if (t) onToggleLock(t.id);
    }
  };

  const moving = placed.filter((t) => t.heading_deg != null).length;
  // info card next to the hovered target (also when hovered in the Objects table), flipped to the
  // left near the right edge; hidden while the target is outside the view
  const hovered = placed.find((t) => t.id === hoverId);
  let card: { left: number; top: number; t: typeof placed[number] } | null = null;
  if (hovered && view.k) {
    const [x, y] = toScreen(view, phi, width, height, ...hovered.ground_xy);
    if (x >= 0 && x <= width && y >= 0 && y <= height) {
      const left = x + 12 + CARD_W > width ? x - 12 - CARD_W : x + 12;
      card = { left: Math.max(4, left), top: Math.min(Math.max(4, y - 22), height - 60), t: hovered };
    }
  }
  const tools = (
    <span className="map-tools">
      <button className={`chip-btn ${northUp ? "on" : ""}`} title="Switch between north up and camera direction up"
        onClick={() => setNorthUp(!northUp)}>{northUp ? "N ↑" : "Cam ↑"}</button>
      <button className={`chip-btn ${mode === "follow" ? "on" : ""}`} disabled={!locked} title="Keep the locked target centred"
        onClick={() => setMode(mode === "follow" ? "fit" : "follow")}>Follow</button>
      <button className={`chip-btn ${mode === "fit" ? "on" : ""}`} title="Fit the camera footprint (or double-click the map)" onClick={() => setMode("fit")}>Fit</button>
      <button className="chip-btn" title={expanded ? "Collapse" : "Expand"} onClick={() => setExpanded(!expanded)}>{expanded ? "✕" : "⤢"}</button>
    </span>
  );
  return (
    <>
      {expanded && <div className="map-backdrop" onClick={() => setExpanded(false)} />}
      <div className={expanded ? "map-expanded" : undefined}>
        <Section title="Ground map" right={<span className="muted">{placed.length} on ground · {moving} moving</span>}>
          {tools}
          <div ref={wrapRef} className="ground-map" style={expanded ? undefined : { height: HEIGHT }}>
            <canvas ref={canvasRef} className={drag.current?.moved ? "panning" : undefined}
              onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onDoubleClick={() => setMode("fit")}
              onMouseLeave={() => { drag.current = null; onHover(null); }} />
            {mode === "manual" && <div className="map-badge">Manual view · double-click to fit</div>}
            {card && (
              <div className="map-card" style={{ left: card.left, top: card.top, width: CARD_W }}>
                <div className="map-card-title">
                  <span className="dot" style={{ background: card.t.locked ? LOCK_COLOR : classColor(card.t.cls) }} />
                  #{card.t.id} {card.t.name} <span className="muted">· {cardState(card.t)}</span>
                </div>
                {cardLines(card.t, yaw).map((l) => <div key={l} className="mono">{l}</div>)}
              </div>
            )}
          </div>
          <p className="muted small">
            {northUp ? <>North up · camera facing {compass(yaw)}. </> : <>Camera direction up (facing {compass(yaw)}). </>}
            The shaded area is what the camera sees. Arrows show where each target will be in {ARROW_S} s.
            Scroll to zoom, drag to pan, click a target to lock it.
          </p>
        </Section>
      </div>
    </>
  );
}

function toScreen(v: View, phi: number, width: number, height: number, r: number, f: number): [number, number] {
  const [x, y] = rotate(phi, r, f);
  return [width / 2 + (x - v.cx) * v.k, height / 2 - (y - v.cy) * v.k];
}
