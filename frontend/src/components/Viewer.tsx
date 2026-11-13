import { useEffect, useRef, useState } from "react";
import type { Track } from "../api";
import type { Stream } from "../useStream";
import { classColor, LOCK_COLOR } from "../theme";

export interface Overlay {
  style: "corners" | "box";
  labels: boolean;
  distance: boolean;
  trails: boolean;
  lockedOnly: boolean;
  inset: boolean;
}

interface Props {
  stream: Stream;
  overlay: Overlay;
  hoverId: number | null;
  onHover: (id: number | null) => void;
}

interface View {
  scale: number;
  ox: number;
  oy: number;
}

const INSET = 168;

/**
 * Centred moving average (window 5) to hide detector box jitter on small thermal blobs.
 * The newest point is pinned to the real position so the trail stays attached to the box.
 */
function smoothTrail(pts: [number, number][], half = 2): [number, number][] {
  const n = pts.length;
  return pts.map((p, i) => {
    if (i === n - 1) return p;
    const k = Math.min(half, i, n - 1 - i);
    let x = 0, y = 0;
    for (let j = i - k; j <= i + k; j++) (x += pts[j][0]), (y += pts[j][1]);
    return [x / (2 * k + 1), y / (2 * k + 1)];
  });
}

function hitTest(tracks: Track[], x: number, y: number, scale: number): Track | null {
  let best: Track | null = null;
  let bestD = Infinity;
  const pad = 8 / scale; // ~8 screen px tolerance: targets are tiny
  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.box;
    if (x >= x1 - pad && x <= x2 + pad && y >= y1 - pad && y <= y2 + pad) {
      const d = Math.hypot((x1 + x2) / 2 - x, (y1 + y2) / 2 - y);
      if (d < bestD) (best = t), (bestD = d);
    }
  }
  return best;
}

export function Viewer({ stream, overlay, hoverId, onHover }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<View>({ scale: 1, ox: 0, oy: 0 });
  const dirty = useRef(true);
  const [cursor, setCursor] = useState("default");
  const overlayRef = useRef(overlay);
  const hoverRef = useRef(hoverId);
  overlayRef.current = overlay;
  hoverRef.current = hoverId;

  useEffect(() => {
    dirty.current = true;
  }, [overlay, hoverId]);

  // keep the canvas backing store matched to its CSS size * devicePixelRatio
  useEffect(() => {
    const wrap = wrapRef.current!;
    const ro = new ResizeObserver(() => {
      const c = canvasRef.current!;
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.max(1, Math.round(wrap.clientWidth * dpr));
      c.height = Math.max(1, Math.round(wrap.clientHeight * dpr));
      dirty.current = true;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // render loop: redraw only when a new frame arrived or the overlay changed
  useEffect(() => {
    let raf = 0;
    let seen = -1;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const f = stream.frameRef.current;
      if (!f || (seen === stream.frameSeq.current && !dirty.current)) return;
      seen = stream.frameSeq.current;
      dirty.current = false;
      const c = canvasRef.current!;
      const ctx = c.getContext("2d")!;
      const dpr = window.devicePixelRatio || 1;
      const { bitmap, header } = f;
      const scale = Math.min(c.width / bitmap.width, c.height / bitmap.height);
      const ox = (c.width - bitmap.width * scale) / 2;
      const oy = (c.height - bitmap.height * scale) / 2;
      viewRef.current = { scale: scale / dpr, ox: ox / dpr, oy: oy / dpr };

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#06080b";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.imageSmoothingEnabled = scale < 2;
      ctx.drawImage(bitmap, ox, oy, bitmap.width * scale, bitmap.height * scale);

      const ov = overlayRef.current;
      const anyLocked = header.tracks.some((t) => t.locked);
      const tracks = ov.lockedOnly && anyLocked ? header.tracks.filter((t) => t.locked) : header.tracks;
      const X = (x: number) => ox + x * scale;
      const Y = (y: number) => oy + y * scale;
      const u = dpr; // 1 css px

      if (ov.trails) {
        ctx.lineWidth = 1.5 * u;
        ctx.lineJoin = "round";
        for (const t of tracks) {
          const raw = f.trails.get(t.id);
          if (!raw || raw.length < 3) continue;
          const pts = smoothTrail(raw);
          const col = t.locked ? LOCK_COLOR : classColor(t.cls);
          ctx.strokeStyle = col;
          for (let i = 1; i < pts.length; i++) {
            ctx.globalAlpha = (i / pts.length) * 0.85; // fade towards the tail
            ctx.beginPath();
            ctx.moveTo(X(pts[i - 1][0]), Y(pts[i - 1][1]));
            ctx.lineTo(X(pts[i][0]), Y(pts[i][1]));
            ctx.stroke();
          }
        }
        ctx.globalAlpha = 1;
      }

      ctx.font = `${600} ${11 * u}px "JetBrains Mono", ui-monospace, monospace`;
      ctx.textBaseline = "middle";
      for (const t of tracks) {
        const hovered = t.id === hoverRef.current;
        const col = t.locked ? LOCK_COLOR : classColor(t.cls);
        let x1 = X(t.box[0]), y1 = Y(t.box[1]), x2 = X(t.box[2]), y2 = Y(t.box[3]);
        const minSide = 14 * u; // tiny targets: draw a readable minimum-size reticle
        if (x2 - x1 < minSide) { const cx = (x1 + x2) / 2; x1 = cx - minSide / 2; x2 = cx + minSide / 2; }
        if (y2 - y1 < minSide) { const cy = (y1 + y2) / 2; y1 = cy - minSide / 2; y2 = cy + minSide / 2; }
        ctx.strokeStyle = col;
        ctx.lineWidth = (t.locked || hovered ? 2 : 1.25) * u;
        ctx.setLineDash(t.coasting ? [4 * u, 3 * u] : []);
        if (ov.style === "box" || t.coasting) {
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        } else {
          const L = Math.max(4 * u, Math.min(x2 - x1, y2 - y1) * 0.32);
          ctx.beginPath();
          for (const [px, py, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
            ctx.moveTo(px + dx * L, py);
            ctx.lineTo(px, py);
            ctx.lineTo(px, py + dy * L);
          }
          ctx.stroke();
        }
        ctx.setLineDash([]);
        if (t.locked) {
          const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, r = Math.max(x2 - x1, y2 - y1) / 2 + 8 * u;
          ctx.beginPath();
          for (const [a, b] of [[-r - 6 * u, -r + 2 * u], [r - 2 * u, r + 6 * u]]) {
            ctx.moveTo(cx + a, cy); ctx.lineTo(cx + b, cy);
            ctx.moveTo(cx, cy + a); ctx.lineTo(cx, cy + b);
          }
          ctx.stroke();
        }
        if (ov.labels || hovered || t.locked) {
          let label = `${t.locked ? "◎ " : ""}${t.id} ${t.name}`;
          if (ov.distance && t.ground_m != null) label += ` ${t.ground_m.toFixed(0)}m`;
          const w = ctx.measureText(label).width + 8 * u;
          const h = 16 * u;
          const ly = y1 - h - 3 * u > 0 ? y1 - h - 3 * u : y2 + 3 * u;
          ctx.fillStyle = "rgba(8,11,15,0.82)";
          ctx.fillRect(x1, ly, w, h);
          ctx.fillStyle = col;
          ctx.fillRect(x1, ly, 2 * u, h);
          ctx.fillText(label, x1 + 5 * u, ly + h / 2 + 0.5 * u);
        }
      }

      // zoomed inset of the first locked (or hovered) target
      const focus = header.tracks.find((t) => t.locked) ?? header.tracks.find((t) => t.id === hoverRef.current);
      if (ov.inset && focus) {
        const [bx1, by1, bx2, by2] = focus.box;
        const side = Math.max(32, (Math.max(bx2 - bx1, by2 - by1)) * 2.5);
        const S = Math.min(INSET * u, c.width * 0.28);
        const zoom = S / u / side; // css px per image px
        if (zoom < 1.5 * scale / u) return; // inset must magnify beyond the main view
        const sx = Math.min(Math.max((bx1 + bx2) / 2 - side / 2, 0), bitmap.width - side);
        const sy = Math.min(Math.max((by1 + by2) / 2 - side / 2, 0), bitmap.height - side);
        // put the inset in the corner opposite the target so it never hides it
        const tx = X((bx1 + bx2) / 2), ty = Y((by1 + by2) / 2);
        const ix = tx > c.width / 2 ? 12 * u : c.width - S - 12 * u;
        const iy = ty < c.height / 2 ? c.height - S - 30 * u : 12 * u;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(bitmap, sx, sy, side, side, ix, iy, S, S);
        const k = S / side;
        ctx.strokeStyle = focus.locked ? LOCK_COLOR : classColor(focus.cls);
        ctx.lineWidth = 1.5 * u;
        ctx.strokeRect((bx1 - sx) * k + ix, (by1 - sy) * k + iy, (bx2 - bx1) * k, (by2 - by1) * k);
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.strokeRect(ix, iy, S, S);
        ctx.fillStyle = "rgba(8,11,15,0.82)";
        ctx.fillRect(ix, iy + S, S, 18 * u);
        ctx.fillStyle = "#e6edf3";
        ctx.fillText(`ID ${focus.id} · ${(zoom / (scale / u)).toFixed(1)}× zoom`, ix + 6 * u, iy + S + 9 * u);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [stream.frameRef, stream.frameSeq]);

  const toImage = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    const v = viewRef.current;
    return { x: (e.clientX - r.left - v.ox) / v.scale, y: (e.clientY - r.top - v.oy) / v.scale, scale: v.scale };
  };

  const onMove = (e: React.MouseEvent) => {
    const f = stream.frameRef.current;
    if (!f) return;
    const p = toImage(e);
    const t = hitTest(f.header.tracks, p.x, p.y, p.scale);
    setCursor(t ? "pointer" : "crosshair");
    if ((t?.id ?? null) !== hoverRef.current) onHover(t?.id ?? null);
  };

  const onClick = (e: React.MouseEvent) => {
    const f = stream.frameRef.current;
    if (!f) return;
    const p = toImage(e);
    const t = hitTest(f.header.tracks, p.x, p.y, p.scale);
    if (t) stream.toggleLock(t.id);
  };

  return (
    <div className="viewer" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ cursor }} onMouseMove={onMove} onMouseLeave={() => onHover(null)} onClick={onClick} />
    </div>
  );
}
