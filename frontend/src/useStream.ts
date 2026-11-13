import { useCallback, useEffect, useRef, useState } from "react";
import type { FrameHeader, Params, StreamMeta } from "./api";

export type StreamStatus = "idle" | "connecting" | "loading" | "playing" | "paused" | "ended" | "error";

export interface LiveFrame {
  bitmap: ImageBitmap;
  header: FrameHeader;
  /** trail points per visible track, in this frame's pixel coordinates */
  trails: Map<number, [number, number][]>;
}

export interface OpenOptions {
  video: string;
  model: string | null;
  params: Params;
  palette: string;
  autoplay?: boolean;
}

const HISTORY = 120;
const TRAIL_LEN = 24;

export interface Trail {
  pts: [number, number][];
  lastSeen: number;
}

/**
 * Warp trail points by the camera motion (full similarity transform, not just translation:
 * the drone also rotates and changes altitude) so they stay glued to the ground, then append
 * the new centres.
 */
function updateTrails(trails: Map<number, Trail>, h: FrameHeader) {
  const [a, b, tx, c, d, ty] = h.warp ?? [1, 0, h.motion[0], 0, 1, h.motion[1]];
  for (const t of trails.values())
    for (const p of t.pts) {
      const x = p[0], y = p[1];
      p[0] = a * x + b * y + tx;
      p[1] = c * x + d * y + ty;
    }
  for (const tr of h.tracks) {
    const pt: [number, number] = [(tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2];
    const t = trails.get(tr.id) ?? { pts: [], lastSeen: h.index };
    t.pts.push(pt);
    if (t.pts.length > TRAIL_LEN) t.pts.shift();
    t.lastSeen = h.index;
    trails.set(tr.id, t);
  }
  for (const [id, t] of trails) if (h.index - t.lastSeen > 15) trails.delete(id);
}

/**
 * Owns the /api/stream WebSocket.
 * Frames are kept in a ref (drawn by the canvas on rAF) so React only re-renders at ~10 Hz
 * for the track list and stats, not on every decoded frame.
 */
export function useStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const frameRef = useRef<LiveFrame | null>(null);
  const frameSeq = useRef(0);
  const historyRef = useRef<FrameHeader[]>([]);
  const trailsRef = useRef<Map<number, Trail>>(new Map());
  const lastOpen = useRef<OpenOptions | null>(null);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [meta, setMeta] = useState<StreamMeta | null>(null);
  const [header, setHeader] = useState<FrameHeader | null>(null);
  const [message, setMessage] = useState<string>("");
  const statusRef = useRef<StreamStatus>("idle");

  const setS = (s: StreamStatus) => {
    statusRef.current = s;
    setStatus(s);
  };

  const send = useCallback((msg: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }, []);

  const connect = useCallback((): Promise<WebSocket> => {
    const existing = wsRef.current;
    if (existing && existing.readyState === WebSocket.OPEN) return Promise.resolve(existing);
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/api/stream`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setS("connecting");
      ws.onopen = () => resolve(ws);
      ws.onerror = () => {
        setMessage("Cannot connect to the backend (WebSocket).");
        setS("error");
        reject(new Error("ws error"));
      };
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
        if (statusRef.current !== "error") setS("idle");
      };
      ws.onmessage = async (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data);
          switch (msg.type) {
            case "loading":
              setMessage(msg.message);
              setS("loading");
              break;
            case "ready":
              setMeta(msg.meta);
              setMessage("");
              setS(lastOpen.current?.autoplay === false ? "paused" : "playing");
              break;
            case "end":
              setS("ended");
              break;
            case "error":
              setMessage(msg.message);
              setS("error");
              break;
          }
          return;
        }
        const buf = ev.data as ArrayBuffer;
        const hlen = new DataView(buf).getUint32(0, true);
        const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen))) as FrameHeader;
        // trails must see every frame's camera motion in order, so update them before the async decode
        if (header.index === 0) trailsRef.current.clear();
        updateTrails(trailsRef.current, header);
        const trails = new Map<number, [number, number][]>();
        for (const t of header.tracks) {
          const tr = trailsRef.current.get(t.id);
          if (tr) trails.set(t.id, tr.pts.map((p) => [p[0], p[1]] as [number, number]));
        }
        const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, 4 + hlen)], { type: "image/jpeg" }));
        const prev = frameRef.current;
        if (prev && header.index <= prev.header.index && header.index !== 0) {
          bitmap.close(); // decoded out of order – drop the image only
          return;
        }
        prev?.bitmap.close();
        frameRef.current = { bitmap, header, trails };
        frameSeq.current++;
        const h = historyRef.current;
        h.push(header);
        if (h.length > HISTORY) h.shift();
      };
    });
  }, []);

  // publish the latest header to React at ~10 Hz
  useEffect(() => {
    let seen = -1;
    const id = window.setInterval(() => {
      if (frameSeq.current !== seen && frameRef.current) {
        seen = frameSeq.current;
        setHeader(frameRef.current.header);
      }
    }, 100);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  const open = useCallback(
    async (opts: OpenOptions) => {
      lastOpen.current = opts;
      historyRef.current = [];
      trailsRef.current.clear();
      frameRef.current = null;
      setHeader(null);
      setMeta(null);
      try {
        await connect();
      } catch {
        return;
      }
      setS("loading");
      send({ type: "open", ...opts, autoplay: opts.autoplay ?? true });
    },
    [connect, send],
  );

  const play = useCallback(() => {
    if (statusRef.current === "ended") {
      historyRef.current = [];
      send({ type: "restart" });
    }
    send({ type: "play" });
    setS("playing");
  }, [send]);

  const pause = useCallback(() => {
    send({ type: "pause" });
    setS("paused");
  }, [send]);

  const restart = useCallback(() => {
    historyRef.current = [];
    send({ type: "restart" });
    if (statusRef.current === "ended") setS("paused");
  }, [send]);

  return {
    status,
    meta,
    header,
    message,
    frameRef,
    frameSeq,
    historyRef,
    trailsRef,
    open,
    play,
    pause,
    restart,
    step: () => send({ type: "step" }),
    setSpeed: (value: number) => send({ type: "speed", value }),
    setPalette: (value: string) => send({ type: "palette", value }),
    setParams: (value: Partial<Params>) => send({ type: "params", value }),
    toggleLock: (id: number) => send({ type: "lock", id }),
    clearLocks: () => send({ type: "clear_locks" }),
  };
}

export type Stream = ReturnType<typeof useStream>;
