export interface SystemInfo {
  gpu: string | null;
  torch: string;
  tensorrt: string | null;
  palettes: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  engine: boolean;
}

export interface VideoInfo {
  id: string;
  name: string;
  kind: "sample" | "upload";
  width: number;
  height: number;
  fps: number;
  frames: number;
}

export interface Params {
  conf: number;
  nms_iou: number;
  track_high: number;
  new_track: number;
  match_thresh: number;
  track_buffer_s: number;
  altitude_m: number;
  pitch_deg: number;
  focal_px: number;
  gmc: boolean;
  classes: number[] | null;
}

export interface Track {
  id: number;
  box: [number, number, number, number];
  score: number;
  cls: number;
  name: string;
  locked: boolean;
  coasting: boolean;
  age: number;
  ground_m: number | null;
  slant_m: number | null;
  speed_px: number;
}

export interface Timings {
  decode: number;
  wait: number;
  detect: number;
  gmc_extra: number;
  track: number;
  total: number;
  encode: number;
}

export interface FrameHeader {
  index: number;
  time: number;
  tracks: Track[];
  dets: number;
  motion: [number, number];
  /** camera motion prev->current frame, 2x3 row-major [a, b, tx, c, d, ty] */
  warp: [number, number, number, number, number, number];
  timings: Timings;
  fps: number;
  locked: number[];
}

export interface StreamMeta {
  width: number;
  height: number;
  fps: number;
  frames: number;
  names: Record<string, string>;
  backend: string;
  input: [number, number];
}

export interface ExportJob {
  id: string;
  status: "queued" | "loading" | "running" | "done" | "error" | "cancelled";
  progress: number;
  frames: number;
  total: number;
  error: string | null;
  result: {
    frames: number;
    seconds: number;
    fps: number;
    unique_ids: number;
    metrics: Record<string, number | null> | null;
  } | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).detail ?? msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  system: () => fetch("/api/system").then((r) => json<SystemInfo>(r)),
  models: () => fetch("/api/models").then((r) => json<ModelInfo[]>(r)),
  videos: () => fetch("/api/videos").then((r) => json<VideoInfo[]>(r)),
  defaults: () => fetch("/api/defaults").then((r) => json<Params>(r)),
  thumb: (id: string, palette: string) => `/api/videos/${id}/thumb.jpg?palette=${encodeURIComponent(palette)}`,
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/videos", { method: "POST", body: fd }).then((r) => json<VideoInfo>(r));
  },
  deleteVideo: (id: string) => fetch(`/api/videos/${id}`, { method: "DELETE" }).then((r) => json<{ ok: boolean }>(r)),
  startExport: (config: object, gt?: File | null) => {
    const fd = new FormData();
    fd.append("config", JSON.stringify(config));
    if (gt) fd.append("gt", gt);
    return fetch("/api/exports", { method: "POST", body: fd }).then((r) => json<ExportJob>(r));
  },
  exportStatus: (id: string) => fetch(`/api/exports/${id}`).then((r) => json<ExportJob>(r)),
  cancelExport: (id: string) => fetch(`/api/exports/${id}`, { method: "DELETE" }).then((r) => json<{ ok: boolean }>(r)),
  exportFile: (id: string, name: "video" | "tracks") => `/api/exports/${id}/${name}`,
};
