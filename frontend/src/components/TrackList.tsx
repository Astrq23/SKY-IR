import type { FrameHeader } from "../api";
import { compass, toBearing } from "../compass";
import { classColor, LOCK_COLOR } from "../theme";
import { Section } from "./ui";

const ARROWS = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"];

/** Ground speed in km/h with a camera-relative heading arrow (↑ = moving away from the camera). */
export function formatSpeed(speed: number | null, heading: number | null): string {
  if (speed == null) return "—";
  if (heading == null) return "0";
  return `${ARROWS[Math.round(heading / 45) % 8]} ${(speed * 3.6).toFixed(0)}`;
}

interface Props {
  /** camera compass heading, to turn camera-relative headings into bearings */
  yaw: number;
  header: FrameHeader | null;
  hoverId: number | null;
  onHover: (id: number | null) => void;
  onToggleLock: (id: number) => void;
  onClearLocks: () => void;
}

export function TrackList({ yaw, header, hoverId, onHover, onToggleLock, onClearLocks }: Props) {
  const tracks = [...(header?.tracks ?? [])].sort((a, b) => Number(b.locked) - Number(a.locked) || a.id - b.id);
  const locked = header?.locked ?? [];
  return (
    <Section
      title="Objects"
      right={
        <span className="row gap-s">
          <span className="muted">{tracks.length} track{tracks.length === 1 ? "" : "s"} · {header?.dets ?? 0} det{header?.dets === 1 ? "" : "s"}</span>
          {locked.length > 0 && <button className="link-btn" onClick={onClearLocks}>Unlock all</button>}
        </span>
      }
    >
      {/* fixed-height box: rows coming and going must not push the panels below up and down */}
      <div className="tracks-scroll">
      {tracks.length === 0 ? (
        <p className="empty">No objects yet. Click an object in the video to lock it as a target.</p>
      ) : (
        <table className="tracks">
          <colgroup>
            {/* widths fit the longest value ("123", "0.86", "1234 m", "↙ 120", "999") so columns never shift */}
            <col style={{ width: 32 }} /><col /><col style={{ width: 46 }} /><col style={{ width: 54 }} />
            <col style={{ width: 46 }} /><col style={{ width: 34 }} /><col style={{ width: 22 }} />
          </colgroup>
          <thead>
            <tr><th>ID</th><th>Class</th><th>Conf</th><th>Range</th><th title="Ground speed, km/h. Arrow: ↑ away from camera, → to the right">km/h</th><th>Age</th><th /></tr>
          </thead>
          <tbody>
            {tracks.map((t) => (
              <tr key={t.id} className={`${t.id === hoverId ? "hover" : ""} ${t.locked ? "locked" : ""} ${t.coasting ? "coasting" : ""}`}
                onMouseEnter={() => onHover(t.id)} onMouseLeave={() => onHover(null)} onClick={() => onToggleLock(t.id)}>
                <td className="mono">{t.id}</td>
                <td title={t.size_m == null ? undefined : `Size ≈ ${t.size_m.toFixed(1)} m${t.size_ok ? "" : " (unusual for this class)"}`}>
                  <span className="dot" style={{ background: t.locked ? LOCK_COLOR : classColor(t.cls) }} />{t.name}
                  {!t.size_ok && <span className="warn" title="Size does not fit this class">!</span>}
                </td>
                <td className="mono">{t.coasting ? "—" : t.score.toFixed(2)}</td>
                <td className="mono">{t.ground_m == null ? "∞" : `${t.ground_m.toFixed(0)} m`}</td>
                <td className="mono" title={t.heading_deg == null ? "Stationary" : `Heading ${compass(toBearing(t.heading_deg, yaw))}`}>{formatSpeed(t.speed_ms, t.heading_deg)}</td>
                <td className="mono">{t.age}</td>
                <td><span className={`lock ${t.locked ? "on" : ""}`} title={t.locked ? "Unlock" : "Lock target"}>◎</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      </div>
    </Section>
  );
}
