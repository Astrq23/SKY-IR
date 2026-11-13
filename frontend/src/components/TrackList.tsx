import type { FrameHeader } from "../api";
import { classColor, LOCK_COLOR } from "../theme";
import { Section } from "./ui";

interface Props {
  header: FrameHeader | null;
  hoverId: number | null;
  onHover: (id: number | null) => void;
  onToggleLock: (id: number) => void;
  onClearLocks: () => void;
}

export function TrackList({ header, hoverId, onHover, onToggleLock, onClearLocks }: Props) {
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
      {tracks.length === 0 ? (
        <p className="empty">No objects yet. Click an object in the video to lock it as a target.</p>
      ) : (
        <table className="tracks">
          <thead>
            <tr><th>ID</th><th>Class</th><th>Conf</th><th>Range</th><th>Age</th><th /></tr>
          </thead>
          <tbody>
            {tracks.map((t) => (
              <tr key={t.id} className={`${t.id === hoverId ? "hover" : ""} ${t.locked ? "locked" : ""} ${t.coasting ? "coasting" : ""}`}
                onMouseEnter={() => onHover(t.id)} onMouseLeave={() => onHover(null)} onClick={() => onToggleLock(t.id)}>
                <td className="mono">{t.id}</td>
                <td><span className="dot" style={{ background: t.locked ? LOCK_COLOR : classColor(t.cls) }} />{t.name}</td>
                <td className="mono">{t.coasting ? "—" : t.score.toFixed(2)}</td>
                <td className="mono">{t.ground_m == null ? "∞" : `${t.ground_m.toFixed(0)} m`}</td>
                <td className="mono">{t.age}</td>
                <td><span className={`lock ${t.locked ? "on" : ""}`} title={t.locked ? "Unlock" : "Lock target"}>◎</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}
