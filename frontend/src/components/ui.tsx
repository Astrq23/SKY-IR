import type { ReactNode } from "react";

export function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="section">
      <header className="section-head">
        <h3>{title}</h3>
        {right}
      </header>
      <div className="section-body">{children}</div>
    </section>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  hint?: string;
  onChange: (v: number) => void;
}

export function Slider({ label, value, min, max, step, unit, hint, onChange }: SliderProps) {
  const decimals = step < 1 ? Math.max(0, -Math.floor(Math.log10(step))) : 0;
  return (
    <label className="field slider" title={hint}>
      <span className="field-row">
        <span className="field-label">{label}</span>
        <span className="field-value">
          {value.toFixed(decimals)}
          {unit && <small>{unit}</small>}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ ["--pct" as string]: `${((value - min) / (max - min)) * 100}%` }}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function NumberField({ label, value, unit, min, max, step = 1, onChange }: {
  label: string; value: number; unit?: string; min?: number; max?: number; step?: number; onChange: (v: number) => void;
}) {
  return (
    <label className="field number">
      <span className="field-label">{label}</span>
      <span className="number-wrap">
        <input type="number" value={value} min={min} max={max} step={step}
          onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) onChange(v); }} />
        {unit && <small>{unit}</small>}
      </span>
    </label>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="toggle" title={hint}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track"><span className="toggle-thumb" /></span>
      <span>{label}</span>
    </label>
  );
}

export function Segmented<T extends string | number>({ options, value, onChange }: {
  options: { value: T; label: string }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {options.map((o) => (
        <button key={String(o.value)} role="radio" aria-checked={o.value === value}
          className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
