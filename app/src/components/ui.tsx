import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import { dismissToast, useStore } from "../store";
import { clamp, hexToRgba, rgbToHex } from "../lib/format";
import { t } from "../i18n";

// ---------------------------------------------------------------------------------------------
export function Segmented<T extends string | number>(props: {
  value: T;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  full?: boolean;
  ariaLabel?: string;
}) {
  return (
    <div className={`seg${props.full ? " full" : ""}`} role="radiogroup" aria-label={props.ariaLabel}>
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === props.value}
          className={o.value === props.value ? "on" : ""}
          disabled={o.disabled}
          data-tip={o.title}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
export function Slider(props: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  const p = ((props.value - props.min) / (props.max - props.min)) * 100;
  return (
    <div className="slider">
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        aria-label={props.ariaLabel}
        disabled={props.disabled}
        style={{ ["--p" as string]: `${clamp(p, 0, 100)}%` }}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
      <span className="value">{props.format ? props.format(props.value) : props.value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
/** Number input with an optional prefix label that can be dragged to scrub the value. */
export function NumberField(props: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  unit?: string;
  ariaLabel?: string;
  disabled?: boolean;
  integer?: boolean;
}) {
  const [text, setText] = useState(String(props.value));
  const focused = useRef(false);
  const discard = useRef(false); // Escape: leave without committing what was typed
  useEffect(() => {
    if (!focused.current) setText(String(props.value));
  }, [props.value]);
  const lo = props.min ?? -Infinity;
  const hi = props.max ?? Infinity;
  const commit = (raw: string) => {
    let v = Number(raw.replace(",", "."));
    if (!Number.isFinite(v)) {
      setText(String(props.value));
      return;
    }
    v = clamp(props.integer !== false ? Math.round(v) : v, lo, hi);
    setText(String(v));
    if (v !== props.value) props.onChange(v);
  };
  const scrub = (e: React.PointerEvent) => {
    if (props.disabled) return;
    e.preventDefault();
    const startX = e.clientX;
    const start = props.value;
    const step = props.step ?? 1;
    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      const mult = ev.shiftKey ? 10 : 1;
      let v = start + Math.round(dx / 3) * step * mult;
      v = clamp(v, lo, hi);
      props.onChange(v);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  return (
    <div className="numfield">
      {props.label && (
        <span className="label" onPointerDown={scrub}>
          {props.label}
        </span>
      )}
      <input
        inputMode="decimal"
        value={text}
        disabled={props.disabled}
        aria-label={props.ariaLabel ?? props.label}
        onFocus={(e) => {
          focused.current = true;
          e.currentTarget.select();
        }}
        onBlur={(e) => {
          focused.current = false;
          if (discard.current) {
            discard.current = false;
            setText(String(props.value));
            return;
          }
          commit(e.currentTarget.value);
        }}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "Escape") {
            discard.current = true;
            (e.target as HTMLInputElement).blur();
          }
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            const d = (e.key === "ArrowUp" ? 1 : -1) * (props.step ?? 1) * (e.shiftKey ? 10 : 1);
            const v = clamp(props.value + d, lo, hi);
            props.onChange(v);
            setText(String(v));
          }
        }}
      />
      {props.unit && <span className="unit">{props.unit}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
export function Switch(props: { checked: boolean; onChange: (v: boolean) => void; label?: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={props.checked}
      aria-label={props.label}
      disabled={props.disabled}
      className={`switch${props.checked ? " on" : ""}`}
      onClick={(e) => {
        e.stopPropagation();
        props.onChange(!props.checked);
      }}
    />
  );
}

export function CheckRow(props: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode; disabled?: boolean }) {
  return (
    <label className="check" style={{ opacity: props.disabled ? 0.5 : 1 }}>
      <Switch checked={props.checked} onChange={props.onChange} disabled={props.disabled} />
      <span>{props.children}</span>
    </label>
  );
}

// ---------------------------------------------------------------------------------------------
export interface MenuItem {
  label?: ReactNode;
  icon?: ReactNode;
  hint?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  danger?: boolean;
  sep?: boolean;
  title?: string;
}

/** Popover menu anchored to an element. */
export function Menu(props: { anchor: HTMLElement | null; items: MenuItem[]; onClose: () => void; align?: "left" | "right"; width?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const [focus, setFocus] = useState(-1);
  useLayoutEffect(() => {
    if (!props.anchor || !ref.current) return;
    const r = props.anchor.getBoundingClientRect();
    const m = ref.current.getBoundingClientRect();
    let left = props.align === "right" ? r.right - m.width : r.left;
    left = clamp(left, 8, window.innerWidth - m.width - 8);
    let top = r.bottom + 4;
    if (top + m.height > window.innerHeight - 8) top = Math.max(8, r.top - m.height - 4);
    setPos({ left, top });
  }, [props.anchor, props.align]);
  useEffect(() => {
    const down = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && !props.anchor?.contains(e.target as Node)) props.onClose();
    };
    const key = (e: KeyboardEvent) => {
      const actionable = props.items.map((it, i) => (!it.sep && !it.title && !it.disabled ? i : -1)).filter((i) => i >= 0);
      if (e.key === "Escape") props.onClose();
      else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const cur = actionable.indexOf(focus);
        const next = e.key === "ArrowDown" ? actionable[(cur + 1) % actionable.length] : actionable[(cur - 1 + actionable.length) % actionable.length];
        setFocus(next ?? -1);
      } else if (e.key === "Enter" && focus >= 0) {
        e.preventDefault();
        props.items[focus]?.onClick?.();
        props.onClose();
      }
    };
    window.addEventListener("mousedown", down, true);
    window.addEventListener("keydown", key, true);
    return () => {
      window.removeEventListener("mousedown", down, true);
      window.removeEventListener("keydown", key, true);
    };
  }, [props, focus]);
  return createPortal(
    <div ref={ref} className="menu" role="menu" style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999, width: props.width }}>
      {props.items.map((it, i) =>
        it.sep ? (
          <div key={i} className="menu-sep" />
        ) : it.title ? (
          <div key={i} className="menu-title">
            {it.title}
          </div>
        ) : (
          <button
            key={i}
            role="menuitem"
            className={`menu-item${it.danger ? " danger" : ""}${focus === i ? " focus" : ""}`}
            disabled={it.disabled}
            onMouseEnter={() => setFocus(i)}
            onClick={() => {
              it.onClick?.();
              props.onClose();
            }}
          >
            {it.icon}
            <span style={{ flex: 1 }}>{it.label}</span>
            {it.hint && <span className="hint">{it.hint}</span>}
          </button>
        ),
      )}
    </div>,
    document.body,
  );
}

export function useMenu() {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  return {
    anchor,
    open: (el: HTMLElement) => setAnchor((a) => (a ? null : el)),
    close: () => setAnchor(null),
  };
}

// ---------------------------------------------------------------------------------------------
export function Dialog(props: { title: ReactNode; onClose: () => void; children: ReactNode; footer?: ReactNode; wide?: boolean; icon?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        props.onClose();
      }
      if (e.key === "Tab" && ref.current) {
        const f = ref.current.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])');
        if (!f.length) return;
        const first = f[0];
        const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", key, true);
    setTimeout(() => (ref.current?.querySelector<HTMLElement>(".dialog-body input, .dialog-body select") ?? ref.current)?.focus(), 30);
    return () => {
      window.removeEventListener("keydown", key, true);
      prev?.focus?.();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return createPortal(
    <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div ref={ref} className={`dialog${props.wide ? " wide" : ""}`} role="dialog" aria-modal="true" tabIndex={-1} style={{ outline: "none" }}>
        <div className="dialog-head">
          {props.icon}
          <h2>{props.title}</h2>
          <span className="spacer" />
          <button className="icon-btn sm" onClick={props.onClose} aria-label={t("Close")} data-tip={t("Close (Esc)")} data-tip-pos="left">
            <X size={16} />
          </button>
        </div>
        {props.children}
        {props.footer && <div className="dialog-foot">{props.footer}</div>}
      </div>
    </div>,
    document.body,
  );
}

// ---------------------------------------------------------------------------------------------
const SWATCHES = ["#FFFFFF", "#000000", "#F3F4F6", "#111827"];

export function ColorField(props: { value: [number, number, number, number]; onChange: (v: [number, number, number, number]) => void; allowTransparent?: boolean }) {
  const transparent = props.value[3] === 0;
  const hex = rgbToHex(props.value);
  return (
    <div className="color-field">
      {props.allowTransparent && (
        <button className={`sw transparent${transparent ? " on" : ""}`} data-tip={t("Transparent")} aria-label={t("Transparent")} onClick={() => props.onChange([0, 0, 0, 0])} />
      )}
      {SWATCHES.map((s) => (
        <button
          key={s}
          className={`sw${!transparent && hex === s ? " on" : ""}`}
          style={{ background: s }}
          aria-label={s}
          data-tip={s}
          onClick={() => props.onChange(hexToRgba(s))}
        />
      ))}
      <input type="color" aria-label={t("Custom color")} value={transparent ? "#ffffff" : hex.toLowerCase()} onChange={(e) => props.onChange(hexToRgba(e.target.value))} />
      <span className="faint num" style={{ fontSize: 12 }}>
        {transparent ? t("Transparent") : hex}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((tt) => (
        <div key={tt.id} className={`toast ${tt.kind}`} role={tt.kind === "error" ? "alert" : "status"}>
          {tt.kind === "success" ? <CheckCircle2 size={18} /> : tt.kind === "warning" ? <AlertTriangle size={18} /> : tt.kind === "error" ? <XCircle size={18} /> : <Info size={18} />}
          <div style={{ minWidth: 0 }}>
            <div className="t-title">{tt.title}</div>
            {tt.body && <div className="t-body">{tt.body}</div>}
            {tt.action && (
              <div className="t-actions">
                <button
                  className="btn sm"
                  onClick={() => {
                    tt.action!.run();
                    dismissToast(tt.id);
                  }}
                >
                  {tt.action.label}
                </button>
              </div>
            )}
          </div>
          <button className="icon-btn sm" aria-label={t("Dismiss")} onClick={() => dismissToast(tt.id)}>
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <pattern id="lg-ck" width="8" height="8" patternUnits="userSpaceOnUse">
          <rect width="8" height="8" fill="#e9ebf0" />
          <rect width="4" height="4" fill="#c4c9d4" />
          <rect x="4" y="4" width="4" height="4" fill="#c4c9d4" />
        </pattern>
        <clipPath id="lg-a">
          <path d="M32 6 L58 56 H44 L32 31 L20 56 H6 Z" />
        </clipPath>
      </defs>
      <rect width="64" height="64" rx="14" fill="#1b1d23" />
      <g clipPath="url(#lg-a)">
        <rect x="0" y="0" width="32" height="64" fill="#5b8cff" />
        <rect x="32" y="0" width="32" height="64" fill="url(#lg-ck)" />
      </g>
    </svg>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}
