export function fmtBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10 * 1024 ? 1 : 0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(n < 100 * 1024 * 1024 ? 1 : 0)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function fmtDims(w?: number | null, h?: number | null): string {
  if (!w || !h) return "—";
  return `${w} × ${h}`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}:${String(s).padStart(2, "0")} min`;
}

/** Signed size change, e.g. "−82%" or "+12%". */
export function fmtSaved(before: number, after: number): { text: string; better: boolean } {
  if (!before || !after) return { text: "—", better: true };
  const pct = Math.round((1 - after / before) * 100);
  if (pct >= 0) return { text: `−${pct}%`, better: true };
  return { text: `+${-pct}%`, better: false };
}

export function hexToRgba(hex: string, alpha = 255): [number, number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [255, 255, 255, alpha];
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255, alpha];
}

export function rgbToHex(c: readonly number[]): string {
  return "#" + [c[0], c[1], c[2]].map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("").toUpperCase();
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function basename(p: string): string {
  const i = Math.max(p.lastIndexOf("\\"), p.lastIndexOf("/"));
  return i >= 0 ? p.slice(i + 1) : p;
}

export function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf("\\"), p.lastIndexOf("/"));
  return i >= 0 ? p.slice(0, i) : p;
}

/** Shorten a path for display: "C:\…\Photos\AlphaForge". */
export function shortPath(p: string, max = 42): string {
  if (p.length <= max) return p;
  const parts = p.split(/[\\/]/);
  if (parts.length <= 3) return "…" + p.slice(-max);
  let tail = parts[parts.length - 1];
  for (let i = parts.length - 2; i > 0; i--) {
    const next = parts[i] + "\\" + tail;
    if (parts[0].length + next.length + 3 > max) break;
    tail = next;
  }
  return `${parts[0]}\\…\\${tail}`;
}
