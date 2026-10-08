import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ChevronsLeftRight, Download, FileWarning, RotateCcw, Trash2 } from "lucide-react";
import { imgUrl } from "../api";
import { commitStroke, installModel, removeItems, schedulePreview, setPipeline } from "../actions";
import { S, useStore, viewerBus } from "../store";
import type { Stroke } from "../types";
import { clamp, fmtBytes } from "../lib/format";
import { MaskBar } from "./MaskBar";
import { EmptyState } from "./EmptyState";
import { t, tr } from "../i18n";

interface Shown {
  url: string;
  w: number;
  h: number;
  seq: number;
  partial: boolean;
  id: number;
}

interface View {
  scale: number;
  x: number;
  y: number;
  fit: boolean;
}

const PAD = 28;
const BG_COLORS: Record<string, string> = { white: "#ffffff", black: "#000000", gray: "#808080" };

export function Viewer() {
  const items = useStore((s) => s.items);
  const selectedId = useStore((s) => s.selectedId);
  const item = items.find((i) => i.id === selectedId) ?? null;
  const preview = useStore((s) => s.preview);
  const viewMode = useStore((s) => s.viewMode);
  const mask = useStore((s) => s.mask);
  const bg = useStore((s) => s.settings?.viewerBg ?? "checker");
  const models = useStore((s) => s.models);
  const downloads = useStore((s) => s.downloads);

  const ref = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [box, setBox] = useState({ w: 800, h: 600 });
  const [shown, setShown] = useState<Shown | null>(null);
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0, fit: true });
  const [split, setSplit] = useState(0.5);
  const [panning, setPanning] = useState(false);
  const [space, setSpace] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [live, setLive] = useState<Stroke | null>(null);
  const [origReady, setOrigReady] = useState(false);

  // container size
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setBox({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, [item != null]);

  // reset when the selection changes
  useEffect(() => {
    setShown(null);
    setOrigReady(false);
    setView((v) => ({ ...v, fit: true }));
  }, [selectedId]);

  // load the finished preview off-screen, swap when decoded (no flicker)
  useEffect(() => {
    const d = preview.done;
    if (!d || preview.id !== selectedId || d.seq !== preview.seq) return;
    if (shown && shown.seq === d.seq) return;
    const url = imgUrl("result", d.id, d.seq);
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    let cancelled = false;
    img
      .decode()
      .then(() => {
        if (cancelled) return;
        setShown({ url, w: d.width, h: d.height, seq: d.seq, partial: d.partial, id: d.id });
        const st = S();
        useStore.setState({ preview: { ...st.preview, shownSeq: d.seq }, mask: { ...st.mask, pending: [] } });
        setLive(null);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [preview.done, preview.id, preview.seq, selectedId]); // eslint-disable-line react-hooks/exhaustive-deps

  const ow = item?.info?.width ?? 0;
  const oh = item?.info?.height ?? 0;
  const origUrl = item?.info ? imgUrl("original", item.id, item.size) : null;
  const result = shown && shown.id === selectedId ? shown : null;
  const mode = mask.active ? "result" : viewMode;

  // base geometry of what is on stage
  const geom = useMemo(() => {
    if (mode === "original" || !result) return { w: ow, h: oh, kind: "single" as const };
    if (mode === "result") return { w: result.w, h: result.h, kind: "single" as const };
    // compare: overlay when proportions match, otherwise side by side
    const same = Math.abs(result.w / result.h - ow / oh) < 0.01;
    if (same) return { w: ow, h: oh, kind: "overlay" as const };
    const rs = oh / result.h;
    const gap = Math.round(ow * 0.03);
    return { w: ow + gap + Math.round(result.w * rs), h: oh, kind: "side" as const, rs, gap };
  }, [mode, result, ow, oh]);

  // fit / refit
  const fitView = useCallback((): View => {
    if (!geom.w || !geom.h) return { scale: 1, x: 0, y: 0, fit: true };
    const s = Math.min((box.w - PAD * 2) / geom.w, (box.h - PAD * 2 - (mask.active ? 56 : 0)) / geom.h, 1 / (window.devicePixelRatio || 1) * 4);
    const scale = Math.min(s, 1);
    return { scale, x: (box.w - geom.w * scale) / 2, y: (box.h - geom.h * scale) / 2 - (mask.active ? 22 : 0), fit: true };
  }, [geom.w, geom.h, box.w, box.h, mask.active]);

  useEffect(() => {
    setView((v) => (v.fit ? fitView() : v));
  }, [fitView]);

  // keep geometry stable when only the result size changes (e.g. trim): refit
  const lastGeom = useRef({ w: 0, h: 0 });
  useEffect(() => {
    if (lastGeom.current.w !== geom.w || lastGeom.current.h !== geom.h) {
      lastGeom.current = { w: geom.w, h: geom.h };
      setView(fitView());
    }
  }, [geom.w, geom.h, fitView]);

  // zoom % in the top bar
  useEffect(() => {
    useStore.setState({ zoomPct: view.scale * (window.devicePixelRatio || 1) * 100 });
  }, [view.scale]);

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setView((v) => {
      const scale = clamp(v.scale * factor, 0.02, 40);
      const k = scale / v.scale;
      return { scale, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k, fit: false };
    });
  }, []);

  // commands from top bar / keyboard
  useEffect(() => {
    const zin = () => zoomAt(1.25, box.w / 2, box.h / 2);
    const zout = () => zoomAt(0.8, box.w / 2, box.h / 2);
    const fit = () => setView(fitView());
    const actual = () =>
      setView((v) => {
        const scale = 1 / (window.devicePixelRatio || 1);
        const cx = box.w / 2;
        const cy = box.h / 2;
        const k = scale / v.scale;
        return { scale, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k, fit: false };
      });
    viewerBus.addEventListener("zoom-in", zin);
    viewerBus.addEventListener("zoom-out", zout);
    viewerBus.addEventListener("fit", fit);
    viewerBus.addEventListener("actual", actual);
    return () => {
      viewerBus.removeEventListener("zoom-in", zin);
      viewerBus.removeEventListener("zoom-out", zout);
      viewerBus.removeEventListener("fit", fit);
      viewerBus.removeEventListener("actual", actual);
    };
  }, [zoomAt, fitView, box.w, box.h]);

  // wheel zoom (non-passive)
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      if (e.ctrlKey || Math.abs(e.deltaY) >= Math.abs(e.deltaX)) {
        zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
      } else {
        setView((v) => ({ ...v, x: v.x - e.deltaX, y: v.y - e.deltaY, fit: false }));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt, item != null]);

  // space = temporary pan
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      // Space still activates focused controls (buttons, switches, lists) for keyboard users.
      const interactive = e.target instanceof Element && e.target.closest('button, a, input, textarea, select, [role="slider"], [role="switch"], [role="menuitem"], [role="option"], [role="tab"], [contenteditable="true"]');
      if (e.code === "Space" && !interactive) {
        e.preventDefault();
        setSpace(true);
      }
    };
    const up = (e: KeyboardEvent) => e.code === "Space" && setSpace(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const toImage = (cx: number, cy: number): [number, number] => [(cx - view.x) / view.scale, (cy - view.y) / view.scale];

  const onPointerDown = (e: React.PointerEvent) => {
    const el = ref.current!;
    const r = el.getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    const brush = mask.active && e.button === 0 && !space && result;
    if (brush) {
      el.setPointerCapture(e.pointerId);
      const st: Stroke = { mode: mask.tool, radius: mask.size / 2, hardness: mask.hardness, points: [toImage(cx, cy)] };
      setLive(st);
      const move = (ev: PointerEvent) => {
        const p = toImage(ev.clientX - r.left, ev.clientY - r.top);
        const last = st.points[st.points.length - 1];
        if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= Math.max(1, st.radius / 4)) {
          st.points.push(p);
          setLive({ ...st, points: st.points.slice() });
        }
        setCursor({ x: ev.clientX - r.left, y: ev.clientY - r.top });
      };
      const up = () => {
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
        const clipped = { ...st, points: st.points.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10] as [number, number]) };
        commitStroke(clipped);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
      return;
    }
    if (e.button === 0 || e.button === 1) {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      setPanning(true);
      const sx = e.clientX;
      const sy = e.clientY;
      const start = view;
      const move = (ev: PointerEvent) => setView({ ...start, x: start.x + ev.clientX - sx, y: start.y + ev.clientY - sy, fit: false });
      const up = () => {
        setPanning(false);
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
    }
  };

  // compare handle drag
  const onSplitDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    const el = ref.current!;
    const r = el.getBoundingClientRect();
    const move = (ev: PointerEvent) => setSplit(clamp((ev.clientX - r.left - view.x) / (geom.w * view.scale), 0, 1));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // brush overlay drawing (pending strokes + live stroke) in screen space
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(box.w * dpr);
    c.height = Math.round(box.h * dpr);
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, box.w, box.h);
    if (!mask.active) return;
    const strokes = [...mask.pending, ...(live ? [live] : [])];
    for (const st of strokes) {
      g.fillStyle = st.mode === "keep" ? "rgba(63,207,142,0.38)" : st.mode === "erase" ? "rgba(242,87,92,0.38)" : "rgba(91,140,255,0.38)";
      g.strokeStyle = g.fillStyle;
      g.lineCap = "round";
      g.lineJoin = "round";
      g.lineWidth = st.radius * 2 * view.scale;
      g.beginPath();
      st.points.forEach(([x, y], i) => {
        const sx = view.x + x * view.scale;
        const sy = view.y + y * view.scale;
        if (i === 0) g.moveTo(sx, sy);
        else g.lineTo(sx, sy);
      });
      if (st.points.length === 1) {
        const [x, y] = st.points[0];
        g.beginPath();
        g.arc(view.x + x * view.scale, view.y + y * view.scale, st.radius * view.scale, 0, Math.PI * 2);
        g.fill();
      } else g.stroke();
    }
  }, [mask.active, mask.pending, live, view, box]);

  if (!items.length) return <EmptyState />;
  if (!item) return <div className="viewer" ref={ref} />;

  if (!item.info) {
    return (
      <div className="viewer" ref={ref}>
        {item.status === "error" ? (
          <div className="viewer-error">
            <h3>
              <FileWarning size={18} color="var(--danger)" /> {t("Can’t open this file")}
            </h3>
            <p>{tr(item.error)}</p>
            <div className="row">
              <button className="btn" onClick={() => removeItems([item.id])}>
                <Trash2 size={14} /> {t("Remove from list")}
              </button>
            </div>
          </div>
        ) : (
          <div className="viewer-overlay">
            <span className="spinner" /> {t("Loading {name}…", { name: item.name })}
          </div>
        )}
      </div>
    );
  }

  const showBackdrop = mode !== "original" || item.info.hasAlpha;
  const bgStyle: React.CSSProperties = bg === "checker" ? {} : { background: BG_COLORS[bg] ?? bg };
  const pixel = view.scale * (window.devicePixelRatio || 1) >= 2;
  const dispW = geom.w * view.scale;
  const dispH = geom.h * view.scale;
  const running = preview.status === "running" && preview.id === item.id;
  const err = preview.status === "error" && preview.id === item.id ? preview.error : null;
  const missingSpec = err?.missingModel ? models.find((m) => m.id === err.missingModel) : null;
  const dl = err?.missingModel ? downloads[err.missingModel] : undefined;
  const stepCount = useStore.getState().pipeline.steps.filter((s) => s.enabled).length;

  const cls = ["viewer", mask.active && !space ? "brush" : panning ? "panning" : "pan"].join(" ");

  return (
    <div
      className={cls}
      ref={ref}
      onPointerDown={onPointerDown}
      onPointerMove={(e) => {
        if (!mask.active) return;
        const r = ref.current!.getBoundingClientRect();
        setCursor({ x: e.clientX - r.left, y: e.clientY - r.top });
      }}
      onPointerLeave={() => setCursor(null)}
      onDoubleClick={() => setView(fitView())}
    >
      {showBackdrop && (
        <div
          className={`backdrop${bg === "checker" ? " checker" : ""}`}
          style={{ ...bgStyle, transform: `translate(${view.x}px, ${view.y}px)`, width: dispW, height: dispH }}
        />
      )}
      <div className={`stage${pixel ? " pixelated" : ""}`} style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`, width: geom.w, height: geom.h }}>
        {mode === "original" && origUrl && <img src={origUrl} width={ow} height={oh} alt="" onLoad={() => setOrigReady(true)} draggable={false} />}
        {mode === "result" && !result && origUrl && <img src={origUrl} width={ow} height={oh} alt="" style={{ opacity: 0.45, filter: "saturate(0.6)" }} draggable={false} />}
        {mode === "result" && result && (
          <>
            {mask.active && mask.showOriginal && origUrl && result.w === ow && result.h === oh && (
              <img src={origUrl} width={ow} height={oh} alt="" style={{ opacity: 0.35 }} draggable={false} />
            )}
            <img src={result.url} width={result.w} height={result.h} alt={t("Result")} draggable={false} />
          </>
        )}
        {mode === "compare" && result && origUrl && geom.kind === "overlay" && (
          <>
            <img src={origUrl} width={ow} height={oh} alt={t("Original")} draggable={false} style={{ clipPath: `inset(0 ${(1 - split) * 100}% 0 0)` }} />
            <img src={result.url} width={ow} height={oh} alt={t("Result")} draggable={false} style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }} />
          </>
        )}
        {mode === "compare" && result && origUrl && geom.kind === "side" && (
          <>
            <img src={origUrl} width={ow} height={oh} alt={t("Original")} draggable={false} />
            <img
              src={result.url}
              alt={t("Result")}
              draggable={false}
              style={{ left: ow + (geom.gap ?? 0), width: Math.round(result.w * (geom.rs ?? 1)), height: oh }}
            />
          </>
        )}
        {mode === "compare" && !result && origUrl && <img src={origUrl} width={ow} height={oh} alt="" draggable={false} />}
      </div>
      {mode === "compare" && result && geom.kind === "overlay" && (
        <>
          <div className="compare-handle" style={{ left: view.x + dispW * split, top: Math.max(0, view.y), bottom: Math.max(0, box.h - view.y - dispH) }} onPointerDown={onSplitDown}>
            <span className="grip">
              <ChevronsLeftRight size={16} />
            </span>
          </div>
          <span className="compare-label" style={{ left: Math.max(8, view.x + 8) }}>
            {t("Original")}
          </span>
          <span className="compare-label" style={{ right: Math.max(8, box.w - view.x - dispW + 8) }}>
            {t("Result")}
          </span>
        </>
      )}
      {mode === "compare" && result && geom.kind === "side" && (
        <>
          <span className="compare-label" style={{ left: Math.max(8, view.x + 8) }}>
            {t("Original")}
          </span>
          <span className="compare-label" style={{ left: view.x + (ow + (geom.gap ?? 0)) * view.scale + 8 }}>
            {t("Result")}
          </span>
        </>
      )}

      <canvas ref={canvasRef} className="stroke-canvas" style={{ width: box.w, height: box.h }} />
      {mask.active && cursor && !space && (
        <div className={`brush-cursor ${mask.tool}`} style={{ left: cursor.x, top: cursor.y, width: mask.size * view.scale, height: mask.size * view.scale }} />
      )}

      {running && !err && (
        <div className="viewer-overlay" role="status">
          <span className="spinner" />
          <span>
            {preview.label || t("Processing")}
            {stepCount > 1 ? <span className="faint"> · {t("step {a}/{b}", { a: Math.min(preview.step + 1, stepCount), b: stepCount })}</span> : null}
          </span>
          <div className="progress">
            <div style={{ width: `${Math.round(((preview.step + preview.fraction) / Math.max(1, stepCount)) * 100)}%` }} />
          </div>
        </div>
      )}
      {!origReady && mode === "original" && <div className="viewer-overlay"><span className="spinner" /> {t("Loading…")}</div>}

      {err && (
        <div className="viewer-error" onPointerDown={(e) => e.stopPropagation()}>
          {missingSpec ? (
            <>
              <h3>
                <Download size={18} color="var(--accent)" /> {t("{name} model needed", { name: t(missingSpec.name) })}
              </h3>
              <p>
                {t("This step uses {family}. It is a one-time {size} download from the model author’s official repository, verified with a checksum. Images are never uploaded.", { family: missingSpec.family, size: fmtBytes(missingSpec.downloadBytes) })}
              </p>
              {dl ? (
                <div className="col">
                  <div className="progress">
                    <div style={{ width: `${(dl.done / Math.max(1, dl.total)) * 100}%` }} />
                  </div>
                  <span className="faint num" style={{ fontSize: 12 }}>
                    {t("{a} of {b}", { a: fmtBytes(dl.done), b: fmtBytes(dl.total) })}
                  </span>
                </div>
              ) : (
                <div className="row">
                  <button className="btn primary" onClick={() => installModel(missingSpec.id)}>
                    <Download size={14} /> {t("Download {size}", { size: fmtBytes(missingSpec.downloadBytes) })}
                  </button>
                  {missingSpec.kind === "depth" ? (
                    <button
                      className="btn"
                      onClick={() => {
                        const p = S().pipeline;
                        setPipeline({ ...p, steps: p.steps.map((s) => (s.type === "background" ? { ...s, depth: false } : s)) });
                      }}
                    >
                      {t("Blur without AI")}
                    </button>
                  ) : (
                    <button
                      className="btn"
                      onClick={() => {
                        const p = S().pipeline;
                        setPipeline({ ...p, steps: p.steps.map((s) => (s.type === "removeBackground" ? { ...s, model: "fast" } : s)) });
                      }}
                    >
                      {t("Use Fast model instead")}
                    </button>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <h3>
                <AlertTriangle size={18} color="var(--warning)" /> {t("Processing failed")}
              </h3>
              <p>{tr(err.message)}</p>
              <div className="row">
                <button className="btn" onClick={() => schedulePreview(0, true)}>
                  <RotateCcw size={14} /> {t("Try again")}
                </button>
              </div>
            </>
          )}
        </div>
      )}
      {mask.active && <MaskBar />}
    </div>
  );
}
