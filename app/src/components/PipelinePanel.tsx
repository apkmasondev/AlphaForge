import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  ChevronDown,
  Copy,
  Crop,
  GripVertical,
  Maximize2,
  MoreHorizontal,
  PaintBucket,
  Plus,
  Save,
  Scaling,
  Scissors,
  SlidersHorizontal,
  Sparkles,
  SquareDashed,
  Trash2,
  Undo2,
  Wand2,
} from "lucide-react";
import { api } from "../api";
import { applyPreset, deletePreset, presetModified, setPipeline } from "../actions";
import { insertIndex, newStep, outputSummary, STEP_ORDER, stepHint, stepLabel, stepSummary } from "../pipeline";
import { S, useStore } from "../store";
import type { OutFormat, PngLevel, StepEntry, StepType } from "../types";
import { fmtBytes, fmtSaved } from "../lib/format";
import { CheckRow, ColorField, Menu, Segmented, Slider, Switch, useMenu } from "./ui";
import { BackgroundEditor, EnhanceEditor, PaddingEditor, RemoveBackgroundEditor, ResizeEditor, TrimEditor, UpscaleEditor } from "./steps";
import { t, tr } from "../i18n";

const ICON: Record<StepType, React.ReactNode> = {
  removeBackground: <Scissors size={15} />,
  trim: <Crop size={15} />,
  padding: <SquareDashed size={15} />,
  resize: <Scaling size={15} />,
  upscale: <Maximize2 size={15} />,
  enhance: <Wand2 size={15} />,
  background: <PaintBucket size={15} />,
};
const AI_STEP: Partial<Record<StepType, boolean>> = { removeBackground: true, upscale: true };

function updateStep(id: string, patch: Partial<StepEntry>) {
  const p = S().pipeline;
  setPipeline({ ...p, steps: p.steps.map((s) => (s.id === id ? ({ ...s, ...patch } as StepEntry) : s)) });
}

function moveStep(id: string, to: number) {
  const p = S().pipeline;
  const from = p.steps.findIndex((s) => s.id === id);
  if (from < 0) return;
  const steps = p.steps.slice();
  const [st] = steps.splice(from, 1);
  steps.splice(Math.max(0, Math.min(steps.length, to)), 0, st);
  setPipeline({ ...p, steps });
}

function PresetBar() {
  const presets = useStore((s) => s.presets);
  const presetId = useStore((s) => s.presetId);
  useStore((s) => s.pipeline);
  const set = useStore((s) => s.set);
  const menu = useMenu();
  const current = presets.find((p) => p.id === presetId);
  const modified = presetModified();
  const builtins = presets.filter((p) => p.builtin);
  const mine = presets.filter((p) => !p.builtin);
  return (
    <div className="preset-bar">
      <button className="preset-btn" onClick={(e) => menu.open(e.currentTarget)} aria-haspopup="menu" data-tip={t("Presets")} data-tip-pos="right">
        <SlidersHorizontal size={15} className="faint" />
        <span className="p-name ellipsis">{current ? t(current.name) : t("Custom pipeline")}</span>
        {modified && <span className="modified">{t("Modified")}</span>}
        <ChevronDown size={15} className="faint" />
      </button>
      <button className="icon-btn" aria-label={t("Save as preset")} data-tip={t("Save as preset")} data-tip-pos="left" onClick={() => set({ savePresetOpen: true })}>
        <Save size={16} />
      </button>
      {menu.anchor && (
        <Menu
          anchor={menu.anchor}
          onClose={menu.close}
          width={300}
          items={[
            ...(modified && current
              ? [{ label: t("Reset to “{name}”", { name: t(current.name) }), icon: <Undo2 size={15} />, onClick: () => applyPreset(current) }, { sep: true }]
              : []),
            { title: t("Presets") },
            ...builtins.map((p) => ({ label: <PresetLabel name={t(p.name)} desc={t(p.description)} active={p.id === presetId} />, onClick: () => applyPreset(p) })),
            ...(mine.length
              ? [
                  { sep: true },
                  { title: t("My presets") },
                  ...mine.map((p) => ({ label: <PresetLabel name={p.name} desc={outputSummary(p.pipeline.output)} active={p.id === presetId} />, onClick: () => applyPreset(p) })),
                ]
              : []),
            { sep: true },
            { label: t("Save current as preset…"), icon: <Save size={15} />, onClick: () => set({ savePresetOpen: true }) },
            ...(current && !current.builtin ? [{ label: t("Delete “{name}”", { name: current.name }), icon: <Trash2 size={15} />, danger: true, onClick: () => deletePreset(current.id) }] : []),
          ]}
        />
      )}
    </div>
  );
}

function PresetLabel({ name, desc, active }: { name: string; desc: string; active: boolean }) {
  return (
    <span style={{ display: "flex", flexDirection: "column", gap: 1, padding: "2px 0" }}>
      <span style={{ fontWeight: active ? 600 : 500, color: active ? "var(--accent-text)" : undefined }}>{name}</span>
      <span style={{ fontSize: 12, color: "var(--text-3)" }}>{desc}</span>
    </span>
  );
}

function StepCard({ step, index, total, open, toggle, onGrip, dragState }: {
  step: StepEntry;
  index: number;
  total: number;
  open: boolean;
  toggle: () => void;
  onGrip: (e: React.PointerEvent, id: string) => void;
  dragState: { id: string | null; over: number | null };
}) {
  const menu = useMenu();
  const preview = useStore((s) => s.preview);
  const activeIndex = useStore((s) => s.pipeline.steps.filter((x) => x.enabled).findIndex((x) => x.id === step.id));
  const running = preview.status === "running" && step.enabled && activeIndex === preview.step;
  const upd = (patch: Partial<StepEntry>) => updateStep(step.id, patch);
  const cls = ["step", !step.enabled && "disabled", running && "running", dragState.id === step.id && "dragging",
    dragState.over === index && dragState.id !== step.id && "drop-before", dragState.over === total && index === total - 1 && dragState.id !== step.id && "drop-after"]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={cls} data-step-index={index}>
      <div className="step-head" onClick={toggle} role="button" aria-expanded={open} tabIndex={0} onKeyDown={(e) => e.target === e.currentTarget && (e.key === "Enter" || e.key === " ") && (e.preventDefault(), toggle())}>
        <span className="step-grip" onPointerDown={(e) => onGrip(e, step.id)} onClick={(e) => e.stopPropagation()} aria-label={t("Drag to reorder")} data-tip={t("Drag to reorder")}>
          <GripVertical size={14} />
        </span>
        <span className={`step-icon${AI_STEP[step.type] || (step.type === "enhance" && step.denoise > 0) ? " ai" : ""}`}>{ICON[step.type]}</span>
        <span style={{ minWidth: 0 }}>
          <div className="step-title">
            {stepLabel(step.type)}
            {AI_STEP[step.type] && <span className="badge accent" style={{ height: 16, fontSize: 10 }}>AI</span>}
          </div>
          <div className="step-sum ellipsis">{stepSummary(step)}</div>
        </span>
        <span className="step-right" onClick={(e) => e.stopPropagation()}>
          <Switch checked={step.enabled} onChange={(enabled) => upd({ enabled })} label={t("Enable {s}", { s: stepLabel(step.type) })} />
          <button className="icon-btn sm" aria-label={t("Step options")} onClick={(e) => menu.open(e.currentTarget)}>
            <MoreHorizontal size={15} />
          </button>
        </span>
      </div>
      {running && (
        <div className="step-progress">
          <div style={{ width: `${Math.round(preview.fraction * 100)}%` }} />
        </div>
      )}
      {open && (
        <div className="step-body">
          {step.type === "removeBackground" && <RemoveBackgroundEditor step={step} upd={upd} />}
          {step.type === "trim" && <TrimEditor step={step} upd={upd} />}
          {step.type === "padding" && <PaddingEditor step={step} upd={upd} />}
          {step.type === "resize" && <ResizeEditor step={step} upd={upd} />}
          {step.type === "upscale" && <UpscaleEditor step={step} upd={upd} />}
          {step.type === "enhance" && <EnhanceEditor step={step} upd={upd} />}
          {step.type === "background" && <BackgroundEditor step={step} upd={upd} />}
        </div>
      )}
      {menu.anchor && (
        <Menu
          anchor={menu.anchor}
          onClose={menu.close}
          align="right"
          items={[
            { label: t("Move up"), icon: <ArrowUp size={15} />, disabled: index === 0, onClick: () => moveStep(step.id, index - 1) },
            { label: t("Move down"), icon: <ArrowDown size={15} />, disabled: index === total - 1, onClick: () => moveStep(step.id, index + 1) },
            {
              label: t("Duplicate"),
              icon: <Copy size={15} />,
              onClick: () => {
                const p = S().pipeline;
                const copy = { ...JSON.parse(JSON.stringify(step)), id: newStep(step.type).id };
                const steps = p.steps.slice();
                steps.splice(index + 1, 0, copy);
                setPipeline({ ...p, steps });
              },
            },
            { sep: true },
            {
              label: t("Remove step"),
              icon: <Trash2 size={15} />,
              danger: true,
              onClick: () => {
                const p = S().pipeline;
                setPipeline({ ...p, steps: p.steps.filter((s) => s.id !== step.id) });
              },
            },
          ]}
        />
      )}
    </div>
  );
}

function OutputSection() {
  const out = useStore((s) => s.pipeline.output);
  const preview = useStore((s) => s.preview);
  const item = useStore((s) => s.items.find((i) => i.id === s.selectedId) ?? null);
  const set = (patch: Partial<typeof out>) => {
    const p = S().pipeline;
    setPipeline({ ...p, output: { ...p.output, ...patch } });
  };
  const refining = useStore((s) => s.mask.active);
  const sizeKnown = !refining && preview.size && preview.id === item?.id && preview.size.seq === preview.seq;
  const saved = sizeKnown && item ? fmtSaved(item.size, preview.size!.bytes) : null;
  const lossy = out.format === "jpeg" || out.format === "avif" || (out.format === "webp" && !out.lossless) || out.format === "same";
  return (
    <div className="output">
      <div className="output-head">
        <span className="panel-title">{t("Output")}</span>
        <span className="faint" style={{ fontSize: 12 }}>{outputSummary(out)}</span>
      </div>
      <Segmented<OutFormat>
        full
        ariaLabel={t("Format")}
        value={out.format}
        onChange={(format) => set({ format })}
        options={[
          { value: "png", label: "PNG", title: t("Lossless, transparency") },
          { value: "webp", label: "WebP", title: t("Small, transparency") },
          { value: "avif", label: "AVIF", title: t("Smallest, transparency") },
          { value: "jpeg", label: "JPG", title: t("Universal, no transparency") },
          { value: "same", label: t("Same"), title: t("Keep the source format") },
        ]}
      />
      {lossy && (
        <div className="field-row">
          <span className="field-label">{t("Quality")}</span>
          <Slider value={out.quality} min={1} max={100} onChange={(quality) => set({ quality })} ariaLabel={t("Quality")} />
        </div>
      )}
      {out.format === "png" && (
        <>
          <div className="field-row">
            <span className="field-label">{t("Compression")}</span>
            <Segmented<PngLevel>
              full
              value={out.pngLevel}
              onChange={(pngLevel) => set({ pngLevel })}
              options={[
                { value: "fast", label: t("Fast") },
                { value: "balanced", label: t("Balanced") },
                { value: "max", label: t("Max") },
              ]}
            />
          </div>
          <CheckRow checked={out.pngColors != null} onChange={(v) => set({ pngColors: v ? 256 : null })}>
            {t("Reduce colors (smaller, lossy)")}
          </CheckRow>
          {out.pngColors != null && (
            <div className="field-row">
              <span className="field-label">{t("Colors")}</span>
              <Segmented<number>
                full
                value={out.pngColors}
                onChange={(pngColors) => set({ pngColors })}
                options={[256, 128, 64, 32].map((v) => ({ value: v, label: String(v) }))}
              />
            </div>
          )}
        </>
      )}
      {out.format === "webp" && (
        <CheckRow checked={out.lossless} onChange={(lossless) => set({ lossless })}>
          {t("Lossless")}
        </CheckRow>
      )}
      {(out.format === "jpeg" || out.format === "same") && (
        <div className="field-row">
          <span className="field-label">{t("Transparency")}</span>
          <ColorField value={[...out.background, 255] as [number, number, number, number]} onChange={(c) => set({ background: [c[0], c[1], c[2]] })} />
        </div>
      )}
      {item && (
        <div className="estimate" aria-live="polite">
          <div className="e-col">
            <span className="e-k">{t("Original size")}</span>
            <span className="e-v">{fmtBytes(item.size)}</span>
          </div>
          {saved ? <span className={`e-pct ${saved.better ? "better" : "worse"}`}>{saved.text}</span> : <span className="spinner" style={{ opacity: !refining && (preview.status === "running" || (preview.done && !sizeKnown)) ? 1 : 0 }} />}
          <div className="e-col">
            <span className="e-k">{t("Output size")} {sizeKnown ? preview.size!.format : ""}</span>
            <span className="e-v">{sizeKnown ? fmtBytes(preview.size!.bytes) : refining ? t("after refining") : "…"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export function PipelinePanel() {
  const steps = useStore((s) => s.pipeline.steps);
  const pipeline = useStore((s) => s.pipeline);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [warnings, setWarnings] = useState<string[]>([]);
  const [drag, setDrag] = useState<{ id: string | null; over: number | null }>({ id: null, over: null });
  const listRef = useRef<HTMLDivElement>(null);
  const addMenu = useMenu();

  // default expansion: first two steps
  useEffect(() => {
    setOpen((o) => {
      const n = { ...o };
      steps.forEach((s, i) => {
        if (n[s.id] === undefined) n[s.id] = i < 2;
      });
      return n;
    });
  }, [steps]);

  useEffect(() => {
    const t = setTimeout(() => api.pipelineWarnings(pipeline).then(setWarnings).catch(() => {}), 250);
    return () => clearTimeout(t);
  }, [pipeline]);

  const onGrip = (e: React.PointerEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    const list = listRef.current;
    if (!list) return;
    setDrag({ id, over: null });
    const move = (ev: PointerEvent) => {
      const cards = Array.from(list.querySelectorAll<HTMLElement>("[data-step-index]"));
      let over = cards.length;
      for (const c of cards) {
        const r = c.getBoundingClientRect();
        if (ev.clientY < r.top + r.height / 2) {
          over = Number(c.dataset.stepIndex);
          break;
        }
      }
      setDrag({ id, over });
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setDrag((d) => {
        if (d.id && d.over != null) {
          const from = S().pipeline.steps.findIndex((s) => s.id === d.id);
          const to = d.over > from ? d.over - 1 : d.over;
          if (to !== from) moveStep(d.id, to);
        }
        return { id: null, over: null };
      });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const addStep = (t: StepType) => {
    const p = S().pipeline;
    const st = newStep(t);
    const steps = p.steps.slice();
    steps.splice(insertIndex(steps, t), 0, st);
    setOpen((o) => ({ ...o, [st.id]: true }));
    setPipeline({ ...p, steps });
  };

  return (
    <aside className="right" aria-label={t("Pipeline")}>
      <PresetBar />
      <div className="pipeline" ref={listRef}>
        <div className="row" style={{ justifyContent: "space-between", padding: "0 2px 2px" }}>
          <span className="panel-title">{t("Pipeline")}</span>
          <span className="faint" style={{ fontSize: 12 }}>{t("{a} of {b} steps on", { a: steps.filter((s) => s.enabled).length, b: steps.length })}</span>
        </div>
        {steps.map((s, i) => (
          <StepCard key={s.id} step={s} index={i} total={steps.length} open={!!open[s.id]} toggle={() => setOpen((o) => ({ ...o, [s.id]: !o[s.id] }))} onGrip={onGrip} dragState={drag} />
        ))}
        {!steps.length && (
          <div className="notice info">
            <Sparkles size={15} />
            <span>{t("No steps — images are only converted and compressed with the output settings below. Add steps to remove backgrounds, resize or upscale.")}</span>
          </div>
        )}
        <button className="add-step" onClick={(e) => addMenu.open(e.currentTarget)}>
          <Plus size={15} /> {t("Add step")}
        </button>
        {addMenu.anchor && (
          <Menu
            anchor={addMenu.anchor}
            onClose={addMenu.close}
            width={300}
            items={STEP_ORDER.map((st) => ({
              label: <PresetLabel name={stepLabel(st)} desc={stepHint(st)} active={false} />,
              icon: ICON[st],
              onClick: () => addStep(st),
            }))}
          />
        )}
        {warnings.map((w, i) => (
          <div key={i} className="notice warning">
            <AlertTriangle size={15} />
            <span>{tr(w)}</span>
          </div>
        ))}
      </div>
      <OutputSection />
    </aside>
  );
}
