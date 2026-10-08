import { getCurrentWindow } from "@tauri-apps/api/window";
import { api, on, type Events } from "./api";
import { clonePipeline, normalizePipeline, samePipeline } from "./pipeline";
import { initialPreview, removedIds, S, toast, upsertItem, useStore } from "./store";
import type { Item, Pipeline, Preset, Settings, Stroke } from "./types";
import { fmtBytes, fmtMs, fmtSaved } from "./lib/format";
import { resolveLang, setLang, t, tr } from "./i18n";

// ---------------------------------------------------------------------------------------------
// start-up
// ---------------------------------------------------------------------------------------------

export async function bootstrap() {
  const b = await api.bootstrap();
  setLang(resolveLang(b.settings.language));
  const presets = b.presets.map((p) => ({ ...p, pipeline: normalizePipeline(p.pipeline) }));
  const preset = presets.find((p) => p.id === b.settings.presetId) ?? presets[0];
  const pipeline = b.settings.pipeline ? normalizePipeline(b.settings.pipeline) : clonePipeline(preset.pipeline);
  useStore.setState({
    ready: true,
    version: b.version,
    system: b.system,
    models: b.models,
    gpuPack: b.gpuPack,
    items: b.items,
    selectedId: b.items[0]?.id ?? null,
    presets,
    presetId: b.settings.presetId ?? preset.id,
    pipeline,
    settings: b.settings,
  });
  applyTheme(b.settings.theme);
  await wireEvents();
  api.runtimeInfo().then((r) => {
    useStore.setState({ runtime: r });
    maybeSuggestGpu();
  });
  await api.frontendReady();
  schedulePreview(0);
}

function maybeSuggestGpu() {
  const s = S();
  const rt = s.runtime?.info;
  if (!rt || !s.settings || s.settings.gpuPromptDismissed) return;
  if (rt.gpuSupported && !rt.gpuPackInstalled && rt.gpu) {
    toast({
      kind: "info",
      title: t("{gpu} detected", { gpu: rt.gpu.name.replace(/^NVIDIA\s+/, "") }),
      body: t("Turn on GPU acceleration for ~10× faster AI processing (one-time {size} download).", { size: fmtBytes(s.gpuPack?.downloadBytes ?? 0) }),
      action: { label: t("Set up"), run: () => useStore.setState({ settingsOpen: "ai" }) },
      timeout: 15000,
    });
    updateSettings({ gpuPromptDismissed: true });
  }
}

export function applyTheme(theme: Settings["theme"]) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const resolved = theme === "system" ? (mq.matches ? "dark" : "light") : theme;
  document.documentElement.dataset.theme = resolved;
  getCurrentWindow().setTheme(theme === "system" ? null : theme).catch(() => {});
}

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  const t = S().settings?.theme;
  if (t === "system") applyTheme(t);
});

async function wireEvents() {
  await on("item-updated", (it) => {
    upsertItem(it);
    if (S().selectedId == null) useStore.setState({ selectedId: it.id });
    const s = S();
    if (it.id === s.selectedId && it.status === "ready" && (s.preview.status === "idle" || s.preview.id !== it.id)) schedulePreview(0);
  });
  await on("items-changed", async () => {
    const items = await api.listItems();
    useStore.setState({ items });
    if (S().selectedId == null && items[0]) select(items[0].id);
  });
  await onPreview("preview-progress", (p) => {
    const pv = S().preview;
    useStore.setState({ preview: { ...pv, status: "running", step: p.step, fraction: p.fraction, label: p.label ? t(p.label) : pv.label } });
  });
  await onPreview("preview-done", (d) => {
    const pv = S().preview;
    useStore.setState({ preview: { ...pv, status: "done", done: d, error: null, fraction: 1 } });
    const notes = d.reports.map((r) => r.note).filter(Boolean) as string[];
    for (const n of notes) toast({ kind: "warning", title: t("Heads up"), body: tr(n) });
  });
  await onPreview("preview-size", (z) => {
    const pv = S().preview;
    useStore.setState({ preview: { ...pv, size: z } });
  });
  await onPreview("preview-error", (e) => {
    const pv = S().preview;
    useStore.setState({ preview: { ...pv, status: "error", error: e } });
  });
  await on("export-progress", (p) => useStore.setState((s) => ({ exportRun: { ...s.exportRun, running: true, progress: p } })));
  await on("export-done", (sum) => {
    useStore.setState({ exportRun: { running: false, progress: null, summary: sum } });
    if (sum.cancelled) {
      toast({ kind: "info", title: t("Export cancelled"), body: t(sum.ok === 1 ? "{n} image saved before cancelling." : "{n} images saved before cancelling.", { n: sum.ok }) });
      return;
    }
    const saved = fmtSaved(sum.inBytes, sum.outBytes);
    const parts = [`${fmtBytes(sum.inBytes)} → ${fmtBytes(sum.outBytes)} (${saved.text})`, fmtMs(sum.elapsedMs)];
    if (sum.failed) {
      toast({
        kind: "warning",
        title: t("{a} exported, {b} failed", { a: sum.ok, b: sum.failed }),
        body: sum.errors.slice(0, 3).map(([n, e]) => `${n}: ${tr(e)}`).join("\n"),
        action: sum.folder ? { label: t("Open folder"), run: () => api.openFolder(sum.folder!) } : undefined,
      });
    } else {
      toast({
        kind: "success",
        title: t(sum.ok === 1 ? "{n} image exported" : "{n} images exported", { n: sum.ok }) + (sum.skipped ? t(", {n} skipped", { n: sum.skipped }) : ""),
        body: parts.join(" · "),
        action: sum.folder ? { label: t("Open folder"), run: () => api.openFolder(sum.folder!) } : undefined,
      });
    }
  });
  await on("download-progress", (p) => useStore.setState((s) => ({ downloads: { ...s.downloads, [p.key]: p } })));
  await on("download-done", async (d) => {
    useStore.setState((s) => {
      const downloads = { ...s.downloads };
      delete downloads[d.key];
      return { downloads };
    });
    if (d.key === "gpu-pack") {
      useStore.setState({ gpuPack: await api.gpuPackStatus() });
      if (d.ok)
        toast({
          kind: "success",
          title: t("GPU acceleration installed"),
          body: t("Restart AlphaForge to start using the GPU."),
          action: { label: t("Restart now"), run: () => api.restartApp() },
          timeout: 0,
        });
    } else {
      useStore.setState({ models: await api.models() });
      if (d.ok) {
        toast({ kind: "success", title: t("Model ready"), body: t("The model was downloaded and verified.") });
        schedulePreview(0);
      }
    }
    if (d.error) toast({ kind: "error", title: t("Download failed"), body: tr(d.error) });
  });
  await on("runtime-ready", async () => useStore.setState({ runtime: await api.runtimeInfo() }));
  await on("models-unloaded", async () => useStore.setState({ runtime: await api.runtimeInfo() }));
}

// ---------------------------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------------------------

function afterAdd(added: Item[], skipped: number) {
  if (added.length) {
    useStore.setState((s) => ({ items: mergeItems(s.items, added) }));
    if (S().selectedId == null || added.length === 1) select(added[0].id);
  }
  if (skipped > 0) toast({ kind: "info", title: t(skipped === 1 ? "{n} file skipped" : "{n} files skipped", { n: skipped }), body: t("Only JPG, PNG, WebP, AVIF, BMP, GIF and TIFF images are added.") });
  if (!added.length && !skipped) toast({ kind: "info", title: t("Already in the list") });
}

function mergeItems(cur: Item[], add: Item[]): Item[] {
  const ids = new Set(cur.map((i) => i.id));
  return [...cur, ...add.filter((i) => !ids.has(i.id))];
}

export async function addPaths(paths: string[]) {
  try {
    const r = await api.addPaths(paths);
    afterAdd(r.added, r.skipped);
  } catch (e) {
    toast({ kind: "error", title: t("Could not add files"), body: tr(String(e)) });
  }
}

export async function openFiles() {
  const r = await api.openFilesDialog();
  if (r.added.length || r.skipped) afterAdd(r.added, r.skipped);
}

export async function openFolder() {
  const r = await api.openFolderDialog();
  if (r.added.length || r.skipped) afterAdd(r.added, r.skipped);
  else if (r.added.length === 0 && r.skipped === 0) {
    /* dialog cancelled */
  }
}

export async function paste() {
  try {
    const r = await api.pasteClipboard();
    if (r.kind === "none") {
      toast({ kind: "info", title: t("Nothing to paste"), body: t("Copy an image or image files first (e.g. Win + Shift + S for a screenshot).") });
      return;
    }
    const items = await api.listItems();
    useStore.setState({ items });
    if (r.added.length) select(r.added[r.added.length - 1]);
  } catch (e) {
    toast({ kind: "error", title: t("Paste failed"), body: tr(String(e)) });
  }
}

export async function removeItems(ids: number[]) {
  if (!ids.length) return;
  for (const id of ids) removedIds.add(id);
  await api.removeItems(ids);
  for (const id of ids) strokeCache.delete(id);
  const s = S();
  const items = s.items.filter((i) => !ids.includes(i.id));
  let selectedId = s.selectedId;
  if (selectedId != null && ids.includes(selectedId)) {
    const idx = s.items.findIndex((i) => i.id === selectedId);
    selectedId = items[Math.min(idx, items.length - 1)]?.id ?? null;
  }
  useStore.setState({ items, selection: s.selection.filter((i) => !ids.includes(i)), selectedId });
  if (selectedId != null) select(selectedId);
  else useStore.setState({ preview: initialPreview });
}

export async function clearAll() {
  for (const it of S().items) removedIds.add(it.id);
  await api.clearItems();
  strokeCache.clear();
  useStore.setState({ items: [], selection: [], selectedId: null, preview: initialPreview });
  exitMask();
}

export function select(id: number, opts: { additive?: boolean; range?: boolean } = {}) {
  const s = S();
  if (opts.additive) {
    const sel = s.selection.includes(id) ? s.selection.filter((x) => x !== id) : [...s.selection, id];
    useStore.setState({ selection: sel, selectedId: id });
  } else if (opts.range && s.selectedId != null) {
    const a = s.items.findIndex((i) => i.id === s.selectedId);
    const b = s.items.findIndex((i) => i.id === id);
    const [lo, hi] = a < b ? [a, b] : [b, a];
    useStore.setState({ selection: s.items.slice(lo, hi + 1).map((i) => i.id), selectedId: id });
  } else {
    useStore.setState({ selection: [], selectedId: id });
  }
  if (id !== s.selectedId) {
    if (s.mask.active) exitMask();
    useStore.setState({ preview: { ...initialPreview } });
    schedulePreview(0);
  }
}

export function selectRelative(delta: number) {
  const s = S();
  if (!s.items.length) return;
  const idx = s.items.findIndex((i) => i.id === s.selectedId);
  const next = s.items[Math.max(0, Math.min(s.items.length - 1, (idx < 0 ? 0 : idx) + delta))];
  if (next) select(next.id);
}

export async function reprocess(id: number) {
  const it = await api.reprocess(id);
  if (it) upsertItem(it);
  if (id === S().selectedId) schedulePreview(0, true);
}

// ---------------------------------------------------------------------------------------------
// preview
// ---------------------------------------------------------------------------------------------

let previewTimer: number | undefined;

/**
 * Preview events that arrived before `api.preview` returned their seq. Fast pipelines (resize,
 * convert) can finish before the invoke response reaches the UI; dropping those events left the
 * preview stuck on "Processing".
 */
let earlyPreviewEvents: { seq: number; apply: () => void }[] = [];

/** Listen to a preview event, applying it only for the current request (buffering early ones). */
type PreviewEvent = "preview-progress" | "preview-done" | "preview-size" | "preview-error";
function onPreview<K extends PreviewEvent>(name: K, handler: (payload: Events[K]) => void) {
  return on(name, (payload) => {
    const cur = S().preview.seq;
    if (payload.seq === cur) handler(payload);
    else if (payload.seq > cur && earlyPreviewEvents.length < 64) earlyPreviewEvents.push({ seq: payload.seq, apply: () => handler(payload) });
  });
}

/** Re-render the selected image after `delay` ms (debounced). */
export function schedulePreview(delay = 220, force = false) {
  window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(() => runPreview(force), delay);
}

async function runPreview(_force: boolean) {
  const s = S();
  const id = s.selectedId;
  if (id == null || !s.ready) return;
  const item = s.items.find((i) => i.id === id);
  if (!item || item.status === "loading" || (item.status === "error" && !item.info)) return;
  if (!s.settings?.autoPreview && !_force && s.preview.status !== "idle") return;
  const stage = s.mask.active ? "mask" : "final";
  const seq = await api.preview(id, s.pipeline, stage);
  const pv = S().preview;
  useStore.setState({
    preview: { ...pv, id, seq, status: "running", step: 0, fraction: 0, label: t("Processing"), error: null, size: pv.id === id ? pv.size : null, stage },
  });
  const early = earlyPreviewEvents.filter((e) => e.seq === seq);
  earlyPreviewEvents = earlyPreviewEvents.filter((e) => e.seq > seq);
  for (const e of early) e.apply();
}

// ---------------------------------------------------------------------------------------------
// pipeline / presets / settings
// ---------------------------------------------------------------------------------------------

let saveTimer: number | undefined;
export function updateSettings(patch: Partial<Settings>) {
  const cur = S().settings;
  if (!cur) return;
  const next = { ...cur, ...patch };
  useStore.setState({ settings: next });
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => {
    saveTimer = undefined;
    api.saveSettings(S().settings!);
  }, 400);
}

/** Write a pending (debounced) settings change now — the backend reads export options from it. */
async function flushSettings() {
  if (saveTimer === undefined) return;
  window.clearTimeout(saveTimer);
  saveTimer = undefined;
  await api.saveSettings(S().settings!);
}

export function setPipeline(p: Pipeline, opts: { preview?: boolean } = {}) {
  useStore.setState({ pipeline: p });
  updateSettings({ pipeline: p });
  if (opts.preview !== false) schedulePreview();
}

export function applyPreset(preset: Preset) {
  const p = clonePipeline(preset.pipeline);
  useStore.setState({ presetId: preset.id, pipeline: p });
  updateSettings({ presetId: preset.id, pipeline: p });
  if (S().mask.active) exitMask();
  schedulePreview(0);
}

export function presetModified(): boolean {
  const s = S();
  const p = s.presets.find((x) => x.id === s.presetId);
  return !!p && !samePipeline(p.pipeline, s.pipeline);
}

export async function savePreset(name: string, overwriteId: string | null) {
  const presets = (await api.savePreset(overwriteId, name, S().pipeline)).map((p) => ({ ...p, pipeline: normalizePipeline(p.pipeline) }));
  const saved = overwriteId ? presets.find((p) => p.id === overwriteId) : presets.filter((p) => !p.builtin).slice(-1)[0];
  useStore.setState({ presets, presetId: saved?.id ?? S().presetId });
  updateSettings({ presetId: saved?.id ?? null });
  toast({ kind: "success", title: t("Preset “{name}” saved", { name: saved?.name ?? name }) });
}

export async function deletePreset(id: string) {
  const presets = (await api.deletePreset(id)).map((p) => ({ ...p, pipeline: normalizePipeline(p.pipeline) }));
  useStore.setState({ presets });
  if (S().presetId === id) useStore.setState({ presetId: null });
}

// ---------------------------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------------------------

export async function startExport(ids?: number[]) {
  const s = S();
  if (s.exportRun.running) return; // double click / Ctrl+E while an export is running
  const list = ids ?? (s.selection.length > 1 ? s.selection : s.items.map((i) => i.id));
  const usable = list.filter((id) => {
    const it = s.items.find((i) => i.id === id);
    return it && it.info;
  });
  if (!usable.length) {
    toast({ kind: "info", title: t("Nothing to export"), body: t("Add images first.") });
    return;
  }
  const needed = s.pipeline.steps
    .filter((x) => x.enabled && x.type === "removeBackground" && (x.model === "quality" || x.model === "hair"))
    .map((x): string => (x.type === "removeBackground" && x.model === "hair" ? "bg-hair" : "bg-quality"));
  const missing = s.models.find((m) => needed.includes(m.id) && !m.installed);
  if (missing) {
    toast({
      kind: "warning",
      title: t("“{name}” model not downloaded yet", { name: t(missing.name) }),
      body: t("This pipeline needs a one-time {size} download before exporting.", { size: fmtBytes(missing.downloadBytes) }),
      action: { label: t("Download"), run: () => installModel(missing.id) },
    });
    return;
  }
  if (s.settings?.export.location === "custom" && !s.settings.export.folder) {
    useStore.setState({ exportDialogOpen: true });
    return;
  }
  try {
    useStore.setState({ exportRun: { running: true, progress: { done: 0, total: usable.length, current: null, label: t("Starting"), fraction: 0 }, summary: null } });
    await flushSettings();
    await api.startExport(usable, s.pipeline);
  } catch (e) {
    useStore.setState({ exportRun: { running: false, progress: null, summary: null } });
    toast({ kind: "error", title: t("Export could not start"), body: tr(String(e)) });
  }
}

export async function copyResult() {
  const id = S().selectedId;
  if (id == null) return;
  try {
    await api.copyResult(id);
    toast({ kind: "success", title: t("Copied to clipboard"), body: t("Paste into any app that supports images.") });
  } catch (e) {
    toast({ kind: "error", title: t("Copy failed"), body: tr(String(e)) });
  }
}

// ---------------------------------------------------------------------------------------------
// mask editing
// ---------------------------------------------------------------------------------------------

export function enterMask() {
  const s = S();
  const hasBg = s.pipeline.steps.some((x) => x.enabled && x.type === "removeBackground");
  if (!hasBg || s.selectedId == null) {
    toast({ kind: "info", title: t("Add “Remove background” first"), body: t("The brush refines the cut-out of the background removal step.") });
    return;
  }
  useStore.setState({ mask: { ...s.mask, active: true, undo: [], redo: [], pending: [] }, viewMode: "result" });
  schedulePreview(0);
}

export function exitMask() {
  const s = S();
  if (!s.mask.active) return;
  useStore.setState({ mask: { ...s.mask, active: false, pending: [] } });
  schedulePreview(0);
}

/** Brush strokes per item (items live for the session only, so the UI is authoritative). */
const strokeCache = new Map<number, Stroke[]>();

export function strokesOf(id: number): Stroke[] {
  return strokeCache.get(id) ?? [];
}

async function pushStrokes(id: number, strokes: Stroke[]) {
  strokeCache.set(id, strokes);
  const it = await api.setStrokes(id, strokes);
  if (it) upsertItem(it);
  schedulePreview(0);
}

export async function commitStroke(st: Stroke) {
  const s = S();
  const id = s.selectedId;
  if (id == null) return;
  const prev = strokesOf(id);
  useStore.setState({ mask: { ...s.mask, undo: [...s.mask.undo, prev], redo: [], pending: [...s.mask.pending, st] } });
  await pushStrokes(id, [...prev, st]);
}

export async function undoStroke() {
  const s = S();
  const id = s.selectedId;
  if (id == null || !s.mask.undo.length) return;
  const prev = s.mask.undo[s.mask.undo.length - 1];
  useStore.setState({ mask: { ...s.mask, undo: s.mask.undo.slice(0, -1), redo: [...s.mask.redo, strokesOf(id)], pending: [] } });
  await pushStrokes(id, prev);
}

export async function redoStroke() {
  const s = S();
  const id = s.selectedId;
  if (id == null || !s.mask.redo.length) return;
  const next = s.mask.redo[s.mask.redo.length - 1];
  useStore.setState({ mask: { ...s.mask, undo: [...s.mask.undo, strokesOf(id)], redo: s.mask.redo.slice(0, -1), pending: [] } });
  await pushStrokes(id, next);
}

export async function resetStrokes() {
  const s = S();
  const id = s.selectedId;
  if (id == null) return;
  useStore.setState({ mask: { ...s.mask, undo: [...s.mask.undo, strokesOf(id)], redo: [], pending: [] } });
  await pushStrokes(id, []);
}

// ---------------------------------------------------------------------------------------------
// models / GPU
// ---------------------------------------------------------------------------------------------

export async function installModel(id: string) {
  if (S().downloads[id]) return; // already downloading
  try {
    useStore.setState((s) => ({ downloads: { ...s.downloads, [id]: { key: id, done: 0, total: 1, label: t("Connecting") } } }));
    await api.installModel(id);
  } catch (e) {
    useStore.setState((s) => {
      const d = { ...s.downloads };
      delete d[id];
      return { downloads: d };
    });
    toast({ kind: "error", title: t("Download failed"), body: tr(String(e)) });
  }
}

export async function installGpuPack() {
  if (S().downloads["gpu-pack"]) return; // already downloading
  try {
    useStore.setState((s) => ({ downloads: { ...s.downloads, "gpu-pack": { key: "gpu-pack", done: 0, total: 1, label: t("Connecting") } } }));
    await api.installGpuPack();
  } catch (e) {
    useStore.setState((s) => {
      const d = { ...s.downloads };
      delete d["gpu-pack"];
      return { downloads: d };
    });
    toast({ kind: "error", title: t("GPU pack download failed"), body: tr(String(e)) });
  }
}
