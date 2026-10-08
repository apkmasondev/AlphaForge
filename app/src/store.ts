import { create } from "zustand";
import type {
  DownloadProgress,
  ExportProgress,
  ExportSummary,
  Item,
  ModelStatus,
  PackStatus,
  Pipeline,
  Preset,
  PreviewDone,
  PreviewError,
  PreviewSize,
  RuntimeDto,
  Settings,
  Stroke,
  StrokeMode,
  SystemInfo,
} from "./types";

export type ViewMode = "original" | "result" | "compare";
export type SettingsTab = "general" | "ai" | "models" | "about";

export interface Toast {
  id: number;
  kind: "info" | "success" | "warning" | "error";
  title: string;
  body?: string;
  action?: { label: string; run: () => void };
  timeout?: number;
}

export interface PreviewState {
  id: number | null;
  seq: number;
  status: "idle" | "running" | "done" | "error";
  step: number;
  fraction: number;
  label: string;
  done: PreviewDone | null;
  size: PreviewSize | null;
  error: PreviewError | null;
  /** seq of the result image currently displayed (after it finished loading) */
  shownSeq: number;
  stage: "final" | "mask";
}

export interface MaskState {
  active: boolean;
  tool: StrokeMode;
  size: number;
  hardness: number;
  undo: Stroke[][];
  redo: Stroke[][];
  /** strokes drawn but not yet reflected in the displayed result */
  pending: Stroke[];
  showOriginal: boolean;
}

export interface AppStore {
  ready: boolean;
  version: string;
  system: SystemInfo | null;
  runtime: RuntimeDto | null;
  models: ModelStatus[];
  gpuPack: PackStatus | null;
  items: Item[];
  selectedId: number | null;
  selection: number[];
  pipeline: Pipeline;
  presetId: string | null;
  presets: Preset[];
  settings: Settings | null;
  preview: PreviewState;
  exportRun: { running: boolean; progress: ExportProgress | null; summary: ExportSummary | null };
  downloads: Record<string, DownloadProgress>;
  mask: MaskState;
  viewMode: ViewMode;
  zoomPct: number;
  dragging: boolean;
  toasts: Toast[];
  settingsOpen: SettingsTab | null;
  exportDialogOpen: boolean;
  savePresetOpen: boolean;
  set: (p: Partial<AppStore>) => void;
}

export const initialPreview: PreviewState = {
  id: null,
  seq: 0,
  status: "idle",
  step: 0,
  fraction: 0,
  label: "",
  done: null,
  size: null,
  error: null,
  shownSeq: 0,
  stage: "final",
};

export const useStore = create<AppStore>((set) => ({
  ready: false,
  version: "",
  system: null,
  runtime: null,
  models: [],
  gpuPack: null,
  items: [],
  selectedId: null,
  selection: [],
  pipeline: { steps: [], output: { format: "png", quality: 85, lossless: false, pngLevel: "balanced", pngColors: null, jpegProgressive: true, chroma: "auto", avifSpeed: 7, background: [255, 255, 255] } },
  presetId: null,
  presets: [],
  settings: null,
  preview: initialPreview,
  exportRun: { running: false, progress: null, summary: null },
  downloads: {},
  mask: { active: false, tool: "erase", size: 40, hardness: 0.6, undo: [], redo: [], pending: [], showOriginal: false },
  viewMode: "result",
  zoomPct: 100,
  dragging: false,
  toasts: [],
  settingsOpen: null,
  exportDialogOpen: false,
  savePresetOpen: false,
  set: (p) => set(p),
}));

export const S = () => useStore.getState();

let toastId = 1;
export function toast(t: Omit<Toast, "id">): number {
  const id = toastId++;
  useStore.setState((s) => ({ toasts: [...s.toasts.slice(-3), { ...t, id }] }));
  const timeout = t.timeout ?? (t.kind === "error" ? 9000 : t.action ? 8000 : 4500);
  if (timeout > 0) setTimeout(() => dismissToast(id), timeout);
  return id;
}
export function dismissToast(id: number) {
  useStore.setState((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
}

/** Ids removed from the list (ids are never reused). A late "item-updated" must not bring them back. */
export const removedIds = new Set<number>();

export function upsertItem(it: Item) {
  if (removedIds.has(it.id)) return;
  useStore.setState((s) => {
    const i = s.items.findIndex((x) => x.id === it.id);
    if (i < 0) return { items: [...s.items, it] };
    const items = s.items.slice();
    items[i] = it;
    return { items };
  });
}

export function selectedItem(): Item | null {
  const s = S();
  return s.items.find((i) => i.id === s.selectedId) ?? null;
}

/** Commands for the viewer (zoom buttons in the top bar, keyboard shortcuts). */
export const viewerBus = new EventTarget();
export function viewerCmd(cmd: "zoom-in" | "zoom-out" | "fit" | "actual") {
  viewerBus.dispatchEvent(new Event(cmd));
}
