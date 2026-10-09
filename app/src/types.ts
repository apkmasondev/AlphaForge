// Mirrors of the Rust DTOs (serde camelCase).

export type Rgba = [number, number, number, number];

export interface SourceInfo {
  format: string;
  width: number;
  height: number;
  hasAlpha: boolean;
  bitDepth: number;
  colorConverted: boolean;
}

export type ItemStatus = "loading" | "ready" | "queued" | "processing" | "done" | "skipped" | "error";

export interface OutInfo {
  path: string | null;
  bytes: number;
  width: number;
  height: number;
  format: string;
  ms: number;
}

export interface Item {
  id: number;
  name: string;
  path: string | null;
  relDir: string | null;
  size: number;
  info: SourceInfo | null;
  hasThumb: boolean;
  status: ItemStatus;
  error: string | null;
  out: OutInfo | null;
  edits: number;
  canRedo: boolean;
}

// ---- pipeline ----
export type BgModel = "auto" | "fast" | "quality" | "hair";
export interface Refine {
  edgeShift: number;
  feather: number;
  hardness: number;
  removeIslands: boolean;
  decontaminate: boolean;
  edgeSnap: boolean;
}
export type TrimMode = "auto" | "alpha" | "color";
export type PadUnit = "px" | "percent";
export type ResizeMode = "percent" | "width" | "height" | "fit" | "fill" | "pad" | "exact";
export type Filter = "lanczos" | "bicubic" | "bilinear" | "nearest";
export type ShadowMode = "ground" | "drop";
export type BackdropMode = "color" | "gradient" | "blur" | "image";
export type ImageFit = "cover" | "contain";
export type SrModel = "general" | "photo" | "illustration";

export type Step =
  | { type: "removeBackground"; model: BgModel; refine: Refine }
  | { type: "trim"; mode: TrimMode; threshold: number }
  | { type: "padding"; top: number; right: number; bottom: number; left: number; unit: PadUnit; color: Rgba }
  | { type: "resize"; mode: ResizeMode; width: number; height: number; percent: number; filter: Filter; enlarge: boolean; background: Rgba }
  | { type: "upscale"; model: SrModel; scale: number; denoise: number }
  | { type: "enhance"; denoise: number; sharpen: number; autoLevels: boolean }
  | { type: "outline"; thickness: number; smooth: number; color: Rgba }
  | { type: "shadow"; mode: ShadowMode; opacity: number; softness: number; size: number; angle: number; distance: number; color: Rgba }
  | {
      type: "background";
      color: Rgba;
      mode: BackdropMode;
      color2: Rgba;
      angle: number;
      radial: boolean;
      blur: number;
      depth: boolean;
      focus: number;
      dim: number;
      image: string | null;
      fit: ImageFit;
    };

export type StepType = Step["type"];
export type StepEntry = Step & { id: string; enabled: boolean };

export type OutFormat = "same" | "png" | "jpeg" | "webp" | "avif";
export type PngLevel = "fast" | "balanced" | "max";
export type Chroma = "auto" | "420" | "444";

export interface Output {
  format: OutFormat;
  quality: number;
  lossless: boolean;
  pngLevel: PngLevel;
  pngColors: number | null;
  jpegProgressive: boolean;
  chroma: Chroma;
  avifSpeed: number;
  background: [number, number, number];
}

export interface Pipeline {
  steps: StepEntry[];
  output: Output;
}

export interface Preset {
  id: string;
  name: string;
  description: string;
  builtin: boolean;
  pipeline: Pipeline;
}

// ---- settings ----
export type Theme = "system" | "light" | "dark";
export type DevicePref = "auto" | "gpu" | "cpu";
export type Location = "subfolder" | "sameFolder" | "custom";
export type Conflict = "rename" | "overwrite" | "skip";

export interface ExportSettings {
  location: Location;
  folder: string | null;
  prefix: string;
  suffix: string;
  conflict: Conflict;
  keepStructure: boolean;
}

export interface Settings {
  theme: Theme;
  language: "system" | "en" | "pl";
  device: DevicePref;
  unloadAfterMin: number;
  cacheMb: number;
  export: ExportSettings;
  pipeline: Pipeline | null;
  presetId: string | null;
  autoPreview: boolean;
  viewerBg: string;
  gpuPromptDismissed: boolean;
  leftPanelWidth: number;
  rightPanelWidth: number;
}

// ---- system / AI ----
export interface GpuInfo {
  name: string;
  vramTotalMb: number;
  vramFreeMb: number;
  driverVersion: string;
  cudaDriver: [number, number];
  computeCapability: [number, number];
}

export interface SystemInfo {
  cpuName: string;
  cpuCores: number;
  cpuThreads: number;
  ramTotalMb: number;
  ramAvailableMb: number;
  gpus: GpuInfo[];
  os: string;
}

export interface RuntimeInfo {
  ortPath: string;
  cudaReady: boolean;
  gpuStatus: string;
  gpu: GpuInfo | null;
  gpuPackInstalled: boolean;
  gpuSupported: boolean;
  cpuThreads: number;
}

export type Device = "cpu" | "cuda";

export interface LoadedModel {
  template: string;
  device: Device;
  idleSecs: number;
}

export interface RuntimeDto {
  info: RuntimeInfo | null;
  error: string | null;
  loaded: LoadedModel[];
}

export interface ModelStatus {
  id: string;
  kind: "background" | "upscale" | "depth";
  name: string;
  tagline: string;
  family: string;
  template: string;
  license: string;
  homepage: string;
  bundled: boolean;
  downloadBytes: number;
  minVramMb: number;
  gpuSecs: number;
  cpuSecs: number;
  scale: number;
  installed: boolean;
  path: string | null;
  diskBytes: number;
}

export interface PackStatus {
  installed: boolean;
  dir: string;
  downloadBytes: number;
  diskBytes: number;
  installedBytes: number;
}

export interface Bootstrap {
  version: string;
  settings: Settings;
  presets: Preset[];
  items: Item[];
  system: SystemInfo;
  models: ModelStatus[];
  gpuPack: PackStatus;
}

export interface StepReport {
  id: string;
  kind: StepType;
  ms: number;
  device: Device | null;
  model: string | null;
  note: string | null;
  cached: boolean;
}

export interface PreviewDone {
  id: number;
  seq: number;
  width: number;
  height: number;
  partial: boolean;
  hasAlpha: boolean;
  format: string;
  ms: number;
  reports: StepReport[];
}

export interface PreviewProgress {
  id: number;
  seq: number;
  step: number;
  fraction: number;
  label: string;
}

export interface PreviewSize {
  id: number;
  seq: number;
  bytes: number;
  format: string;
  ms: number;
}

export interface PreviewError {
  id: number;
  seq: number;
  message: string;
  missingModel: string | null;
}

export interface ExportProgress {
  done: number;
  total: number;
  current: number | null;
  label: string;
  fraction: number;
}

export interface ExportSummary {
  ok: number;
  failed: number;
  skipped: number;
  cancelled: boolean;
  elapsedMs: number;
  inBytes: number;
  outBytes: number;
  folder: string | null;
  errors: [string, string][];
}

export interface DownloadProgress {
  key: string;
  done: number;
  total: number;
  label: string;
}

export interface DownloadDone {
  key: string;
  ok: boolean;
  cancelled: boolean;
  error: string | null;
}

export type StrokeMode = "keep" | "erase" | "restore";
export interface Stroke {
  mode: StrokeMode;
  radius: number;
  hardness: number;
  points: [number, number][];
}
