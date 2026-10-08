import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  Bootstrap,
  DownloadDone,
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
  PreviewProgress,
  PreviewSize,
  RuntimeDto,
  Settings,
  Stroke,
} from "./types";

export interface AddResult {
  added: Item[];
  skipped: number;
  limited: boolean;
}

export const api = {
  bootstrap: () => invoke<Bootstrap>("bootstrap"),
  runtimeInfo: () => invoke<RuntimeDto>("runtime_info"),
  frontendReady: () => invoke<void>("frontend_ready"),

  addPaths: (paths: string[]) => invoke<AddResult>("add_paths", { paths }),
  openFilesDialog: () => invoke<AddResult>("open_files_dialog"),
  openFolderDialog: () => invoke<AddResult>("open_folder_dialog"),
  chooseFolder: (title: string) => invoke<string | null>("choose_folder", { title }),
  chooseImage: (title: string) => invoke<string | null>("choose_image", { title }),
  listItems: () => invoke<Item[]>("list_items"),
  removeItems: (ids: number[]) => invoke<void>("remove_items", { ids }),
  clearItems: () => invoke<void>("clear_items"),
  reprocess: (id: number) => invoke<Item | null>("reprocess", { id }),

  preview: (id: number, pipeline: Pipeline, stage: "final" | "mask") => invoke<number>("preview", { id, pipeline, stage }),
  cancelPreview: () => invoke<void>("cancel_preview"),
  setStrokes: (id: number, strokes: Stroke[]) => invoke<Item | null>("set_strokes", { id, strokes }),
  pipelineWarnings: (pipeline: Pipeline) => invoke<string[]>("pipeline_warnings", { pipeline }),

  startExport: (ids: number[], pipeline: Pipeline) => invoke<void>("start_export", { ids, pipeline }),
  cancelExport: () => invoke<void>("cancel_export"),

  pasteClipboard: () => invoke<{ added: number[]; kind: "files" | "image" | "none" }>("paste_clipboard"),
  addImageBytes: (name: string, bytes: number[]) => invoke<Item | null>("add_image_bytes", { name, bytes }),
  copyResult: (id: number) => invoke<void>("copy_result", { id }),
  revealPath: (path: string) => invoke<void>("reveal_path", { path }),
  openFolder: (path: string) => invoke<void>("open_folder", { path }),
  openLicenses: () => invoke<void>("open_licenses"),
  openDataFolder: () => invoke<void>("open_data_folder"),

  saveSettings: (settings: Settings) => invoke<void>("save_settings", { settings }),
  savePreset: (id: string | null, name: string, pipeline: Pipeline) => invoke<Preset[]>("save_preset", { id, name, pipeline }),
  deletePreset: (id: string) => invoke<Preset[]>("delete_preset", { id }),

  models: () => invoke<ModelStatus[]>("models"),
  installModel: (id: string) => invoke<void>("install_model", { id }),
  removeModel: (id: string) => invoke<ModelStatus[]>("remove_model", { id }),
  gpuPackStatus: () => invoke<PackStatus>("gpu_pack_status"),
  installGpuPack: () => invoke<void>("install_gpu_pack"),
  removeGpuPack: () => invoke<PackStatus>("remove_gpu_pack"),
  cancelDownload: (key: string) => invoke<void>("cancel_download", { key }),
  unloadModels: () => invoke<void>("unload_models"),
  restartApp: () => invoke<void>("restart_app"),
};

export interface Events {
  "item-updated": Item;
  "items-changed": null;
  "preview-progress": PreviewProgress;
  "preview-done": PreviewDone;
  "preview-size": PreviewSize;
  "preview-error": PreviewError;
  "export-progress": ExportProgress;
  "export-done": ExportSummary;
  "download-progress": DownloadProgress;
  "download-done": DownloadDone;
  "runtime-ready": null;
  "models-unloaded": number;
}

export function on<K extends keyof Events>(name: K, cb: (payload: Events[K]) => void): Promise<UnlistenFn> {
  return listen<Events[K]>(name, (e) => cb(e.payload));
}

/** URL of an image served by the backend (`afimg://`, exposed as http://afimg.localhost on Windows). */
export function imgUrl(kind: "thumb" | "original" | "result", id: number, v: number | string = 0): string {
  return `http://afimg.localhost/${kind}/${id}?v=${v}`;
}
