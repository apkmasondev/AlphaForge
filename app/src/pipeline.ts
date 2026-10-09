import type { Output, Pipeline, Refine, Step, StepEntry, StepType } from "./types";
import { rgbToHex } from "./lib/format";
import { t } from "./i18n";

export const DEFAULT_REFINE: Refine = {
  edgeShift: 0,
  feather: 0,
  hardness: 0,
  removeIslands: true,
  decontaminate: true,
  edgeSnap: true,
};

export const DEFAULT_OUTPUT: Output = {
  format: "png",
  quality: 85,
  lossless: false,
  pngLevel: "balanced",
  pngColors: null,
  jpegProgressive: true,
  chroma: "auto",
  avifSpeed: 7,
  background: [255, 255, 255],
};

export const STEP_ORDER: StepType[] = ["removeBackground", "trim", "padding", "shadow", "resize", "upscale", "enhance", "background"];

export const STEP_LABEL: Record<StepType, string> = {
  removeBackground: "Remove background",
  trim: "Trim",
  padding: "Padding",
  resize: "Resize",
  upscale: "AI upscale",
  enhance: "Enhance",
  shadow: "Shadow",
  background: "Background",
};

export const STEP_HINT: Record<StepType, string> = {
  removeBackground: "Cut out the main subject with a true alpha channel",
  trim: "Remove empty or transparent margins",
  padding: "Add a margin around the image",
  resize: "Scale, fit or crop to a size",
  upscale: "Increase resolution 2× or 4× with AI",
  enhance: "Reduce noise, sharpen, fix levels",
  shadow: "A soft shadow on the ground, or a drop shadow",
  background: "Colour, gradient, picture or the original background blurred",
};

let counter = 0;
export function newId(type: string): string {
  counter += 1;
  return `${type}-${Date.now().toString(36)}-${counter}`;
}

export function newStep(type: StepType): StepEntry {
  const id = newId(type);
  const base = { id, enabled: true };
  switch (type) {
    case "removeBackground":
      return { ...base, type, model: "auto", refine: { ...DEFAULT_REFINE } };
    case "trim":
      return { ...base, type, mode: "auto", threshold: 8 };
    case "padding":
      return { ...base, type, top: 32, right: 32, bottom: 32, left: 32, unit: "px", color: [0, 0, 0, 0] };
    case "resize":
      return { ...base, type, mode: "fit", width: 1024, height: 1024, percent: 50, filter: "lanczos", enlarge: false, background: [0, 0, 0, 0] };
    case "upscale":
      return { ...base, type, model: "general", scale: 2, denoise: 0.5 };
    case "enhance":
      return { ...base, type, denoise: 0, sharpen: 0.3, autoLevels: false };
    case "shadow":
      return { ...base, type, mode: "ground", opacity: 0.6, softness: 0.5, size: 0.5, angle: 60, distance: 0.25, color: [0, 0, 0, 255] };
    case "background":
      return { ...base, type, color: [255, 255, 255, 255], mode: "color", color2: [32, 34, 40, 255], angle: 90, radial: false, blur: 0.5, depth: true, focus: 0.5, dim: 0, image: null, fit: "cover" };
  }
}

/** Sensible position for a new step (keeps AI steps early, fill late). */
export function insertIndex(steps: StepEntry[], type: StepType): number {
  const rank = STEP_ORDER.indexOf(type);
  for (let i = 0; i < steps.length; i++) {
    if (STEP_ORDER.indexOf(steps[i].type) > rank) return i;
  }
  return steps.length;
}

const BG_MODEL_NAME: Record<string, string> = { auto: "Auto", fast: "Fast", quality: "Best quality", hair: "Hair & fur" };
const SR_NAME: Record<string, string> = { general: "General", photo: "Photo", illustration: "Illustration" };
const RESIZE_NAME: Record<string, string> = { percent: "Scale", width: "Width", height: "Height", fit: "Fit", fill: "Fill & crop", pad: "Fit & pad", exact: "Exact" };

export function stepLabel(type: StepType): string {
  return t(STEP_LABEL[type]);
}

export function stepHint(type: StepType): string {
  return t(STEP_HINT[type]);
}

export function stepSummary(s: Step): string {
  switch (s.type) {
    case "removeBackground": {
      const bits = [t(BG_MODEL_NAME[s.model])];
      if (s.refine.edgeShift) bits.push(t("edge {v}px", { v: `${s.refine.edgeShift > 0 ? "+" : ""}${s.refine.edgeShift}` }));
      if (s.refine.feather) bits.push(t("feather {v}px", { v: s.refine.feather }));
      if (s.refine.hardness) bits.push(t("hardness {v}%", { v: Math.round(s.refine.hardness * 100) }));
      return bits.join(" · ");
    }
    case "trim":
      return t(s.mode === "auto" ? "Transparent or uniform borders" : s.mode === "alpha" ? "Transparent margins" : "Uniform-color borders");
    case "padding": {
      const u = s.unit === "px" ? "px" : "%";
      const same = s.top === s.right && s.right === s.bottom && s.bottom === s.left;
      const fill = s.color[3] === 0 ? t("transparent") : rgbToHex(s.color);
      return same ? `${s.top} ${u} · ${fill}` : `${s.top}/${s.right}/${s.bottom}/${s.left} ${u} · ${fill}`;
    }
    case "resize":
      if (s.mode === "percent") return `${s.percent}%`;
      if (s.mode === "width") return t("Width {v} px", { v: s.width });
      if (s.mode === "height") return t("Height {v} px", { v: s.height });
      return `${t(RESIZE_NAME[s.mode])} ${s.width} × ${s.height}`;
    case "upscale":
      return `${s.scale}× · ${t(SR_NAME[s.model])}`;
    case "enhance": {
      const bits: string[] = [];
      if (s.denoise > 0) bits.push(t("denoise {v}%", { v: Math.round(s.denoise * 100) }));
      if (s.sharpen > 0) bits.push(t("sharpen {v}%", { v: Math.round(s.sharpen * 100) }));
      if (s.autoLevels) bits.push(t("auto levels"));
      return bits.length ? bits.join(" · ") : t("No changes");
    }
    case "shadow":
      return `${t(s.mode === "ground" ? "On the ground" : "Drop shadow")} · ${Math.round(s.opacity * 100)}%`;
    case "background":
      switch (s.mode) {
        case "gradient":
          return `${t(s.radial ? "Radial gradient" : "Gradient")} ${rgbToHex(s.color)} → ${rgbToHex(s.color2)}`;
        case "blur":
          return `${t("Blurred original")} ${Math.round(s.blur * 100)}%${s.depth ? ` · ${t("AI depth")}` : ""}`;
        case "image":
          return s.image ? `${t("Picture")} · ${s.image.split(/[\\/]/).pop()}` : t("Picture (none chosen)");
        default:
          return rgbToHex(s.color);
      }
  }
}

export function outputSummary(o: Output): string {
  switch (o.format) {
    case "png":
      return o.pngColors ? t("PNG · {n} colors", { n: o.pngColors }) : t(`PNG · ${o.pngLevel}`);
    case "jpeg":
      return t("JPG · quality {q}", { q: o.quality });
    case "webp":
      return o.lossless ? t("WebP · lossless") : t("WebP · quality {q}", { q: o.quality });
    case "avif":
      return t("AVIF · quality {q}", { q: o.quality });
    case "same":
      return t("Same as source · quality {q}", { q: o.quality });
  }
}

export function clonePipeline(p: Pipeline): Pipeline {
  return JSON.parse(JSON.stringify(p));
}

/** Pipelines equal ignoring step ids (used for the "modified" indicator). */
export function samePipeline(a: Pipeline | null | undefined, b: Pipeline | null | undefined): boolean {
  if (!a || !b) return false;
  const strip = (p: Pipeline) => JSON.stringify({ ...p, steps: p.steps.map(({ id: _id, ...rest }) => rest) });
  return strip(a) === strip(b);
}

/** Make sure every field exists (presets saved by older versions, partial JSON). */
export function normalizePipeline(p: Pipeline): Pipeline {
  return {
    output: { ...DEFAULT_OUTPUT, ...(p.output ?? {}) },
    steps: (p.steps ?? []).map((s) => {
      const d = newStep(s.type);
      const merged = { ...d, ...s, id: s.id || d.id } as StepEntry;
      if (merged.type === "removeBackground") merged.refine = { ...DEFAULT_REFINE, ...(merged.refine ?? {}) };
      return merged;
    }),
  };
}

export function hasStep(p: Pipeline, t: StepType): boolean {
  return p.steps.some((s) => s.enabled && s.type === t);
}
