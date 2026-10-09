import { Brush, Download, ImagePlus, Link2, Unlink2 } from "lucide-react";
import { api } from "../api";
import { enterMask, installModel } from "../actions";
import { useStore } from "../store";
import type { BackdropMode, BgModel, ImageFit, Refine, ResizeMode, Rgba, ShadowMode, StepEntry } from "../types";
import { fmtBytes } from "../lib/format";
import { CheckRow, ColorField, NumberField, Segmented, Slider } from "./ui";
import { useState } from "react";
import { t } from "../i18n";

type Upd = (patch: Partial<StepEntry>) => void;

// ---------------------------------------------------------------------------------------------
// Remove background
// ---------------------------------------------------------------------------------------------
const BG_OPTIONS: { id: BgModel; model?: string; name: string; desc: string }[] = [
  { id: "auto", name: "Auto", desc: "Best installed model your hardware runs quickly" },
  { id: "fast", model: "bg-fast", name: "Fast", desc: "Quick, solid cut-outs on any computer" },
  { id: "quality", model: "bg-quality", name: "Best quality", desc: "Cleanest edges on products, people and cars" },
  { id: "hair", model: "bg-hair", name: "Hair & fur", desc: "Soft alpha for hair, fur, smoke and glass" },
];

function speed(secs: number) {
  return secs < 1 ? `~${secs.toFixed(1)} s` : `~${Math.round(secs)} s`;
}

export function RemoveBackgroundEditor({ step, upd }: { step: Extract<StepEntry, { type: "removeBackground" }>; upd: Upd }) {
  const models = useStore((s) => s.models);
  const downloads = useStore((s) => s.downloads);
  const cuda = useStore((s) => s.runtime?.info?.cudaReady ?? false);
  const edits = useStore((s) => s.items.find((i) => i.id === s.selectedId)?.edits ?? 0);
  const [adv, setAdv] = useState(false);
  const r = step.refine;
  const setR = (p: Partial<Refine>) => upd({ refine: { ...r, ...p } } as Partial<StepEntry>);
  return (
    <>
      <div className="model-pick" role="radiogroup" aria-label={t("Model")}>
        {BG_OPTIONS.map((o) => {
          const m = o.model ? models.find((x) => x.id === o.model) : null;
          const dl = o.model ? downloads[o.model] : undefined;
          const installed = !m || m.installed;
          return (
            <div key={o.id} className={`model-opt${step.model === o.id ? " on" : ""}`} role="radio" aria-checked={step.model === o.id} tabIndex={0}
              onClick={() => upd({ model: o.id } as Partial<StepEntry>)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), upd({ model: o.id } as Partial<StepEntry>))}>
              <span className="radio" />
              <span style={{ minWidth: 0 }}>
                <div className="m-name">{t(o.name)}</div>
                <div className="m-desc">{t(o.desc)}</div>
              </span>
              <span className="m-side">
                {m && !installed ? (
                  dl ? (
                    <span className="num">{Math.round((dl.done / Math.max(1, dl.total)) * 100)}%</span>
                  ) : (
                    <button className="btn sm" onClick={(e) => { e.stopPropagation(); installModel(m.id); }} data-tip={t("One-time {size} download", { size: fmtBytes(m.downloadBytes) })} data-tip-pos="left">
                      <Download size={13} /> {fmtBytes(m.downloadBytes)}
                    </button>
                  )
                ) : m ? (
                  speed(cuda ? m.gpuSecs : m.cpuSecs)
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
      <div className="field">
        <div className="field-label">{t("Edge")}</div>
        <div className="field-row">
          <span className="field-label">{t("Shift edge")}</span>
          <Slider value={r.edgeShift} min={-10} max={10} onChange={(v) => setR({ edgeShift: v })} format={(v) => `${v > 0 ? "+" : ""}${v} px`} ariaLabel={t("Shift edge")} />
        </div>
        <div className="field-row">
          <span className="field-label">{t("Feather")}</span>
          <Slider value={r.feather} min={0} max={20} step={0.5} onChange={(v) => setR({ feather: v })} format={(v) => `${v} px`} ariaLabel={t("Feather")} />
        </div>
        <div className="field-row">
          <span className="field-label">{t("Hardness")}</span>
          <Slider value={Math.round(r.hardness * 100)} min={0} max={100} onChange={(v) => setR({ hardness: v / 100 })} format={(v) => `${v}%`} ariaLabel={t("Edge hardness")} />
        </div>
      </div>
      <CheckRow checked={r.decontaminate} onChange={(v) => setR({ decontaminate: v })}>
        {t("Remove background color from edges")}
      </CheckRow>
      <button className="subtle-toggle" onClick={() => setAdv(!adv)}>
        {adv ? t("Hide options") : t("More options")}
      </button>
      {adv && (
        <>
          <CheckRow checked={r.removeIslands} onChange={(v) => setR({ removeIslands: v })}>
            {t("Remove stray specks")}
          </CheckRow>
          <CheckRow checked={r.edgeSnap} onChange={(v) => setR({ edgeSnap: v })}>
            {t("Snap edges to detail (large photos)")}
          </CheckRow>
        </>
      )}
      <button className="btn" onClick={enterMask}>
        <Brush size={14} /> {t("Refine with brush")}{edits ? ` · ${t(edits === 1 ? "{n} edit" : "{n} edits", { n: edits })}` : ""}
      </button>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
export function TrimEditor({ step, upd }: { step: Extract<StepEntry, { type: "trim" }>; upd: Upd }) {
  return (
    <>
      <Segmented
        full
        value={step.mode}
        onChange={(mode) => upd({ mode } as Partial<StepEntry>)}
        options={[
          { value: "auto", label: t("Auto"), title: t("Transparent margins, or solid borders on opaque images") },
          { value: "alpha", label: t("Transparent") },
          { value: "color", label: t("Solid color") },
        ]}
      />
      <div className="field-row">
        <span className="field-label">{step.mode === "color" ? t("Tolerance") : t("Threshold")}</span>
        <Slider value={step.threshold} min={0} max={64} onChange={(threshold) => upd({ threshold } as Partial<StepEntry>)} ariaLabel={t("Threshold")} />
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
export function PaddingEditor({ step, upd }: { step: Extract<StepEntry, { type: "padding" }>; upd: Upd }) {
  const linked = step.top === step.right && step.right === step.bottom && step.bottom === step.left;
  const [unlinked, setUnlinked] = useState(!linked);
  const setAll = (v: number) => upd({ top: v, right: v, bottom: v, left: v } as Partial<StepEntry>);
  const quick = step.unit === "px" ? [0, 16, 32, 64, 128] : [0, 2, 5, 10, 15];
  return (
    <>
      <div className="row">
        <div className="chips" style={{ flex: 1 }}>
          {quick.map((q) => (
            <button key={q} className={`chip${linked && step.top === q ? " on" : ""}`} onClick={() => { setUnlinked(false); setAll(q); }}>
              {q}
            </button>
          ))}
        </div>
        <Segmented value={step.unit} onChange={(unit) => upd({ unit } as Partial<StepEntry>)} options={[{ value: "px", label: "px" }, { value: "percent", label: "%" }]} />
      </div>
      {!unlinked ? (
        <div className="row">
          <div style={{ flex: 1 }}>
            <NumberField label={t("All")} value={step.top} min={0} max={step.unit === "px" ? 4000 : 100} unit={step.unit === "px" ? "px" : "%"} onChange={setAll} />
          </div>
          <button className="icon-btn" data-tip={t("Set each side")} onClick={() => setUnlinked(true)} aria-label={t("Set each side")}>
            <Link2 size={15} />
          </button>
        </div>
      ) : (
        <div className="row" style={{ alignItems: "flex-start" }}>
          <div className="grid2" style={{ flex: 1 }}>
            {(["top", "right", "bottom", "left"] as const).map((k) => (
              <NumberField key={k} label={k[0].toUpperCase()} ariaLabel={k} value={step[k]} min={0} max={step.unit === "px" ? 4000 : 100} unit={step.unit === "px" ? "px" : "%"} onChange={(v) => upd({ [k]: v } as Partial<StepEntry>)} />
            ))}
          </div>
          <button className="icon-btn on" data-tip={t("Same on all sides")} onClick={() => { setUnlinked(false); setAll(step.top); }} aria-label={t("Same on all sides")}>
            <Unlink2 size={15} />
          </button>
        </div>
      )}
      <div className="field">
        <div className="field-label">{t("Fill")}</div>
        <ColorField allowTransparent value={step.color as Rgba} onChange={(color) => upd({ color } as Partial<StepEntry>)} />
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
const SIZE_PRESETS: { label: string; w: number; h: number }[] = [
  { label: "512²", w: 512, h: 512 },
  { label: "1024²", w: 1024, h: 1024 },
  { label: "2048²", w: 2048, h: 2048 },
  { label: "HD", w: 1280, h: 720 },
  { label: "Full HD", w: 1920, h: 1080 },
  { label: "4K", w: 3840, h: 2160 },
  { label: "4:5 post", w: 1080, h: 1350 },
  { label: "Story", w: 1080, h: 1920 },
];

const MODE_HELP: Record<ResizeMode, string> = {
  fit: "Fits inside the box, keeps proportions",
  fill: "Fills the box exactly, crops the overflow",
  pad: "Fits inside, then pads to exactly this size",
  exact: "Stretches to exactly this size",
  width: "Sets the width, height follows",
  height: "Sets the height, width follows",
  percent: "Scales by a percentage",
};

export function ResizeEditor({ step, upd }: { step: Extract<StepEntry, { type: "resize" }>; upd: Upd }) {
  const byPercent = step.mode === "percent";
  const keep = step.mode !== "exact";
  const boxMode = step.mode === "fit" || step.mode === "fill" || step.mode === "pad" || step.mode === "exact";
  return (
    <>
      <Segmented
        full
        value={byPercent ? "percent" : "size"}
        onChange={(v) => upd({ mode: v === "percent" ? "percent" : "fit" } as Partial<StepEntry>)}
        options={[{ value: "size", label: t("Size in pixels") }, { value: "percent", label: t("Percentage") }]}
      />
      {byPercent ? (
        <>
          <div className="chips">
            {[25, 50, 75, 150, 200].map((p) => (
              <button key={p} className={`chip${step.percent === p ? " on" : ""}`} onClick={() => upd({ percent: p } as Partial<StepEntry>)}>
                {p}%
              </button>
            ))}
          </div>
          <NumberField label={t("Scale")} value={step.percent} min={1} max={800} unit="%" onChange={(percent) => upd({ percent } as Partial<StepEntry>)} />
        </>
      ) : (
        <>
          <div className="grid2">
            <NumberField label="W" ariaLabel={t("Width")} value={step.width} min={0} max={30000} unit="px"
              onChange={(width) => upd({ width, mode: step.mode === "height" && width > 0 ? (step.height ? "fit" : "width") : step.mode } as Partial<StepEntry>)} />
            <NumberField label="H" ariaLabel={t("Height")} value={step.height} min={0} max={30000} unit="px"
              onChange={(height) => upd({ height, mode: step.mode === "width" && height > 0 ? (step.width ? "fit" : "height") : step.mode } as Partial<StepEntry>)} />
          </div>
          <div className="chips">
            {SIZE_PRESETS.map((p) => (
              <button key={p.label} className={`chip${step.width === p.w && step.height === p.h && boxMode ? " on" : ""}`}
                onClick={() => upd({ width: p.w, height: p.h, mode: boxMode ? step.mode : "fit" } as Partial<StepEntry>)}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="field-row">
            <span className="field-label">{t("Behavior")}</span>
            <select className="select" value={step.mode} onChange={(e) => upd({ mode: e.target.value as ResizeMode } as Partial<StepEntry>)}>
              <option value="fit">{t("Fit inside")}</option>
              <option value="fill">{t("Fill & crop")}</option>
              <option value="pad">{t("Fit & pad to size")}</option>
              <option value="exact">{t("Stretch to size")}</option>
              <option value="width">{t("Width only")}</option>
              <option value="height">{t("Height only")}</option>
            </select>
          </div>
          <div className="faint" style={{ fontSize: 12, marginTop: -4 }}>{t(MODE_HELP[step.mode])}</div>
          {step.mode === "pad" && (
            <div className="field">
              <div className="field-label">{t("Pad color")}</div>
              <ColorField allowTransparent value={step.background as Rgba} onChange={(background) => upd({ background } as Partial<StepEntry>)} />
            </div>
          )}
          {step.mode !== "fill" && step.mode !== "exact" && (
            <CheckRow checked={step.enlarge} onChange={(enlarge) => upd({ enlarge } as Partial<StepEntry>)}>
              {t("Allow enlarging small images")}
            </CheckRow>
          )}
          {!keep && <div className="faint" style={{ fontSize: 12 }}>{t("Proportions are not kept in this mode.")}</div>}
        </>
      )}
      <div className="field-row">
        <span className="field-label">{t("Filter")}</span>
        <select className="select" value={step.filter} onChange={(e) => upd({ filter: e.target.value } as Partial<StepEntry>)}>
          <option value="lanczos">{t("Lanczos (sharp)")}</option>
          <option value="bicubic">{t("Bicubic")}</option>
          <option value="bilinear">{t("Bilinear (soft)")}</option>
          <option value="nearest">{t("Nearest (pixel art)")}</option>
        </select>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
export function UpscaleEditor({ step, upd }: { step: Extract<StepEntry, { type: "upscale" }>; upd: Upd }) {
  const cuda = useStore((s) => s.runtime?.info?.cudaReady ?? false);
  const models = useStore((s) => s.models);
  const id = step.model === "general" ? "sr-general" : step.model === "photo" ? (step.scale === 2 ? "sr-photo-x2" : "sr-photo") : "sr-anime";
  const spec = models.find((m) => m.id === id);
  const perMp = spec ? (cuda ? spec.gpuSecs : spec.cpuSecs) : 0;
  return (
    <>
      <Segmented full value={step.scale} onChange={(scale) => upd({ scale } as Partial<StepEntry>)} options={[{ value: 2, label: "2×" }, { value: 4, label: "4×" }]} />
      <div className="model-pick" role="radiogroup" aria-label={t("Upscale model")}>
        {([
          { id: "general", name: t("General"), desc: t("Fast. Photos, screenshots and graphics") },
          { id: "photo", name: t("Photo (max quality)"), desc: t("Most detail on photos; best with an NVIDIA GPU") },
          { id: "illustration", name: t("Illustration"), desc: t("Drawings, anime, logos and flat art") },
        ] as const).map((o) => (
          <div key={o.id} className={`model-opt${step.model === o.id ? " on" : ""}`} role="radio" aria-checked={step.model === o.id} tabIndex={0}
            onClick={() => upd({ model: o.id } as Partial<StepEntry>)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), upd({ model: o.id } as Partial<StepEntry>))}>
            <span className="radio" />
            <span>
              <div className="m-name">{o.name}</div>
              <div className="m-desc">{o.desc}</div>
            </span>
            <span />
          </div>
        ))}
      </div>
      {step.model === "general" && (
        <div className="field-row">
          <span className="field-label">{t("Noise reduction")}</span>
          <Slider value={Math.round(step.denoise * 100)} min={0} max={100} step={50} onChange={(v) => upd({ denoise: v / 100 } as Partial<StepEntry>)}
            format={(v) => (v < 25 ? t("Low") : v < 75 ? t("Medium") : t("High"))} ariaLabel={t("Noise reduction")} />
        </div>
      )}
      <div className={`notice${!cuda && perMp > 20 ? " warning" : ""}`}>
        <span>
          {t("About {s} s per megapixel of input on {d}.", { s: perMp < 1 ? perMp.toFixed(1) : Math.round(perMp), d: cuda ? t("your GPU") : t("the CPU") })}
          {!cuda && perMp > 20 ? t(" This model is slow without an NVIDIA GPU — try “General”.") : ""}
        </span>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
export function EnhanceEditor({ step, upd }: { step: Extract<StepEntry, { type: "enhance" }>; upd: Upd }) {
  const cuda = useStore((s) => s.runtime?.info?.cudaReady ?? false);
  return (
    <>
      <div className="field-row">
        <span className="field-label">{t("AI denoise")}</span>
        <Slider value={Math.round(step.denoise * 100)} min={0} max={100} step={50} onChange={(v) => upd({ denoise: v / 100 } as Partial<StepEntry>)}
          format={(v) => (v === 0 ? t("Off") : v < 75 ? t("Medium") : t("Strong"))} ariaLabel={t("AI denoise")} />
      </div>
      {step.denoise > 0 && !cuda && <div className="notice warning"><span>{t("AI denoising is slow on the CPU (about 10–30 s per megapixel).")}</span></div>}
      <div className="field-row">
        <span className="field-label">{t("Sharpen")}</span>
        <Slider value={Math.round(step.sharpen * 100)} min={0} max={100} onChange={(v) => upd({ sharpen: v / 100 } as Partial<StepEntry>)} format={(v) => (v === 0 ? t("Off") : `${v}%`)} ariaLabel={t("Sharpen")} />
      </div>
      <CheckRow checked={step.autoLevels} onChange={(autoLevels) => upd({ autoLevels } as Partial<StepEntry>)}>
        {t("Auto levels (fix flat contrast)")}
      </CheckRow>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
export function BackgroundEditor({ step, upd }: { step: Extract<StepEntry, { type: "background" }>; upd: Upd }) {
  const depthModel = useStore((s) => s.models.find((m) => m.id === "depth"));
  const dl = useStore((s) => s.downloads["depth"]);
  const set = (p: Partial<Extract<StepEntry, { type: "background" }>>) => upd(p as Partial<StepEntry>);
  const opaque = (c: Rgba): Rgba => [c[0], c[1], c[2], 255];
  return (
    <>
      <Segmented
        full
        ariaLabel={t("Background")}
        value={step.mode}
        onChange={(mode: BackdropMode) => set({ mode })}
        options={[
          { value: "color", label: t("Colour") },
          { value: "gradient", label: t("Gradient") },
          { value: "blur", label: t("Blur"), title: t("The original background, blurred") },
          { value: "image", label: t("Picture") },
        ]}
      />
      {step.mode === "color" && <ColorField value={step.color as Rgba} onChange={(color) => set({ color: opaque(color) })} />}
      {step.mode === "gradient" && (
        <>
          <div className="field">
            <div className="field-label">{t("From")}</div>
            <ColorField value={step.color as Rgba} onChange={(color) => set({ color: opaque(color) })} />
          </div>
          <div className="field">
            <div className="field-label">{t("To")}</div>
            <ColorField value={step.color2 as Rgba} onChange={(color2) => set({ color2: opaque(color2) })} />
          </div>
          <Segmented
            ariaLabel={t("Gradient shape")}
            value={step.radial ? "radial" : "linear"}
            onChange={(v: string) => set({ radial: v === "radial" })}
            options={[
              { value: "linear", label: t("Linear") },
              { value: "radial", label: t("Radial") },
            ]}
          />
          {!step.radial && (
            <div className="field-row">
              <span className="field-label">{t("Angle")}</span>
              <Slider min={0} max={355} step={5} value={step.angle} onChange={(angle) => set({ angle })} format={(v) => `${v}°`} ariaLabel={t("Angle")} />
            </div>
          )}
        </>
      )}
      {step.mode === "blur" && (
        <>
          <div className="field-row">
            <span className="field-label">{t("Blur")}</span>
            <Slider min={0.05} max={1} step={0.05} value={step.blur} onChange={(blur) => set({ blur })} format={(v) => `${Math.round(v * 100)}%`} ariaLabel={t("Blur")} />
          </div>
          <CheckRow checked={step.depth} onChange={(depth) => set({ depth })}>
            {t("Lens-like blur by distance")} <span className="badge accent" style={{ height: 16, fontSize: 10 }}>AI</span>
          </CheckRow>
          <div className="faint" style={{ fontSize: 12, marginTop: -4 }}>
            {step.depth
              ? t("Things near the subject stay sharp, the far background is blurred most — like a real camera.")
              : t("The whole background is blurred evenly.")}
          </div>
          {step.depth && (
            <div className="field-row" data-tip={t("Narrow: only the subject's plane stays sharp. Wide: more of the ground around it stays sharp.")} data-tip-pos="left">
              <span className="field-label">{t("Focus range")}</span>
              <Slider min={0} max={1} step={0.05} value={step.focus ?? 0.5} onChange={(focus) => set({ focus })} format={(v) => (v < 0.25 ? t("narrow") : v > 0.75 ? t("wide") : `${Math.round(v * 100)}%`)} ariaLabel={t("Focus range")} />
            </div>
          )}
          {step.depth && depthModel && !depthModel.installed && (
            <div className="row" style={{ gap: 8, alignItems: "center" }}>
              {dl ? (
                <span className="faint num" style={{ fontSize: 12 }}>{t("Downloading the depth model… {p}%", { p: Math.round((dl.done / Math.max(1, dl.total)) * 100) })}</span>
              ) : (
                <button className="btn sm" onClick={() => installModel("depth")}>
                  <Download size={13} /> {t("Download depth model ({size})", { size: fmtBytes(depthModel.downloadBytes) })}
                </button>
              )}
            </div>
          )}
        </>
      )}
      {step.mode === "image" && (
        <>
          <div className="row" style={{ gap: 8, alignItems: "center", minWidth: 0 }}>
            <button className="btn sm" onClick={async () => { const f = await api.chooseImage(t("Choose a background picture")); if (f) set({ image: f }); }}>
              <ImagePlus size={13} /> {step.image ? t("Change…") : t("Choose picture…")}
            </button>
            <span className="faint ellipsis" style={{ fontSize: 12, minWidth: 0 }} title={step.image ?? ""}>{step.image ? step.image.split(/[\\/]/).pop() : t("No picture chosen")}</span>
          </div>
          <Segmented
            ariaLabel={t("Fit")}
            value={step.fit}
            onChange={(fit: ImageFit) => set({ fit })}
            options={[
              { value: "cover", label: t("Fill"), title: t("Fill the whole canvas, cropping the picture if needed") },
              { value: "contain", label: t("Fit"), title: t("Show the whole picture") },
            ]}
          />
          {step.fit === "contain" && (
            <div className="field">
              <div className="field-label">{t("Border colour")}</div>
              <ColorField value={step.color as Rgba} onChange={(color) => set({ color: opaque(color) })} />
            </div>
          )}
        </>
      )}
      {step.mode !== "color" && (
        <div className="field-row">
          <span className="field-label">{t("Darken")}</span>
          <Slider min={0} max={1} step={0.05} value={step.dim} onChange={(dim) => set({ dim })} format={(v) => `${Math.round(v * 100)}%`} ariaLabel={t("Darken")} />
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------------------------
export function ShadowEditor({ step, upd }: { step: Extract<StepEntry, { type: "shadow" }>; upd: Upd }) {
  const set = (p: Partial<Extract<StepEntry, { type: "shadow" }>>) => upd(p as Partial<StepEntry>);
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  return (
    <>
      <Segmented
        full
        ariaLabel={t("Shadow")}
        value={step.mode}
        onChange={(mode: ShadowMode) => set({ mode })}
        options={[
          { value: "ground", label: t("On the ground"), title: t("The object stands on a surface: a contact shadow and a soft shadow under it") },
          { value: "drop", label: t("Drop shadow"), title: t("The silhouette, offset and blurred") },
        ]}
      />
      <div className="field-row">
        <span className="field-label">{t("Opacity")}</span>
        <Slider min={0} max={1} step={0.05} value={step.opacity} onChange={(opacity) => set({ opacity })} format={pct} ariaLabel={t("Opacity")} />
      </div>
      <div className="field-row">
        <span className="field-label">{t("Softness")}</span>
        <Slider min={0} max={1} step={0.05} value={step.softness} onChange={(softness) => set({ softness })} format={pct} ariaLabel={t("Softness")} />
      </div>
      {step.mode === "ground" ? (
        <div className="field-row">
          <span className="field-label">{t("Width")}</span>
          <Slider min={0} max={1} step={0.05} value={step.size} onChange={(size) => set({ size })} format={pct} ariaLabel={t("Width")} />
        </div>
      ) : (
        <>
          <div className="field-row">
            <span className="field-label">{t("Direction")}</span>
            <Slider min={0} max={355} step={5} value={step.angle} onChange={(angle) => set({ angle })} format={(v) => `${v}°`} ariaLabel={t("Direction")} />
          </div>
          <div className="field-row">
            <span className="field-label">{t("Distance")}</span>
            <Slider min={0} max={1} step={0.05} value={step.distance} onChange={(distance) => set({ distance })} format={pct} ariaLabel={t("Distance")} />
          </div>
        </>
      )}
      <div className="field">
        <div className="field-label">{t("Colour")}</div>
        <ColorField value={step.color as Rgba} onChange={(color) => set({ color: [color[0], color[1], color[2], 255] })} />
      </div>
      <div className="faint" style={{ fontSize: 12 }}>{t("Works on cut-outs with a transparent background. The canvas grows if the shadow needs room.")}</div>
    </>
  );
}
