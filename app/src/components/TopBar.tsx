import { Brush, Columns2, Copy, Image as ImageIcon, Maximize, Minus, Plus, Settings2, Sparkles } from "lucide-react";
import { copyResult, enterMask, exitMask, updateSettings } from "../actions";
import { useStore, viewerCmd } from "../store";
import { Logo, Segmented } from "./ui";
import { t } from "../i18n";

const BGS: { id: string; label: string; style?: React.CSSProperties; cls?: string }[] = [
  { id: "checker", label: "Checkerboard", cls: "checker" } as { id: string; label: string; style?: React.CSSProperties; cls?: string },
  { id: "white", label: "White", style: { background: "#ffffff" } },
  { id: "gray", label: "Gray", style: { background: "#808080" } },
  { id: "black", label: "Black", style: { background: "#000000" } },
];

export function TopBar() {
  const viewMode = useStore((s) => s.viewMode);
  const set = useStore((s) => s.set);
  const hasSel = useStore((s) => s.selectedId != null);
  const zoomPct = useStore((s) => s.zoomPct);
  const bg = useStore((s) => s.settings?.viewerBg ?? "checker");
  const maskActive = useStore((s) => s.mask.active);
  const hasResult = useStore((s) => s.preview.done != null);
  const hasBgStep = useStore((s) => s.pipeline.steps.some((x) => x.enabled && x.type === "removeBackground"));

  return (
    <header className="topbar" data-tauri-drag-region>
      <div className="brand" data-tauri-drag-region>
        <Logo />
        <span data-tauri-drag-region>AlphaForge</span>
      </div>
      <div className="center" data-tauri-drag-region>
        <Segmented
          ariaLabel={t("View")}
          value={viewMode}
          onChange={(v) => set({ viewMode: v })}
          options={[
            { value: "original", label: <><ImageIcon size={14} />{t("Original")}</>, title: t("Original (O)"), disabled: !hasSel },
            { value: "result", label: <><Sparkles size={14} />{t("Result")}</>, title: t("Result (R)"), disabled: !hasSel },
            { value: "compare", label: <><Columns2 size={14} />{t("Compare")}</>, title: t("Before / after slider (C)"), disabled: !hasSel || maskActive },
          ]}
        />
        <button
          className={`btn sm${maskActive ? " primary" : ""}`}
          disabled={!hasSel || !hasBgStep}
          onClick={() => (maskActive ? exitMask() : enterMask())}
          data-tip={hasBgStep ? t("Paint to keep or erase parts of the cut-out (B)") : t("Available with “Remove background”")}
        >
          <Brush size={14} />
          {maskActive ? t("Done refining") : t("Refine cut-out")}
        </button>
      </div>
      <div className="tools">
        <div className="bg-swatches" role="radiogroup" aria-label={t("Viewer background")}>
          {BGS.map((b) => (
            <button
              key={b.id}
              className={`bg-swatch ${b.cls ?? ""}${bg === b.id ? " on" : ""}`}
              style={b.style}
              aria-label={t("{c} background", { c: t(b.label) })}
              data-tip={t("{c} background", { c: t(b.label) })}
              onClick={() => updateSettings({ viewerBg: b.id })}
            />
          ))}
        </div>
        <span className="vsep" />
        <button className="icon-btn" aria-label={t("Zoom out")} data-tip={t("Zoom out (−)")} onClick={() => viewerCmd("zoom-out")} disabled={!hasSel}>
          <Minus size={16} />
        </button>
        <button className="zoom-label" onClick={() => viewerCmd("actual")} data-tip={t("Actual size (Ctrl+1)")} disabled={!hasSel}>
          {hasSel ? `${Math.round(zoomPct)}%` : "—"}
        </button>
        <button className="icon-btn" aria-label={t("Zoom in")} data-tip={t("Zoom in (+)")} onClick={() => viewerCmd("zoom-in")} disabled={!hasSel}>
          <Plus size={16} />
        </button>
        <button className="icon-btn" aria-label={t("Fit to window")} data-tip={t("Fit to window (Ctrl+0)")} onClick={() => viewerCmd("fit")} disabled={!hasSel}>
          <Maximize size={15} />
        </button>
        <span className="vsep" />
        <button className="icon-btn" aria-label={t("Copy result")} data-tip={t("Copy result to clipboard (Ctrl+C)")} onClick={copyResult} disabled={!hasResult}>
          <Copy size={15} />
        </button>
        <button className="icon-btn" aria-label={t("Settings")} data-tip={t("Settings")} data-tip-pos="left" onClick={() => set({ settingsOpen: "general" })}>
          <Settings2 size={16} />
        </button>
      </div>
    </header>
  );
}
