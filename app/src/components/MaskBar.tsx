import { Eraser, Eye, Paintbrush, Redo2, RotateCcw, Undo2, Wand2 } from "lucide-react";
import { exitMask, redoStroke, resetStrokes, undoStroke } from "../actions";
import { useStore } from "../store";
import { Segmented, Slider } from "./ui";
import { t } from "../i18n";

export function MaskBar() {
  const mask = useStore((s) => s.mask);
  const set = useStore((s) => s.set);
  const edits = useStore((s) => s.items.find((i) => i.id === s.selectedId)?.edits ?? 0);
  const upd = (p: Partial<typeof mask>) => set({ mask: { ...mask, ...p } });
  return (
    <div className="mask-bar" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
      <Segmented
        ariaLabel={t("Brush")}
        value={mask.tool}
        onChange={(tool) => upd({ tool })}
        options={[
          { value: "keep", label: <><Paintbrush size={14} />{t("Keep")}</>, title: t("Paint areas to keep (K)") },
          { value: "erase", label: <><Eraser size={14} />{t("Erase")}</>, title: t("Paint areas to remove (E)") },
          { value: "restore", label: <><Wand2 size={14} />{t("Restore")}</>, title: t("Undo manual edits under the brush (A)") },
        ]}
      />
      <span className="lbl">{t("Size")}</span>
      <Slider value={mask.size} min={4} max={600} onChange={(size) => upd({ size })} format={(v) => `${v}px`} ariaLabel={t("Brush size")} />
      <span className="lbl">{t("Soft")}</span>
      <Slider value={Math.round((1 - mask.hardness) * 100)} min={0} max={100} step={5} onChange={(v) => upd({ hardness: 1 - v / 100 })} format={(v) => `${v}%`} ariaLabel={t("Brush softness")} />
      <span className="vsep" />
      <button className={`icon-btn${mask.showOriginal ? " on" : ""}`} aria-label={t("Show original underneath")} data-tip={t("Show original underneath")} data-tip-pos="top" onClick={() => upd({ showOriginal: !mask.showOriginal })}>
        <Eye size={16} />
      </button>
      <button className="icon-btn" aria-label={t("Undo")} data-tip={t("Undo (Ctrl+Z)")} data-tip-pos="top" disabled={!mask.undo.length} onClick={undoStroke}>
        <Undo2 size={16} />
      </button>
      <button className="icon-btn" aria-label={t("Redo")} data-tip={t("Redo (Ctrl+Y)")} data-tip-pos="top" disabled={!mask.redo.length} onClick={redoStroke}>
        <Redo2 size={16} />
      </button>
      <button className="icon-btn" aria-label={t("Clear all edits")} data-tip={t("Clear all brush edits")} data-tip-pos="top" disabled={!edits} onClick={resetStrokes}>
        <RotateCcw size={16} />
      </button>
      <span className="vsep" />
      <button className="btn sm primary" onClick={exitMask}>
        {t("Done")}
      </button>
    </div>
  );
}
