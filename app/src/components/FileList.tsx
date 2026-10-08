import { memo } from "react";
import { AlertCircle, Check, ClipboardPaste, Clock, Folder, FolderOpen, ImagePlus, MoreHorizontal, Plus, RotateCw, Trash2, X } from "lucide-react";
import { api, imgUrl } from "../api";
import { clearAll, openFiles, openFolder, paste, removeItems, reprocess, select, startExport } from "../actions";
import { useStore } from "../store";
import type { Item } from "../types";
import { fmtBytes, fmtDims, fmtMs, fmtSaved, shortPath } from "../lib/format";
import { Menu, useMenu } from "./ui";
import { t, tr } from "../i18n";

const Row = memo(function Row({ it, selected, multi }: { it: Item; selected: boolean; multi: boolean }) {
  const menu = useMenu();
  const saved = it.out ? fmtSaved(it.size, it.out.bytes) : null;
  return (
    <div
      className={`file${selected ? " selected" : ""}${multi ? " multi" : ""}`}
      role="option"
      aria-selected={selected}
      tabIndex={-1}
      onMouseDown={(e) => {
        if (e.button !== 0) return;
        select(it.id, { additive: e.ctrlKey, range: e.shiftKey });
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        menu.open(e.currentTarget as HTMLElement);
      }}
    >
      <div className="thumb">{it.hasThumb ? <img src={imgUrl("thumb", it.id, it.size)} alt="" draggable={false} /> : <span className="spinner" style={{ opacity: it.status === "loading" ? 1 : 0 }} />}</div>
      <div style={{ minWidth: 0 }}>
        <div className="name ellipsis" title={it.path ?? it.name}>
          {it.name}
        </div>
        <div className="meta">
          {it.status === "error" && it.error ? (
            <span className="err" title={tr(it.error)}>
              {tr(it.error)}
            </span>
          ) : it.out ? (
            <>
              <span className="grow">
                {fmtDims(it.out.width, it.out.height)} · {fmtBytes(it.out.bytes)}
              </span>
              {saved && <span className={saved.better ? "saved" : "worse"}>{saved.text}</span>}
            </>
          ) : it.info ? (
            <>
              <span>{fmtDims(it.info.width, it.info.height)}</span>
              <span>·</span>
              <span>{fmtBytes(it.size)}</span>
              {it.relDir && (
                <>
                  <span>·</span>
                  <span className="ellipsis" title={it.relDir}>
                    {it.relDir}
                  </span>
                </>
              )}
            </>
          ) : (
            <span>{it.status === "loading" ? t("Reading…") : fmtBytes(it.size)}</span>
          )}
        </div>
      </div>
      <StatusIcon it={it} />
      <div className="actions">
        <button className="icon-btn sm" aria-label={t("More")} onMouseDown={(e) => e.stopPropagation()} onClick={(e) => menu.open(e.currentTarget)}>
          <MoreHorizontal size={15} />
        </button>
        <button
          className="icon-btn sm"
          aria-label={t("Remove from list")}
          data-tip={t("Remove from list")}
          data-tip-pos="left"
          onMouseDown={(e) => e.stopPropagation()}
          onClick={() => removeItems([it.id])}
        >
          <X size={15} />
        </button>
      </div>
      {menu.anchor && (
        <Menu
          anchor={menu.anchor}
          onClose={menu.close}
          align="right"
          items={[
            { label: t("Export this image"), icon: <ImagePlus size={15} />, onClick: () => startExport([it.id]), disabled: !it.info },
            { label: t("Reprocess"), icon: <RotateCw size={15} />, hint: t("re-read file"), onClick: () => reprocess(it.id) },
            { sep: true },
            { label: t("Show output in folder"), icon: <FolderOpen size={15} />, disabled: !it.out?.path, onClick: () => it.out?.path && api.revealPath(it.out.path) },
            { label: t("Show original in folder"), icon: <Folder size={15} />, disabled: !it.path, onClick: () => it.path && api.revealPath(it.path) },
            { sep: true },
            { label: t("Remove from list"), icon: <Trash2 size={15} />, danger: true, onClick: () => removeItems([it.id]) },
          ]}
        />
      )}
    </div>
  );
});

function StatusIcon({ it }: { it: Item }) {
  switch (it.status) {
    case "processing":
      return (
        <span className="status-icon" aria-label={t("Processing…")}>
          <span className="spinner" />
        </span>
      );
    case "queued":
      return (
        <span className="status-icon queued" aria-label={t("Queued")}>
          <Clock size={14} />
        </span>
      );
    case "done":
      return (
        <span className="status-icon done" aria-label={t("Exported")} data-tip={it.out ? t("Saved in {t}", { t: fmtMs(it.out.ms) }) : t("Exported")} data-tip-pos="left">
          <Check size={15} />
        </span>
      );
    case "error":
      return (
        <span className="status-icon error" aria-label={t("Error")}>
          <AlertCircle size={15} />
        </span>
      );
    case "skipped":
      return (
        <span className="status-icon queued" aria-label={t("Skipped")} data-tip={t("Skipped — output already exists")} data-tip-pos="left">
          <Check size={15} />
        </span>
      );
    default:
      return <span className="status-icon" />;
  }
}

export function FileList() {
  const items = useStore((s) => s.items);
  const selectedId = useStore((s) => s.selectedId);
  const selection = useStore((s) => s.selection);
  const exportRun = useStore((s) => s.exportRun);
  const exportSettings = useStore((s) => s.settings?.export);
  const set = useStore((s) => s.set);
  const addMenu = useMenu();

  const count = selection.length > 1 ? selection.length : items.filter((i) => i.info).length;
  const destLabel = !exportSettings
    ? ""
    : exportSettings.location === "custom"
      ? exportSettings.folder
        ? shortPath(exportSettings.folder, 34)
        : t("Choose an output folder…")
      : exportSettings.location === "subfolder"
        ? t("“AlphaForge” folder next to originals")
        : t("Same folder as originals");
  const p = exportRun.progress;
  const pct = p ? Math.round(((p.done + (p.current != null ? Math.min(p.fraction, 0.99) : 0)) / Math.max(1, p.total)) * 100) : 0;

  return (
    <aside className="left" aria-label={t("Files")}>
      <div className="panel-head">
        <span className="panel-title">
          {t("Files")}<span className="count num">{items.length || ""}</span>
        </span>
        <span className="spacer" />
        <button className="icon-btn" aria-label={t("Add")} data-tip={t("Add images (Ctrl+O)")} onClick={(e) => addMenu.open(e.currentTarget)}>
          <Plus size={17} />
        </button>
        {addMenu.anchor && (
          <Menu
            anchor={addMenu.anchor}
            onClose={addMenu.close}
            align="right"
            items={[
              { label: t("Add images…"), icon: <ImagePlus size={15} />, hint: "Ctrl+O", onClick: openFiles },
              { label: t("Add folder…"), icon: <FolderOpen size={15} />, hint: "Ctrl+Shift+O", onClick: openFolder },
              { label: t("Paste from clipboard"), icon: <ClipboardPaste size={15} />, hint: "Ctrl+V", onClick: paste },
              { sep: true },
              { label: t("Clear list"), icon: <Trash2 size={15} />, danger: true, disabled: !items.length || exportRun.running, onClick: clearAll },
            ]}
          />
        )}
      </div>
      <div className="filelist" role="listbox" aria-label={t("Files")} aria-multiselectable>
        {items.map((it) => (
          <Row key={it.id} it={it} selected={it.id === selectedId} multi={selection.includes(it.id) && selection.length > 1} />
        ))}
        {!items.length && (
          <div className="faint" style={{ padding: "12px 8px", fontSize: 12, lineHeight: 1.5 }}>
            {t("No images yet. Drag files or folders anywhere into the window.")}
          </div>
        )}
      </div>
      <div className="left-foot">
        {exportRun.running && p ? (
          <div className="export-progress" role="status">
            <div className="line">
              <span>
                {t("Exporting")} <b className="num">{Math.min(p.done + 1, p.total)}</b> {t("of")} <b className="num">{p.total}</b>
              </span>
              <span className="num">{pct}%</span>
            </div>
            <div className="progress">
              <div style={{ width: `${pct}%` }} />
            </div>
            <div className="line">
              <span className="faint ellipsis">{t(p.label)}</span>
            </div>
            <button className="btn block" onClick={() => api.cancelExport()}>
              {t("Cancel")}
            </button>
          </div>
        ) : (
          <>
            <button className="btn primary lg block" disabled={!count} onClick={() => startExport()}>
              {selection.length > 1 ? t("Export {n} selected", { n: count }) : count === 1 ? t("Export image") : count ? t("Export all {n}", { n: count }) : t("Export")}
            </button>
            <button className="dest" onClick={() => set({ exportDialogOpen: true })} data-tip={t("Output folder & file names")} data-tip-pos="top">
              <FolderOpen size={13} style={{ flex: "none" }} />
              <span>{destLabel}</span>
            </button>
          </>
        )}
      </div>
    </aside>
  );
}
