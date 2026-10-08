import { useEffect } from "react";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Upload } from "lucide-react";
import {
  addPaths,
  copyResult,
  enterMask,
  exitMask,
  openFiles,
  openFolder,
  paste,
  redoStroke,
  removeItems,
  selectRelative,
  startExport,
  undoStroke,
} from "./actions";
import { S, useStore, viewerCmd } from "./store";
import { TopBar } from "./components/TopBar";
import { FileList } from "./components/FileList";
import { Viewer } from "./components/Viewer";
import { PipelinePanel } from "./components/PipelinePanel";
import { StatusBar } from "./components/StatusBar";
import { ExportDialog, SavePresetDialog, SettingsDialog } from "./components/Dialogs";
import { Toasts } from "./components/ui";
import { resolveLang, setLang, t } from "./i18n";

function isTyping(e: KeyboardEvent) {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = S();
      const dialogOpen = s.settingsOpen || s.exportDialogOpen || s.savePresetOpen;
      if (dialogOpen || isTyping(e)) return;
      const k = e.key.toLowerCase();
      const ctrl = e.ctrlKey || e.metaKey;
      if (ctrl && k === "o") {
        e.preventDefault();
        e.shiftKey ? openFolder() : openFiles();
      } else if (ctrl && k === "v") {
        e.preventDefault();
        paste();
      } else if (ctrl && k === "c") {
        if (window.getSelection()?.toString()) return;
        e.preventDefault();
        copyResult();
      } else if (ctrl && k === "e") {
        e.preventDefault();
        startExport();
      } else if (ctrl && k === "0") {
        e.preventDefault();
        viewerCmd("fit");
      } else if (ctrl && k === "1") {
        e.preventDefault();
        viewerCmd("actual");
      } else if (ctrl && k === "z" && s.mask.active) {
        e.preventDefault();
        e.shiftKey ? redoStroke() : undoStroke();
      } else if (ctrl && k === "y" && s.mask.active) {
        e.preventDefault();
        redoStroke();
      } else if (ctrl && k === "a") {
        e.preventDefault();
        useStore.setState({ selection: s.items.map((i) => i.id) });
      } else if (ctrl) {
        return;
      } else if (k === "+" || k === "=") viewerCmd("zoom-in");
      else if (k === "-") viewerCmd("zoom-out");
      else if (k === "arrowdown" || k === "arrowright") {
        e.preventDefault();
        selectRelative(1);
      } else if (k === "arrowup" || k === "arrowleft") {
        e.preventDefault();
        selectRelative(-1);
      } else if (k === "delete") {
        removeItems(s.selection.length > 1 ? s.selection : s.selectedId != null ? [s.selectedId] : []);
      } else if (k === "escape") {
        if (s.mask.active) exitMask();
        else if (s.selection.length > 1) useStore.setState({ selection: [] });
      } else if (k === "b") {
        s.mask.active ? exitMask() : enterMask();
      } else if (s.mask.active && (k === "k" || k === "e" || k === "a")) {
        useStore.setState({ mask: { ...s.mask, tool: k === "k" ? "keep" : k === "e" ? "erase" : "restore" } });
      } else if (s.mask.active && (k === "[" || k === "]")) {
        const f = k === "]" ? 1.2 : 1 / 1.2;
        useStore.setState({ mask: { ...s.mask, size: Math.round(Math.max(4, Math.min(600, s.mask.size * f))) } });
      } else if (k === "o") useStore.setState({ viewMode: "original" });
      else if (k === "r") useStore.setState({ viewMode: "result" });
      else if (k === "c" && !s.mask.active) useStore.setState({ viewMode: "compare" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

function useDragDrop() {
  useEffect(() => {
    let un: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((e) => {
        const p = e.payload;
        if (p.type === "enter" || p.type === "over") {
          if (!S().dragging) useStore.setState({ dragging: true });
        } else if (p.type === "leave") {
          useStore.setState({ dragging: false });
        } else if (p.type === "drop") {
          useStore.setState({ dragging: false });
          if (p.paths.length) addPaths(p.paths);
        }
      })
      .then((u) => (un = u));
    return () => un?.();
  }, []);
}

function DropOverlay() {
  const dragging = useStore((s) => s.dragging);
  if (!dragging) return null;
  return (
    <div className="drop-overlay">
      <div className="box">
        <Upload size={28} />
        {t("Drop to add images")}
        <span>{t("Files and whole folders are welcome")}</span>
      </div>
    </div>
  );
}

export function App() {
  const ready = useStore((s) => s.ready);
  const lw = useStore((s) => s.settings?.leftPanelWidth ?? 272);
  const rw = useStore((s) => s.settings?.rightPanelWidth ?? 340);
  const language = useStore((s) => s.settings?.language ?? "system");
  setLang(resolveLang(language));
  useShortcuts();
  useDragDrop();
  if (!ready) return null;
  return (
    <div key={language} className="app" style={{ ["--left-w" as string]: `${lw}px`, ["--right-w" as string]: `${rw}px` }} onContextMenu={(e) => {
      const t = e.target as HTMLElement;
      if (!(t.tagName === "INPUT" || t.tagName === "TEXTAREA")) e.preventDefault();
    }}>
      <TopBar />
      <FileList />
      <main className="main">
        <Viewer />
      </main>
      <PipelinePanel />
      <StatusBar />
      <SettingsDialog />
      <ExportDialog />
      <SavePresetDialog />
      <DropOverlay />
      <Toasts />
    </div>
  );
}
