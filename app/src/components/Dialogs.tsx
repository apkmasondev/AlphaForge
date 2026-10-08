import { useEffect, useState } from "react";
import { Cpu, Download, FolderOpen, HardDrive, Info, Lock, MonitorCog, Palette, RefreshCcw, Shield, Trash2, Zap } from "lucide-react";
import { api } from "../api";
import { applyTheme, installGpuPack, installModel, savePreset, updateSettings } from "../actions";
import { useStore, type SettingsTab } from "../store";
import type { Conflict, DevicePref, ExportSettings, Location, Theme } from "../types";
import { basename, fmtBytes, shortPath } from "../lib/format";
import { Dialog, Segmented, Switch } from "./ui";
import { resolveLang, setLang, t, tr } from "../i18n";

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------
export function SettingsDialog() {
  const tab = useStore((s) => s.settingsOpen);
  const set = useStore((s) => s.set);
  if (!tab) return null;
  const nav: { id: SettingsTab; label: string; icon: React.ReactNode }[] = [
    { id: "general", label: t("General ").trim(), icon: <Palette size={15} /> },
    { id: "ai", label: t("AI & GPU"), icon: <Zap size={15} /> },
    { id: "models", label: t("Models"), icon: <HardDrive size={15} /> },
    { id: "about", label: t("Privacy & about"), icon: <Shield size={15} /> },
  ];
  return (
    <Dialog title={t("Settings")} wide onClose={() => set({ settingsOpen: null })}>
      <div className="settings">
        <nav>
          {nav.map((n) => (
            <button key={n.id} className={tab === n.id ? "on" : ""} onClick={() => set({ settingsOpen: n.id })}>
              {n.icon}
              {n.label}
            </button>
          ))}
        </nav>
        <div className="pane">
          {tab === "general" && <GeneralPane />}
          {tab === "ai" && <AiPane />}
          {tab === "models" && <ModelsPane />}
          {tab === "about" && <AboutPane />}
        </div>
      </div>
    </Dialog>
  );
}

function Setting(props: { title: string; desc?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="setting">
      <div>
        <div className="s-title">{props.title}</div>
        {props.desc && <div className="s-desc">{props.desc}</div>}
      </div>
      <div>{props.children}</div>
    </div>
  );
}

function GeneralPane() {
  const s = useStore((st) => st.settings)!;
  return (
    <>
      <h3>{t("General ").trim()}</h3>
      <div className="group">
        <Setting title={t("Language")}>
          <Segmented<"system" | "en" | "pl">
            value={s.language ?? "system"}
            onChange={(language) => {
              setLang(resolveLang(language));
              updateSettings({ language });
            }}
            options={[
              { value: "system", label: t("System") },
              { value: "en", label: "English" },
              { value: "pl", label: "Polski" },
            ]}
          />
        </Setting>
        <Setting title={t("Theme")}>
          <Segmented<Theme>
            value={s.theme}
            onChange={(theme) => {
              updateSettings({ theme });
              applyTheme(theme);
            }}
            options={[
              { value: "system", label: t("System") },
              { value: "light", label: t("Light") },
              { value: "dark", label: t("Dark") },
            ]}
          />
        </Setting>
        <Setting title={t("Live preview")} desc={t("Re-process the selected image automatically when a setting changes.")}>
          <Switch checked={s.autoPreview} onChange={(autoPreview) => updateSettings({ autoPreview })} />
        </Setting>
        <Setting title={t("Memory for previews")} desc={t("Decoded images and AI results kept in RAM so changes are instant.")}>
          <select className="select" style={{ width: 130 }} value={s.cacheMb} onChange={(e) => updateSettings({ cacheMb: Number(e.target.value) })}>
            {[...new Set([512, 1024, 2048, 4096, 8192, s.cacheMb])].sort((a, b) => a - b).map((v) => (
              <option key={v} value={v}>
                {v >= 1024 ? `${v / 1024} GB` : `${v} MB`}
              </option>
            ))}
          </select>
        </Setting>
      </div>
      <div className="group">
        <div className="group-title">{t("Keyboard")}</div>
        <div className="kv" style={{ gridTemplateColumns: "200px 1fr" }}>
          {[
            [t("Add images / folder"), "Ctrl+O · Ctrl+Shift+O"],
            [t("Paste image or files"), "Ctrl+V"],
            [t("Copy result"), "Ctrl+C"],
            [t("Export"), "Ctrl+E"],
            [t("Original / Result / Compare"), "O · R · C"],
            [t("Refine cut-out (brush)"), t("B — then K keep, E erase, [ ] size")],
            [t("Fit / 100% / zoom"), "Ctrl+0 · Ctrl+1 · + −"],
            [t("Pan"), t("Drag · Space+drag · middle mouse")],
            [t("Previous / next image"), "↑ ↓"],
            [t("Remove from list"), "Delete"],
          ].map(([k, v]) => (
            <div key={k} style={{ display: "contents" }}>
              <dt>{k}</dt>
              <dd className="num">{v}</dd>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function AiPane() {
  const s = useStore((st) => st.settings)!;
  const runtime = useStore((st) => st.runtime);
  const pack = useStore((st) => st.gpuPack);
  const dl = useStore((st) => st.downloads["gpu-pack"]);
  const system = useStore((st) => st.system);
  const set = useStore((st) => st.set);
  const rt = runtime?.info;
  const gpu = rt?.gpu ?? system?.gpus[0] ?? null;
  const [busy, setBusy] = useState(false);
  const deviceChanged = rt && ((s.device === "cpu" && rt.cudaReady) || (s.device !== "cpu" && !rt.cudaReady && pack?.installed && rt.gpuSupported));
  return (
    <>
      <h3>{t("AI & GPU")}</h3>
      <div className="card">
        <div className="row">
          {rt?.cudaReady ? <Zap size={18} color="var(--success)" /> : <Cpu size={18} color="var(--accent-text)" />}
          <b>{rt?.cudaReady ? t("GPU acceleration is active") : t("AI runs on the CPU")}</b>
        </div>
        <div className="muted" style={{ fontSize: 12.5 }}>{rt ? tr(rt.gpuStatus) : t("Starting…")}</div>
        <dl className="kv">
          <dt>{t("Processor")}</dt>
          <dd>{system?.cpuName} · {system?.cpuCores} cores</dd>
          <dt>{t("Memory")}</dt>
          <dd>{system ? `${(system.ramTotalMb / 1024).toFixed(0)} GB` : "—"}</dd>
          <dt>{t("Graphics")}</dt>
          <dd>{gpu ? `${gpu.name} · ${(gpu.vramTotalMb / 1024).toFixed(0)} GB VRAM · driver ${gpu.driverVersion}` : t("No NVIDIA GPU detected")}</dd>
          {runtime?.loaded.length ? (
            <>
              <dt>{t("Loaded models")}</dt>
              <dd>{runtime.loaded.map((m) => `${m.template} (${m.device.toUpperCase()})`).join(", ")}</dd>
            </>
          ) : null}
        </dl>
      </div>

      <div className="group">
        <Setting title={t("Processing device")} desc={t("Auto uses the NVIDIA GPU when the GPU pack is installed and the model fits in its memory.")}>
          <Segmented<DevicePref>
            value={s.device}
            onChange={(device) => updateSettings({ device })}
            options={[
              { value: "auto", label: t("Auto") },
              { value: "gpu", label: "GPU", disabled: !rt?.gpuSupported },
              { value: "cpu", label: "CPU" },
            ]}
          />
        </Setting>
        {deviceChanged && (
          <div className="notice info">
            <Info size={15} />
            <span style={{ flex: 1 }}>{t("Restart AlphaForge to switch the AI runtime.")}</span>
            <button className="btn sm" onClick={() => api.restartApp()}>{t("Restart")}</button>
          </div>
        )}
        <Setting title={t("GPU acceleration pack")} desc={
          <>
            {t("NVIDIA CUDA 12 + cuDNN 9 runtime and ONNX Runtime GPU, downloaded from the official NVIDIA (PyPI) and Microsoft (GitHub) releases and verified file by file.")}
            {" "}{t("Download {a}, uses {b} on disk.", { a: fmtBytes(pack?.downloadBytes ?? 0), b: fmtBytes(pack?.diskBytes ?? 0) })}
            {!rt?.gpuSupported && t(" Requires an NVIDIA GPU with a current driver.")}
          </>
        }>
          {dl ? (
            <div className="col" style={{ width: 200 }}>
              <div className="progress"><div style={{ width: `${(dl.done / Math.max(1, dl.total)) * 100}%` }} /></div>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="faint num" style={{ fontSize: 12 }}>{fmtBytes(dl.done)} / {fmtBytes(dl.total)}</span>
                <button className="btn sm ghost" onClick={() => api.cancelDownload("gpu-pack")}>{t("Cancel")}</button>
              </div>
            </div>
          ) : pack?.installed ? (
            <div className="row">
              {!rt?.cudaReady && rt?.gpuSupported && <button className="btn sm primary" onClick={() => api.restartApp()}><RefreshCcw size={13} /> {t("Restart to use")}</button>}
              <button
                className="btn sm danger"
                disabled={busy}
                onClick={async () => {
                  setBusy(true);
                  try {
                    set({ gpuPack: await api.removeGpuPack() });
                  } catch (e) {
                    useStore.setState({ toasts: [...useStore.getState().toasts, { id: Date.now(), kind: "info", title: t("GPU pack"), body: tr(String(e)) }] });
                  }
                  setBusy(false);
                }}
              >
                <Trash2 size={13} /> {t("Remove")}
              </button>
            </div>
          ) : (
            <button className="btn primary" disabled={!rt?.gpuSupported} onClick={installGpuPack}>
              <Download size={14} /> {t("Install")}
            </button>
          )}
        </Setting>
        <Setting title={t("Free AI memory when idle")} desc={t("Unload models from RAM / VRAM after this long without use.")}>
          <select className="select" style={{ width: 130 }} value={s.unloadAfterMin} onChange={(e) => updateSettings({ unloadAfterMin: Number(e.target.value) })}>
            <option value={2}>{t("2 minutes")}</option>
            <option value={10}>{t("10 minutes")}</option>
            <option value={30}>{t("30 minutes")}</option>
            <option value={0}>{t("Never")}</option>
          </select>
        </Setting>
        <Setting title={t("Unload models now")} desc={t("Immediately frees the memory used by AI models and cached results.")}>
          <button className="btn sm" onClick={async () => { await api.unloadModels(); set({ runtime: await api.runtimeInfo() }); }}>{t("Unload")}</button>
        </Setting>
      </div>
    </>
  );
}

function ModelsPane() {
  const models = useStore((s) => s.models);
  const downloads = useStore((s) => s.downloads);
  const set = useStore((s) => s.set);
  const group = (kind: "background" | "upscale") => models.filter((m) => m.kind === kind);
  return (
    <>
      <h3>{t("Models")}</h3>
      <div className="notice info">
        <Shield size={15} />
        <span>
          {t("Only permissively licensed models are used (MIT, BSD-3-Clause) — free for commercial work. Downloads come straight from the authors’ official repositories at pinned versions and are checked against SHA-256 checksums before use. Model files are data (safetensors / ONNX); nothing is executed.")}
        </span>
      </div>
      {(["background", "upscale"] as const).map((k) => (
        <div className="group" key={k}>
          <div className="group-title">{k === "background" ? t("Background removal") : t("Upscaling")}</div>
          <div>
            {group(k).map((m) => {
              const dl = downloads[m.id];
              return (
                <div className="model-row" key={m.id}>
                  <div>
                    <div className="mr-name">
                      {t(m.name)}
                      <span className="badge">{m.license}</span>
                      {m.bundled ? <span className="badge success">{t("Built in")}</span> : m.installed ? <span className="badge success">{t("Installed")}</span> : null}
                    </div>
                    <div className="mr-meta">{m.family} · {t(m.tagline)}</div>
                    <div className="mr-meta num">
                      {t("~{g} s on GPU · ~{c} s on CPU", { g: m.gpuSecs < 1 ? m.gpuSecs.toFixed(1) : Math.round(m.gpuSecs), c: Math.round(m.cpuSecs) })} {m.kind === "upscale" ? t("per input megapixel") : t("per image")}
                      {!m.bundled && ` · ${fmtBytes(m.installed ? m.diskBytes : m.downloadBytes)}`}
                    </div>
                  </div>
                  <div>
                    {m.bundled ? null : dl ? (
                      <div className="col" style={{ width: 150 }}>
                        <div className="progress"><div style={{ width: `${(dl.done / Math.max(1, dl.total)) * 100}%` }} /></div>
                        <button className="btn sm ghost" onClick={() => api.cancelDownload(m.id)}>{t("Cancel")}</button>
                      </div>
                    ) : m.installed ? (
                      <button className="btn sm danger" onClick={async () => set({ models: await api.removeModel(m.id) })}>
                        <Trash2 size={13} /> {t("Remove")}
                      </button>
                    ) : (
                      <button className="btn sm" onClick={() => installModel(m.id)}>
                        <Download size={13} /> {fmtBytes(m.downloadBytes)}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </>
  );
}

function AboutPane() {
  const version = useStore((s) => s.version);
  const system = useStore((s) => s.system);
  return (
    <>
      <h3>{t("Privacy & about")}</h3>
      <div className="card">
        <div className="row">
          <Lock size={18} color="var(--success)" />
          <b>{t("Processing happens locally on this computer. Images are not uploaded.")}</b>
        </div>
        <ul className="muted" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.6, fontSize: 12.5 }}>
          <li>{t("No accounts, no telemetry, no analytics, no crash reporting.")}</li>
          <li>{t("The only network connections are downloads you start: optional AI models (Hugging Face) and the optional GPU pack (GitHub / PyPI).")}</li>
          <li>{t("Exported files contain no EXIF or GPS metadata from the originals.")}</li>
          <li>{t("Pasted images stay in memory and are written to disk only when you export them.")}</li>
        </ul>
      </div>
      <dl className="kv">
        <dt>{t("Version")}</dt>
        <dd>AlphaForge {version}</dd>
        <dt>{t("System")}</dt>
        <dd>{system?.os}</dd>
        <dt>{t("AI runtime")}</dt>
        <dd>ONNX Runtime 1.28.3 (CPU build bundled; CUDA 12 build in the GPU pack)</dd>
        <dt>{t("Models")}</dt>
        <dd>BiRefNet (MIT) by Peng Zheng et al. · Real-ESRGAN (BSD-3-Clause) by Xintao Wang et al.</dd>
      </dl>
      <div className="row">
        <button className="btn" onClick={() => api.openLicenses()}>
          <Info size={14} /> {t("Open-source licenses")}
        </button>
        <button className="btn" onClick={() => api.openDataFolder()}>
          <FolderOpen size={14} /> {t("App data folder")}
        </button>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
// Export settings
// ---------------------------------------------------------------------------------------------
export function ExportDialog() {
  const open = useStore((s) => s.exportDialogOpen);
  const settings = useStore((s) => s.settings);
  const set = useStore((s) => s.set);
  const item = useStore((s) => s.items.find((i) => i.id === s.selectedId) ?? s.items[0] ?? null);
  const fmt = useStore((s) => s.pipeline.output.format);
  const [draft, setDraft] = useState<ExportSettings | null>(null);
  useEffect(() => {
    if (open && settings) setDraft({ ...settings.export });
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!open || !draft) return null;
  const upd = (p: Partial<ExportSettings>) => setDraft({ ...draft, ...p });
  const ext = fmt === "jpeg" ? "jpg" : fmt === "same" ? (item?.name.split(".").pop() ?? "png") : fmt;
  const stem = item ? item.name.replace(/\.[^.]+$/, "") : "photo";
  const srcDir = item?.path ? item.path.slice(0, Math.max(item.path.lastIndexOf("\\"), item.path.lastIndexOf("/"))) : "Pictures";
  const dir = draft.location === "custom" ? draft.folder ?? t("(choose a folder)") : draft.location === "subfolder" ? `${srcDir}\\AlphaForge` : srcDir;
  const loc = (id: Location, title: string, desc: string) => (
    <button className={`radio-row${draft.location === id ? " on" : ""}`} onClick={() => upd({ location: id })} role="radio" aria-checked={draft.location === id}>
      <span className="radio" />
      <span>
        <div className="r-title">{t(title)}</div>
        <div className="r-desc">{desc}</div>
      </span>
    </button>
  );
  const save = () => {
    updateSettings({ export: draft });
    set({ exportDialogOpen: false });
  };
  return (
    <Dialog
      title={t("Output folder & file names")}
      icon={<FolderOpen size={18} className="faint" />}
      onClose={() => set({ exportDialogOpen: false })}
      footer={
        <>
          <button className="btn" onClick={() => set({ exportDialogOpen: false })}>{t("Cancel")}</button>
          <button className="btn primary" disabled={draft.location === "custom" && !draft.folder} onClick={save}>{t("Save")}</button>
        </>
      }
    >
      <div className="dialog-body">
        <div className="field">
          <div className="field-label">{t("Save to")}</div>
          <div className="radio-list" role="radiogroup">
            {loc("subfolder", "“AlphaForge” folder next to each original", t("Keeps results together without touching your originals (recommended)"))}
            {loc("sameFolder", "Same folder as the original", t("Results sit right next to the source files"))}
            {loc("custom", "A folder of my choice", draft.folder ? shortPath(draft.folder, 60) : t("All results in one place"))}
          </div>
          {draft.location === "custom" && (
            <div className="row">
              <button className="btn sm" onClick={async () => { const f = await api.chooseFolder(t("Choose output folder")); if (f) upd({ folder: f }); }}>
                <FolderOpen size={13} /> {draft.folder ? t("Change folder…") : t("Choose folder…")}
              </button>
              <label className="check" style={{ marginLeft: 8 }}>
                <Switch checked={draft.keepStructure} onChange={(keepStructure) => upd({ keepStructure })} />
                {t("Keep sub-folder structure of dropped folders")}
              </label>
            </div>
          )}
        </div>
        <div className="grid2">
          <div className="field">
            <div className="field-label">{t("Prefix")}</div>
            <input className="input" value={draft.prefix} maxLength={40} placeholder={t("e.g. web_")} onChange={(e) => upd({ prefix: e.target.value })} />
          </div>
          <div className="field">
            <div className="field-label">{t("Suffix")}</div>
            <input className="input" value={draft.suffix} maxLength={40} placeholder={t("e.g. _cutout")} onChange={(e) => upd({ suffix: e.target.value })} />
          </div>
        </div>
        <div className="field">
          <div className="field-label">{t("If a file with that name exists")}</div>
          <Segmented<Conflict>
            value={draft.conflict}
            onChange={(conflict) => upd({ conflict })}
            options={[
              { value: "rename", label: t("Add a number") },
              { value: "overwrite", label: t("Overwrite") },
              { value: "skip", label: t("Skip") },
            ]}
          />
        </div>
        <div className="field">
          <div className="field-label">{t("Preview")}</div>
          <div className="export-preview">
            {shortPath(dir, 70)}\<b>{draft.prefix.replace(/[<>:"/\\|?*]/g, "_")}{stem}{draft.suffix.replace(/[<>:"/\\|?*]/g, "_")}.{ext}</b>
          </div>
          <span className="faint" style={{ fontSize: 12 }}>{t("Original files are never overwritten. Pasted images go to Pictures\\AlphaForge unless you choose a folder.")}</span>
        </div>
      </div>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------------------------
// Save preset
// ---------------------------------------------------------------------------------------------
export function SavePresetDialog() {
  const open = useStore((s) => s.savePresetOpen);
  const presets = useStore((s) => s.presets);
  const presetId = useStore((s) => s.presetId);
  const set = useStore((s) => s.set);
  const current = presets.find((p) => p.id === presetId);
  const userCurrent = current && !current.builtin ? current : null;
  const [name, setName] = useState("");
  const [mode, setMode] = useState<"new" | "update">("new");
  useEffect(() => {
    if (open) {
      setName(userCurrent ? userCurrent.name : current ? t("{name} (custom)", { name: t(current.name) }) : t("My preset"));
      setMode(userCurrent ? "update" : "new");
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!open) return null;
  const submit = async () => {
    await savePreset(name, mode === "update" && userCurrent ? userCurrent.id : null);
    set({ savePresetOpen: false });
  };
  return (
    <Dialog
      title={t("Save preset")}
      icon={<MonitorCog size={18} className="faint" />}
      onClose={() => set({ savePresetOpen: false })}
      footer={
        <>
          <button className="btn" onClick={() => set({ savePresetOpen: false })}>{t("Cancel")}</button>
          <button className="btn primary" disabled={!name.trim()} onClick={submit}>{t("Save")}</button>
        </>
      }
    >
      <div className="dialog-body">
        {userCurrent && (
          <Segmented
            value={mode}
            onChange={setMode}
            options={[
              { value: "update", label: t("Update “{name}”", { name: basename(userCurrent.name) }) },
              { value: "new", label: t("Save as new") },
            ]}
          />
        )}
        <div className="field">
          <div className="field-label">{t("Name")}</div>
          <input className="input" autoFocus value={name} maxLength={60} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && name.trim() && submit()} />
        </div>
        <span className="faint" style={{ fontSize: 12 }}>{t("The preset stores all steps and output settings. It appears under “My presets”.")}</span>
      </div>
    </Dialog>
  );
}
