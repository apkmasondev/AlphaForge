import { ArrowRight, Lock, Timer } from "lucide-react";
import { useStore } from "../store";
import { fmtBytes, fmtDims, fmtMs, fmtSaved } from "../lib/format";
import { t, tr } from "../i18n";

export function StatusBar() {
  const item = useStore((s) => s.items.find((i) => i.id === s.selectedId) ?? null);
  const preview = useStore((s) => s.preview);
  const runtime = useStore((s) => s.runtime);
  const system = useStore((s) => s.system);
  const set = useStore((s) => s.set);

  const done = preview.done && preview.id === item?.id ? preview.done : null;
  const size = preview.size && preview.id === item?.id && preview.size.seq === preview.seq ? preview.size : null;
  const saved = size && item ? fmtSaved(item.size, size.bytes) : null;
  const usedGpu = done?.reports.some((r) => r.device === "cuda");
  const usedCpu = done?.reports.some((r) => r.device === "cpu");
  const rt = runtime?.info;
  const deviceLabel = !runtime
    ? t("Starting AI runtime…")
    : rt?.cudaReady
      ? `GPU · ${rt.gpu?.name.replace(/^NVIDIA\s+(GeForce\s+)?/, "") ?? "CUDA"}`
      : `CPU · ${(system?.cpuName ?? "").replace(/\s+with .*$/i, "").replace(/\s+\d+-Core Processor/i, "")}`;

  return (
    <footer className="statusbar">
      {item?.info ? (
        <>
          <span className="sb-item" data-tip={t("Original")} data-tip-pos="right">
            <span className="k">{t("Original")}</span>
            <b>{fmtDims(item.info.width, item.info.height)}</b>
            <span>{item.info.format}</span>
            <b>{fmtBytes(item.size)}</b>
          </span>
          <ArrowRight size={13} />
          <span className="sb-item">
            <span className="k">{t("Output")}</span>
            {done ? (
              <>
                <b>{fmtDims(done.width, done.height)}</b>
                <span>{size?.format ?? done.format}</span>
                <b>{size ? fmtBytes(size.bytes) : done.partial ? t("refining") : "…"}</b>
                {saved && <span style={{ color: saved.better ? "var(--success)" : "var(--warning)", fontWeight: 600 }}>{saved.text}</span>}
              </>
            ) : (
              <span>{preview.status === "running" ? t("processing…") : "—"}</span>
            )}
          </span>
          {done && (
            <>
              <span className="sb-sep" />
              <span className="sb-item" data-tip={t("Processing time for this preview")} data-tip-pos="top">
                <Timer size={13} />
                <b>{fmtMs(done.ms)}</b>
                {(usedGpu || usedCpu) && <span>{t("on {d}", { d: usedGpu ? "GPU" : "CPU" })}</span>}
              </span>
            </>
          )}
        </>
      ) : (
        <span className="sb-item">{useStore.getState().items.length ? t("Select an image") : t("Ready")}</span>
      )}
      <span className="spacer" />
      <button className="sb-item" onClick={() => set({ settingsOpen: "ai" })} data-tip={rt ? tr(rt.gpuStatus) : t("AI runtime")} data-tip-pos="top">
        <span className={`dot ${!runtime ? "warn" : rt?.cudaReady ? "gpu" : "cpu"}`} />
        <b>{deviceLabel}</b>
      </button>
      <span className="sb-sep" />
      <span className="sb-item" data-tip={t("Processing happens locally on this computer. Images are not uploaded.")} data-tip-pos="left">
        <Lock size={12} color="var(--success)" />
        {t("Local only")}
      </span>
    </footer>
  );
}
