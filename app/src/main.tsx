import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import * as actions from "./actions";
import { bootstrap } from "./actions";
import { toast, useStore } from "./store";
import { t, tr } from "./i18n";
import "./styles.css";

// Any backend call whose failure is not handled where it is made still tells the user.
window.addEventListener("unhandledrejection", (e) => {
  e.preventDefault();
  toast({ kind: "error", title: t("Something went wrong"), body: tr(String(e.reason ?? "")) });
});

// Resolve the theme before the first paint (the window is shown by the backend afterwards).
document.documentElement.dataset.theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

bootstrap().catch((e) => {
  const pre = document.createElement("pre");
  pre.style.cssText = "padding:24px;color:#f2575c;white-space:pre-wrap;font:13px Consolas,monospace";
  pre.textContent = `AlphaForge failed to start:\n${String(e)}`;
  document.body.replaceChildren(pre);
});

// Test hook for automated UI checks in development builds only.
if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__af = { actions, store: useStore };
