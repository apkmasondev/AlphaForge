import { ClipboardPaste, FolderOpen, ImagePlus, Lock } from "lucide-react";
import { openFiles, openFolder, paste } from "../actions";
import { Kbd } from "./ui";
import { t } from "../i18n";

export function PrivacyNote() {
  return (
    <span className="privacy">
      <Lock size={13} />
      {t("Processing happens locally on this computer. Images are not uploaded.")}
    </span>
  );
}

export function EmptyState() {
  return (
    <div className="empty">
      <div className="dropzone">
        <div className="icon-wrap">
          <ImagePlus size={28} />
        </div>
        <h1>{t("Drop images or folders here")}</h1>
        <p>
          {t("or paste a screenshot with")} <Kbd>Ctrl</Kbd> <Kbd>V</Kbd>
        </p>
        <div className="actions">
          <button className="btn primary lg" onClick={openFiles}>
            <ImagePlus size={16} /> {t("Add images")}
          </button>
          <button className="btn lg" onClick={openFolder}>
            <FolderOpen size={16} /> {t("Add folder")}
          </button>
          <button className="btn lg ghost" onClick={paste}>
            <ClipboardPaste size={16} /> {t("Paste")}
          </button>
        </div>
        <div className="formats">{t("JPG · PNG · WebP · AVIF · BMP · GIF · TIFF — up to 200 megapixels")}</div>
        <PrivacyNote />
      </div>
    </div>
  );
}
