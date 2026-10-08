# Security & privacy review

Scope: the shipped application (Tauri shell, `af-core`, web UI) and the optional downloads.

## Privacy

* No telemetry, analytics, crash reporting or update pings. ONNX Runtime's own telemetry is disabled at
  initialisation (`with_telemetry(false)`).
* WebView2 background networking (component updates, domain reliability, pings, sync) is switched off
  with `additionalBrowserArgs`. Measured: a full session (start, AI previews, copy, export of 10 files)
  opened no network connection from the app or any of its WebView2 processes. Logs contain no image
  names or paths.
* The only outbound connections are **user-initiated** downloads (model weights from Hugging Face, GPU
  pack from GitHub/PyPI). Image data is never part of any request.
* Pasted images live in memory and are written only when exported. Logs (`%LOCALAPPDATA%\AlphaForge\logs`)
  contain file counts, timings and errors — no pixel data — and are capped at 2 MB.
* Exported files contain no EXIF/GPS/maker metadata from the source (metadata is never copied).

## Downloads

| Control | Implementation |
|---|---|
| Transport | HTTPS only (rustls + Windows certificate store via platform verifier) |
| Host allow-list | `github.com`, `objects/release-assets.githubusercontent.com`, `files.pythonhosted.org`, `huggingface.co` + HF CDN / Xet hosts. Checked on **every redirect hop** (redirects are followed manually); user-info tricks rejected (`download::tests::url_allow_list`). |
| Integrity | SHA-256 pinned in the binary for every file (model safetensors = upstream LFS digest; GPU pack = per-DLL digest). Data is written to `*.part` and only renamed after verification; mismatches are deleted (`checksum_mismatch_discards_file`). |
| Pinning | Hugging Face downloads use a fixed commit revision; GitHub/PyPI URLs are immutable release artefacts. |
| Limits | Size known in advance, "more data than expected" aborts; disk space checked first. |
| Zip extraction | Only named entries are extracted to fixed file names (no path from the archive is used → no zip-slip); entry size and hash verified. |

## Code execution surface

* Model files are **safetensors** and **ONNX** (data). No pickle / `torch.load`, no Python, no scripts.
* The GPU pack consists of signed NVIDIA/Microsoft DLLs, verified by SHA-256 when installed ("Verify"
  re-checks every file) and loaded
  by absolute path with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, so a DLL
  placed in `PATH` or the working directory cannot be picked up instead.
* The VC++ runtime is deployed app-locally next to the executable.

## Web view hardening

* Strict CSP: `default-src 'self'`, no inline scripts, `connect-src` limited to the IPC channel, images
  only from the app, `data:`/`blob:` and the in-process `afimg:` protocol; `object-src 'none'`,
  `frame-ancestors 'none'`. `freezePrototype` enabled.
* Capabilities: only `core:default` plus window show/focus/theme. File dialogs, clipboard, opening folders
  and the file system are **not** exposed to JavaScript; they are narrow Rust commands.
* `reveal_path` only reveals an existing item in Explorer; `open_folder` refuses anything that is not a
  directory — the UI can never launch an executable.
* `afimg://` serves only in-memory images by numeric id; it cannot read arbitrary paths.

## Input handling

* Decoders run with limits (40 000 px per side, 200 MP, allocation cap) to stop decompression bombs; a
  2 GB file-size cap is applied before reading.
* Malformed files produce a per-item error and never crash the app (tested with empty, truncated and
  non-image files with image extensions).
* Folder scans skip symlinks/junctions, hidden folders and AlphaForge output folders, with depth (12) and
  count (5 000) limits.

## File system writes

* Output names: prefix/stem/suffix are sanitised (reserved device names, `<>:"/\|?*`, control characters,
  trailing dots/spaces, length). "Keep folder structure" joins only *normal* relative components —
  `..`, roots and drive prefixes are rejected (`paths::tests::joins_safely`).
* The source image is never overwritten, even with the "Overwrite" policy (`never_overwrites_source`);
  two inputs in one batch never write to the same file (`plans_and_renames_within_batch`).
* Writes are atomic (temp file in the target folder + rename) and check free space first.
* Settings/presets are written atomically; an unreadable file is kept as `.bak` instead of silently lost.

## Secrets

The application contains no API keys, tokens or credentials; downloads are anonymous.

## Known limitations

* The installer and executable are not code-signed (no certificate available in this build environment);
  Windows SmartScreen will show a warning until the build is signed.
* Updates are manual (the Tauri updater plugin needs a signed update feed, which requires hosting).
