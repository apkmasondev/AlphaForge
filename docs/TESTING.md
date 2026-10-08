# Test report

Reference machine: Ryzen 7 5800H, 16 GB RAM, RTX 3060 Laptop 6 GB (driver 616.64), Windows 11.
Tests were run against the real application (dev build driven through the WebView2 DevTools protocol,
and the installed NSIS build), the `af-cli` harness and `cargo test`.

## Automated tests (`cargo test -p af-core -- --include-ignored`)

| Area | Tests |
|---|---|
| Masks | box blur, erosion/dilation, island removal, brush keep/erase/restore |
| Paths | sanitising (reserved names, invalid chars), safe joins (`..`, roots, UNC rejected) |
| Export | batch-unique names, folder structure, never overwrite source, relative folders rejected |
| Downloads | host allow-list incl. redirect/user-info tricks, low disk space, offline (DNS failure), checksum mismatch discards the file |
| Presets | JSON round-trip, no duplicate keys |
| Templates | Rust weight assembly is byte-identical to the build-time reference for all bundled models (fp16 + fp32) and bundled weight checksums match |

Also: AVIF size check before decoding (decompression bomb), palette PNG keeps opaque/transparent
pixels exact, panic guard, pipeline warnings, atomic overwrite. All 24 tests pass (`cargo test -p af-core
-- --include-ignored`).

## Audit regression (1.0.3)

| Suite | Result |
|---|---|
| Core pipeline via `af-cli` (trim, padding, 7 resize modes, fill, enhance, AI upscale/denoise, PNG/JPG/WebP/AVIF/Same, EXIF orientation + metadata stripping, 12 input variants, damaged files) | 72/73 — the one difference is RGB under fully transparent pixels of a 16-bit PNG (zeroed on purpose, invisible) |
| All 10 presets × 16 photos (3 background models for Transparent Asset: subject exactly 32 px from every edge) | 346/349 — Web Asset margin on a small photo (expected), Compress can grow an already small JPG by ~0.6 %, MPO is JPEG |
| UI flows in the running app (trim/reorder, every step type, JPG warning, presets, brush undo/redo/reset, compare, clipboard, export names/conflicts/structure, cancel, reprocess, language/theme) | pass |
| Concurrency (clear while loading, double export/download, remove during export) | pass |
| New fixes (duplicates in one drop, Ctrl+click, copy before preview is ready, size encode not blocking the next preview: 151 ms vs a PNG-max encode, Skip without processing: 0.25 s vs 14 s, 5000-file limit notice, Escape in number fields, Space on step switches) | pass |
| Memory soak (34 photos × 3 cycles) | no growth beyond allocator noise |
| Network monitor (full session incl. AI and export) | no connections |

## Image content

| Material | Result |
|---|---|
| Person, long hair (girl-2/3) | Clean; *Hair & fur* gives the softest strands |
| Curly hair / beard (painting) | Good with all BiRefNet models |
| Product (sneakers, bottle) | Clean edges, laces kept |
| Cars (2 photos) | Clean; mirrors and wheels kept |
| Animal fur (white dog on grey, white dog on green, cat, lion, tiger) | Good; colour decontamination removes most background tint; remaining tint on green-lit fur → *Hair & fur* model |
| Transparent / semi-transparent (glassware, dandelion seed head) | Glass kept by general models; dandelion gets real partial alpha |
| Subject on similar colours (chameleon on branches) | Correct with BiRefNet; U²-Net/ISNet fail (research only) |
| Very large image (6000 × 4000, 7058 × 5104) | Works; 1.6 s GPU incl. post-processing; resize to AVIF 0.8 s |
| Small image (480 × 360, 600 × 399) | Works |
| JPEG / PNG (8 & 16-bit RGBA) / WebP / AVIF input | Works (16-bit reduced to 8-bit, alpha preserved; AVIF via rav1d) |
| Pasted screenshot, files copied in Explorer | Works (`Pasted image.png`, CF_HDROP list) |

## Scenarios

| Scenario | Result |
|---|---|
| Batch 24 photos → Transparent Asset (Best quality, GPU) | 24/24 in 28.7 s |
| Batch 24 → Web Asset WebP | 24/24 in 21 s, 20.9 MB → 2.8 MB |
| Batch 50 (sub-folders, keep structure) → Transparent Asset | 50/50 in 55 s; peak 2.8 GB RAM, 4.2 GB VRAM, stable |
| Cancel during batch | Stops within ~2 s; finished files kept, others back to "ready" |
| Damaged files (0 bytes, text renamed to .png, truncated JPEG) | Per-item error, batch continues; truncated JPEG decodes partially |
| Non-image file in a dropped folder | Skipped with a notice |
| No GPU pack / device = CPU | Works; Auto picks Fast (≈4–5 s per image incl. post-processing) |
| GPU out of memory | Falls back to CPU for that image with a note (code path; not reproducible on demand) |
| Model not installed | Preview card with download / "Use Fast instead"; export blocked with Download action; in-app download 424 MB in 13 s, preview resumes |
| No internet / unreachable host | "No internet connection or the server is unreachable" (test `unreachable_host_reports_offline`) |
| Not enough disk space | Checked before every download and every write ("Not enough disk space: … MB needed") |
| Upscale result over 160 MP | Refused with an explanation before allocating |
| GPU pack install | 1.46 GB via range requests, every DLL hash-verified, 49.5 s |
| Production build (installed NSIS) | Strict CSP OK, images via `afimg://`, CUDA active, export works |

## Performance profile

| Stage | Time |
|---|---|
| App start → window | < 1 s (ONNX Runtime / CUDA load in background) |
| Model load (first use) | Fast 2.7 s, Best quality 5.5 s (GPU) |
| Background removal, GPU | 0.25 s (Fast) / 0.5 s (Best quality, Hair & fur) |
| Matte post-processing + decontamination, 24 MP | 1.3 s |
| Changing a refine slider (cached matte) | < 100 ms for typical photos |
| PNG encode 7 MP cut-out | 0.9 s (Fast) / 4 s (Balanced) / 11 s (Max) |
| WebP / AVIF / JPG encode 7 MP | 1.5 s / 2 s / 0.07 s |
| Upscale 4× General / Photo (per input MP, GPU) | 0.4 s / 5 s |
