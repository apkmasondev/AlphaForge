//! `af-cli` — headless harness around `af-core` used for tests, profiling and scripting.
//!
//! ```text
//! af-cli info
//! af-cli run  --preset web-asset [--model fast|quality|hair] [--device auto|cpu|gpu] --out DIR FILES...
//! af-cli run  --pipeline pipeline.json --out DIR FILES...
//! af-cli model install bg-quality
//! af-cli gpupack status|install|verify|remove
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use af_core::ai::{self, runtime, DevicePref, Engine};
use af_core::export::{ExportSettings, Location, Planned, Planner, SourceRef};
use af_core::pipeline::{self, presets, BgModel, ExecContext, Pipeline, Stage, StageCache, Step};
use af_core::{imageio, CancelToken};

fn resources() -> PathBuf {
    if let Ok(p) = std::env::var("AF_RESOURCES") {
        return PathBuf::from(p);
    }
    let exe = std::env::current_exe().unwrap();
    for anc in exe.ancestors() {
        let c = anc.join("app").join("src-tauri").join("resources");
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("resources")
}

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".into())).join("AlphaForge")
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn init(pref: DevicePref) -> Engine {
    let res = resources();
    let rt = runtime::init(&runtime::RuntimePaths { cpu_ort_dir: res.join("onnxruntime"), gpu_pack_dir: data_dir().join("runtime").join("cuda12") }, pref).expect("runtime");
    eprintln!("runtime: {} | cuda_ready={} | {}", rt.ort_path.display(), rt.cuda_ready, rt.gpu_status);
    Engine::new(res.join("models"), data_dir().join("models"), pref)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let pref = match arg(&args, "--device").as_deref() {
        Some("cpu") => DevicePref::Cpu,
        Some("gpu") => DevicePref::Gpu,
        _ => DevicePref::Auto,
    };
    match cmd {
        "info" => {
            let e = init(pref);
            println!("{}", serde_json::to_string_pretty(&af_core::hw::system_info()).unwrap());
            for m in e.status() {
                println!("{:<12} installed={} {}", m.spec.id, m.installed, m.spec.family);
            }
        }
        "model" => {
            let e = init(pref);
            let id = args.get(3).expect("model id");
            let spec = ai::catalog::get(id).expect("unknown model");
            match args.get(2).map(String::as_str) {
                Some("install") => {
                    let c = CancelToken::new();
                    let t = Instant::now();
                    e.install(spec, &c, &|d, t| eprint!("\r{:.1}/{:.1} MB   ", d as f64 / 1e6, t as f64 / 1e6)).expect("install");
                    eprintln!("\ninstalled in {:.1}s", t.elapsed().as_secs_f32());
                }
                Some("remove") => e.remove(spec).expect("remove"),
                _ => eprintln!("model install|remove <id>"),
            }
        }
        "gpupack" => {
            let dir = data_dir().join("runtime").join("cuda12");
            match args.get(2).map(String::as_str) {
                Some("install") => {
                    let c = CancelToken::new();
                    let t = Instant::now();
                    ai::gpupack::install(&dir, &c, &|d, tot, l| eprint!("\r{:>7.1}/{:.1} MB {l:<40}", d as f64 / 1e6, tot as f64 / 1e6)).expect("install");
                    eprintln!("\ninstalled in {:.1}s", t.elapsed().as_secs_f32());
                }
                Some("verify") => {
                    ai::gpupack::verify(&dir).expect("verify");
                    println!("ok");
                }
                Some("remove") => ai::gpupack::uninstall(&dir).unwrap(),
                _ => println!("{}", serde_json::to_string_pretty(&ai::gpupack::status(&dir)).unwrap()),
            }
        }
        "run" => run(&args, pref),
        "encbench" => {
            for f in args.iter().skip(2) {
                let d = imageio::decode_file(Path::new(f)).expect("decode");
                for (label, o) in [
                    ("png-fast", imageio::EncodeOptions { png_level: imageio::PngLevel::Fast, ..Default::default() }),
                    ("png-balanced", imageio::EncodeOptions { png_level: imageio::PngLevel::Balanced, ..Default::default() }),
                    ("png-max", imageio::EncodeOptions { png_level: imageio::PngLevel::Max, ..Default::default() }),
                    ("png-256c", imageio::EncodeOptions { png_colors: Some(256), png_level: imageio::PngLevel::Fast, ..Default::default() }),
                    ("webp-85", imageio::EncodeOptions { format: imageio::Format::Webp, quality: 85, ..Default::default() }),
                    ("webp-lossless", imageio::EncodeOptions { format: imageio::Format::Webp, lossless: true, ..Default::default() }),
                    ("avif-60", imageio::EncodeOptions { format: imageio::Format::Avif, quality: 60, ..Default::default() }),
                    ("jpg-85", imageio::EncodeOptions { format: imageio::Format::Jpeg, quality: 85, ..Default::default() }),
                ] {
                    let t = Instant::now();
                    let b = imageio::encode(&d.image, &o).expect("enc");
                    println!("{:<28} {:<14} {:>8} KB {:>6} ms", Path::new(f).file_name().unwrap().to_string_lossy(), label, b.len() / 1024, t.elapsed().as_millis());
                }
            }
        }
        _ => eprintln!("usage: af-cli info | run | model | gpupack  (see source header)"),
    }
}

fn run(args: &[String], pref: DevicePref) {
    let engine = init(pref);
    let mut pipe: Pipeline = if let Some(p) = arg(args, "--pipeline") {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    } else {
        let id = arg(args, "--preset").unwrap_or_else(|| "transparent-asset".into());
        presets::builtin().into_iter().find(|p| p.id == id).expect("unknown preset").pipeline
    };
    if let Some(m) = arg(args, "--model") {
        let bm = match m.as_str() {
            "fast" => BgModel::Fast,
            "quality" => BgModel::Quality,
            "hair" => BgModel::Hair,
            _ => BgModel::Auto,
        };
        for s in pipe.steps.iter_mut() {
            if let Step::RemoveBackground { model, .. } = &mut s.step {
                *model = bm;
            }
        }
    }
    for w in pipe.warnings() {
        eprintln!("warning: {w}");
    }
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| "out".into()));
    let settings = ExportSettings { location: Location::Custom, folder: Some(out.clone()), ..Default::default() };
    let mut planner = Planner::new(&settings, out.clone());
    let files: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect::<Vec<_>>();
    // skip values that belong to flags
    let flag_vals: Vec<String> = ["--pipeline", "--preset", "--model", "--out", "--device"].iter().filter_map(|f| arg(args, f)).collect();
    let files: Vec<&String> = files.into_iter().filter(|f| !flag_vals.contains(f)).collect();
    let cache = StageCache::new(1 << 30);
    let cancel = CancelToken::new();
    let total = Instant::now();
    for (i, f) in files.iter().enumerate() {
        let t0 = Instant::now();
        let path = Path::new(f.as_str());
        let dec = match imageio::decode_file(path) {
            Ok(d) => d,
            Err(e) => {
                println!("{f}: ERROR {e}");
                continue;
            }
        };
        let t_dec = t0.elapsed().as_millis();
        let ctx = ExecContext { engine: &engine, cache: &cache, cancel: &cancel, item_key: i as u64 + 1, strokes: &[], progress: &|_, _, _| {}, stage: Stage::Final };
        let res = match pipeline::run(&pipe, Arc::new(dec.image), &ctx) {
            Ok(r) => r,
            Err(e) => {
                println!("{f}: ERROR {e}");
                continue;
            }
        };
        let has_alpha = !imageio::is_opaque(&res.image);
        let fmt = pipe.output.resolve(dec.info.format, has_alpha);
        let t1 = Instant::now();
        let bytes = imageio::encode(&res.image, &pipe.output.options_for(fmt)).expect("encode");
        let t_enc = t1.elapsed().as_millis();
        let target = match planner.plan(&SourceRef { path: Some(path), name: f, root: None }, fmt.ext()).unwrap() {
            Planned::Write(p) => p,
            Planned::Skip(p) => {
                println!("skip {}", p.display());
                continue;
            }
        };
        af_core::export::write_atomic(&target, &bytes).unwrap();
        let steps: Vec<String> = res.reports.iter().map(|r| format!("{}:{}ms{}{}", r.kind, r.ms, r.device.map(|d| format!("@{d:?}")).unwrap_or_default(), r.note.as_ref().map(|n| format!(" ({n})")).unwrap_or_default())).collect();
        println!(
            "{} {}x{} -> {}x{} {} {}KB | decode {}ms | {} | encode {}ms | total {}ms",
            path.file_name().unwrap().to_string_lossy(),
            dec.info.width,
            dec.info.height,
            res.image.width(),
            res.image.height(),
            fmt.label(),
            bytes.len() / 1024,
            t_dec,
            steps.join(" "),
            t_enc,
            t0.elapsed().as_millis()
        );
    }
    eprintln!("all done in {:.1}s; loaded models: {:?}", total.elapsed().as_secs_f32(), engine.loaded());
}
