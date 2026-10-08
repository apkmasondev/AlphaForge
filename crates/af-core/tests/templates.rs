//! The Rust weight assembler must produce exactly the bytes the build tool validated.

use std::path::PathBuf;

use af_core::ai::template::{assemble, sha256_bytes, Manifest};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../app/src-tauri/resources/models")
}

fn check(name: &str, variant: &str) {
    let dir = models_dir();
    let m = Manifest::load(&dir.join(format!("{name}.manifest.json"))).unwrap();
    let bundled = dir.join(m.source.bundled.as_ref().expect("bundled weights"));
    let v = &m.variants[variant];
    let blob = assemble(v, &bundled).unwrap();
    assert_eq!(blob.len() as u64, v.blob_size);
    assert_eq!(sha256_bytes(&blob), v.blob_sha256, "{name} {variant}");
}

#[test]
fn birefnet_lite_fp16_blob_matches_reference() {
    check("birefnet-lite", "fp16");
}

#[test]
fn realesrgan_blobs_match_reference() {
    for n in ["realesr-general-x4v3", "realesr-general-wdn-x4v3", "realesr-general-x4v3-dn50", "RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B"] {
        check(n, "fp16");
        check(n, "fp32");
    }
}

#[test]
fn bundled_weights_match_manifest_checksum() {
    let dir = models_dir();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path();
        if p.to_string_lossy().ends_with(".manifest.json") {
            let m = Manifest::load(&p).unwrap();
            if let (Some(b), Some(sha)) = (&m.source.bundled, &m.source.bundled_sha256) {
                assert_eq!(&af_core::ai::template::sha256_file(&dir.join(b)).unwrap(), sha, "{b}");
            }
        }
    }
}
