//! Model templates: an ONNX graph whose weights live in an external blob that is assembled
//! in memory from the model author's original `.safetensors` file.
//!
//! See `tools/build_bg_template.py` for how templates are produced.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub name: String,
    pub arch: String,
    pub input: InputSpec,
    pub output: OutputSpec,
    pub source: SourceSpec,
    pub variants: HashMap<String, Variant>,
    #[serde(default)]
    pub scale: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputSpec {
    pub name: String,
    /// Fixed (width, height) or `None` for fully dynamic inputs (super-resolution).
    #[serde(default)]
    pub size: Option<[u32; 2]>,
    #[serde(default = "zero3")]
    pub mean: [f32; 3],
    #[serde(default = "one3")]
    pub std: [f32; 3],
}
fn zero3() -> [f32; 3] {
    [0.0; 3]
}
fn one3() -> [f32; 3] {
    [1.0; 3]
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputSpec {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SourceSpec {
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub url: String,
    pub sha256: String,
    pub size: u64,
    /// File name of a copy shipped with the application (resources/models), if any.
    #[serde(default)]
    pub bundled: Option<String>,
    #[serde(default)]
    pub bundled_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Variant {
    pub blob: String,
    pub blob_size: u64,
    pub tensors: Vec<TensorEntry>,
    #[serde(default)]
    pub blob_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub src: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub nbytes: u64,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
        let m: Manifest = serde_json::from_str(&s).map_err(|e| Error::Runtime(format!("bad manifest {}: {e}", path.display())))?;
        if m.format != 1 {
            return Err(Error::Runtime(format!("unsupported manifest format {}", m.format)));
        }
        Ok(m)
    }
}

/// Build the external-weights blob for one variant from a safetensors file.
pub fn assemble(variant: &Variant, weights: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(weights).map_err(|e| Error::Path(weights.to_path_buf(), e.to_string()))?;
    // SAFETY: read-only mapping of a file we own; it is not modified while mapped.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::Path(weights.to_path_buf(), e.to_string()))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| Error::Runtime(format!("invalid weights file: {e}")))?;
    let mut blob = vec![0u8; variant.blob_size as usize];
    for t in &variant.tensors {
        let view = st.tensor(&t.src).map_err(|_| Error::Runtime(format!("weights file lacks tensor {}", t.src)))?;
        let numel: u64 = t.shape.iter().product::<u64>().max(1);
        let src_numel: u64 = view.shape().iter().map(|&d| d as u64).product::<u64>().max(1);
        if numel != src_numel {
            return Err(Error::Runtime(format!("shape mismatch for {}", t.src)));
        }
        let elem = match t.dtype.as_str() {
            "f32" => 4,
            "f16" | "bf16" => 2,
            "i64" => 8,
            _ => 0,
        };
        // A damaged manifest must give an error, not an out-of-bounds panic.
        if t.nbytes != numel * elem || t.offset.checked_add(t.nbytes).is_none_or(|end| end > variant.blob_size) {
            return Err(Error::Runtime(format!("bad manifest entry for {}", t.src)));
        }
        let dst = &mut blob[t.offset as usize..(t.offset + t.nbytes) as usize];
        convert(view.dtype(), view.data(), &t.dtype, dst).map_err(|e| Error::Runtime(format!("{}: {e}", t.src)))?;
    }
    Ok(blob)
}

fn convert(src: Dtype, data: &[u8], dst_dtype: &str, out: &mut [u8]) -> std::result::Result<(), String> {
    match (src, dst_dtype) {
        (Dtype::F32, "f32") | (Dtype::F16, "f16") | (Dtype::I64, "i64") | (Dtype::BF16, "bf16") => {
            if data.len() != out.len() {
                return Err("size mismatch".into());
            }
            out.copy_from_slice(data);
        }
        (Dtype::F32, "f16") => {
            for (i, c) in data.chunks_exact(4).enumerate() {
                let v = f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        (Dtype::F16, "f32") => {
            for (i, c) in data.chunks_exact(2).enumerate() {
                let v = f16::from_le_bytes([c[0], c[1]]).to_f32();
                out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        (Dtype::BF16, "f32") => {
            for (i, c) in data.chunks_exact(2).enumerate() {
                let v = bf16::from_le_bytes([c[0], c[1]]).to_f32();
                out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        (Dtype::BF16, "f16") => {
            for (i, c) in data.chunks_exact(2).enumerate() {
                let v = f16::from_f32(bf16::from_le_bytes([c[0], c[1]]).to_f32());
                out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        (s, d) => return Err(format!("unsupported conversion {s:?} -> {d}")),
    }
    Ok(())
}

/// SHA-256 of a file, hex encoded (streamed; used to verify downloads and bundled weights).
pub fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

pub fn sha256_bytes(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}
