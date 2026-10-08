use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// User-facing error. `Display` strings are shown in the UI, so keep them short and actionable.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cancelled")]
    Cancelled,
    #[error("Unsupported or damaged image file{}", .0.as_ref().map(|s| format!(": {s}")).unwrap_or_default())]
    Decode(Option<String>),
    #[error("Image is too large ({0} × {1} px). The limit is {2} megapixels.")]
    TooLarge(u32, u32, u32),
    #[error("Could not encode {0}: {1}")]
    Encode(&'static str, String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("Cannot access {}: {}", .0.display(), .1)]
    Path(PathBuf, String),
    #[error("Not enough disk space: {needed_mb} MB needed, {free_mb} MB free on the target drive")]
    DiskSpace { needed_mb: u64, free_mb: u64 },
    #[error("Model \"{0}\" is not installed")]
    ModelMissing(String),
    #[error("AI runtime error: {0}")]
    Runtime(String),
    #[error("Out of GPU memory while running {0}")]
    GpuOom(String),
    #[error("Download failed: {0}")]
    Download(String),
    #[error("File verification failed for {0} (checksum mismatch). The download was discarded.")]
    Checksum(String),
    #[error("No internet connection or the server is unreachable ({0})")]
    Offline(String),
    #[error("Invalid settings: {0}")]
    Invalid(String),
    /// A safety limit was hit (result too large, ...). The message is complete on its own.
    #[error("{0}")]
    Limit(String),
}

impl Error {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Error::Cancelled)
    }
    pub fn decode(msg: impl Into<String>) -> Self {
        Error::Decode(Some(msg.into()))
    }
    pub fn runtime(e: impl std::fmt::Display) -> Self {
        let s = e.to_string();
        if s.contains("terminate flag") || s.contains("Exiting due to terminate") {
            return Error::Cancelled;
        }
        Error::Runtime(s)
    }
}

impl<R> From<ort::Error<R>> for Error {
    fn from(e: ort::Error<R>) -> Self {
        let s = e.to_string();
        if s.contains("terminate flag") || s.contains("Exiting due to terminate") {
            Error::Cancelled
        } else if s.contains("BFCArena") || s.contains("out of memory") || s.contains("CUDA failure 2") || s.contains("cudaErrorMemoryAllocation") {
            Error::GpuOom(s.chars().take(200).collect())
        } else {
            Error::Runtime(s)
        }
    }
}

impl serde::Serialize for Error {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}
