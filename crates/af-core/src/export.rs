//! Output naming, conflict handling and atomic writes.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::paths::{relative_dir, safe_join, same_path, sanitize_component, sanitize_stem};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum Location {
    /// `<source folder>/AlphaForge/` (default, never touches originals).
    #[default]
    Subfolder,
    /// Next to the source file.
    SameFolder,
    /// A folder chosen by the user.
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Conflict {
    /// Add " (2)", " (3)", ... (default).
    #[default]
    Rename,
    Overwrite,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ExportSettings {
    pub location: Location,
    pub folder: Option<PathBuf>,
    pub prefix: String,
    pub suffix: String,
    pub conflict: Conflict,
    /// Recreate the dropped folder structure below the output folder.
    pub keep_structure: bool,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self { location: Location::Subfolder, folder: None, prefix: String::new(), suffix: String::new(), conflict: Conflict::Rename, keep_structure: true }
    }
}

/// What we know about a source for naming purposes.
pub struct SourceRef<'a> {
    /// File path (None for pasted images).
    pub path: Option<&'a Path>,
    /// Display name, used as stem for pasted images.
    pub name: &'a str,
    /// Folder the user dropped (for keep-structure).
    pub root: Option<&'a Path>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Planned {
    Write(PathBuf),
    /// Target exists and the policy is Skip.
    Skip(PathBuf),
}

/// Plans output paths for a batch; remembers paths it handed out so two inputs never write to
/// the same file within one run, even with the Overwrite policy.
pub struct Planner<'a> {
    pub settings: &'a ExportSettings,
    /// Fallback folder for sources without a path (pasted images), e.g. Pictures/AlphaForge.
    pub fallback_dir: PathBuf,
    taken: HashSet<String>,
}

impl<'a> Planner<'a> {
    pub fn new(settings: &'a ExportSettings, fallback_dir: PathBuf) -> Self {
        Self { settings, fallback_dir, taken: HashSet::new() }
    }

    pub fn dir_for(&self, src: &SourceRef) -> Result<PathBuf> {
        let s = self.settings;
        let base = match (s.location, src.path.and_then(|p| p.parent())) {
            (Location::Custom, _) => {
                let f = s.folder.clone().ok_or_else(|| Error::Invalid("choose an output folder".into()))?;
                if !f.is_absolute() {
                    return Err(Error::Invalid("the output folder must be a full path".into()));
                }
                f
            }
            (Location::Subfolder, Some(dir)) => dir.join("AlphaForge"),
            (Location::SameFolder, Some(dir)) => dir.to_path_buf(),
            (_, None) => self.fallback_dir.clone(),
        };
        if s.location == Location::Custom && s.keep_structure {
            if let (Some(root), Some(path)) = (src.root, src.path) {
                if let Some(rel) = relative_dir(root, path) {
                    let top = root.file_name().map(PathBuf::from).unwrap_or_default();
                    return safe_join(&base, &top.join(rel));
                }
            }
        }
        Ok(base)
    }

    pub fn plan(&mut self, src: &SourceRef, ext: &str) -> Result<Planned> {
        let dir = self.dir_for(src)?;
        let stem_src = src.path.and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| src.name.to_string());
        let stem = format!("{}{}{}", sanitize_component(&self.settings.prefix), sanitize_stem(&stem_src), sanitize_component(&self.settings.suffix));
        let stem = sanitize_stem(&stem);
        let candidate = |n: u32| -> PathBuf { if n <= 1 { dir.join(format!("{stem}.{ext}")) } else { dir.join(format!("{stem} ({n}).{ext}")) } };
        let key = |p: &Path| p.to_string_lossy().to_lowercase();
        let mut first = candidate(1);
        // Never overwrite the source image itself.
        let is_source = |p: &Path| src.path.map(|s| same_path(s, p)).unwrap_or(false);
        let taken_now = self.taken.contains(&key(&first)) || is_source(&first);
        if !taken_now && first.exists() {
            match self.settings.conflict {
                Conflict::Overwrite => {}
                Conflict::Skip => return Ok(Planned::Skip(first)),
                Conflict::Rename => first = self.next_free(&candidate, &is_source),
            }
        } else if taken_now {
            first = self.next_free(&candidate, &is_source);
        }
        self.taken.insert(key(&first));
        Ok(Planned::Write(first))
    }

    fn next_free(&self, candidate: &dyn Fn(u32) -> PathBuf, is_source: &dyn Fn(&Path) -> bool) -> PathBuf {
        let mut n = 2;
        loop {
            let c = candidate(n);
            if !c.exists() && !self.taken.contains(&c.to_string_lossy().to_lowercase()) && !is_source(&c) {
                return c;
            }
            n += 1;
        }
    }
}

/// Write via a temporary file in the same folder, then rename (no half-written outputs).
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let dir = path.parent().ok_or_else(|| Error::Invalid("bad output path".into()))?;
    std::fs::create_dir_all(dir).map_err(|e| Error::Path(dir.to_path_buf(), e.to_string()))?;
    crate::hw::ensure_space(dir, data.len() as u64)?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".aftmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, data).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
    if path.exists() {
        std::fs::remove_file(path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            Error::Path(path.to_path_buf(), e.to_string())
        })?;
    }
    std::fs::rename(&tmp, path).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_and_renames_within_batch() {
        let dir = tempfile::tempdir().unwrap();
        let s = ExportSettings { location: Location::Custom, folder: Some(dir.path().to_path_buf()), suffix: "_cut".into(), ..Default::default() };
        let mut p = Planner::new(&s, dir.path().to_path_buf());
        let a = Path::new("C:\\in\\photo.jpg");
        let b = Path::new("C:\\in\\photo.png");
        let pa = p.plan(&SourceRef { path: Some(a), name: "photo.jpg", root: None }, "webp").unwrap();
        let pb = p.plan(&SourceRef { path: Some(b), name: "photo.png", root: None }, "webp").unwrap();
        assert_eq!(pa, Planned::Write(dir.path().join("photo_cut.webp")));
        assert_eq!(pb, Planned::Write(dir.path().join("photo_cut (2).webp")));
    }

    #[test]
    fn keeps_structure() {
        let dir = tempfile::tempdir().unwrap();
        let s = ExportSettings { location: Location::Custom, folder: Some(dir.path().to_path_buf()), keep_structure: true, ..Default::default() };
        let p = Planner::new(&s, dir.path().to_path_buf());
        let root = Path::new("C:\\shoot");
        let f = Path::new("C:\\shoot\\day1\\a.jpg");
        let d = p.dir_for(&SourceRef { path: Some(f), name: "a.jpg", root: Some(root) }).unwrap();
        assert_eq!(d, dir.path().join("shoot").join("day1"));
    }

    #[test]
    fn rejects_relative_output_folder() {
        let s = ExportSettings { location: Location::Custom, folder: Some(PathBuf::from("D:relative")), ..Default::default() };
        let p = Planner::new(&s, PathBuf::from("C:/x"));
        assert!(p.dir_for(&SourceRef { path: None, name: "a", root: None }).is_err());
    }

    #[test]
    fn never_overwrites_source() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("x.png");
        std::fs::write(&src, b"x").unwrap();
        let s = ExportSettings { location: Location::SameFolder, conflict: Conflict::Overwrite, ..Default::default() };
        let mut p = Planner::new(&s, dir.path().to_path_buf());
        let out = p.plan(&SourceRef { path: Some(&src), name: "x.png", root: None }, "png").unwrap();
        assert_eq!(out, Planned::Write(dir.path().join("x (2).png")));
    }
}
