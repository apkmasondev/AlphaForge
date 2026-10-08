//! Safe file-name and path handling for user-controlled names (prefix/suffix, folder structure).

use std::path::{Component, Path, PathBuf};

use crate::{Error, Result};

const RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Make an arbitrary string safe as a Windows file-name component (no extension handling).
pub fn sanitize_component(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            c if (c as u32) < 32 => '_',
            c => c,
        })
        .collect();
    // no trailing dots/spaces (Windows strips them silently)
    while out.ends_with('.') || out.ends_with(' ') {
        out.pop();
    }
    let trimmed = out.trim_start().to_string();
    let mut out = trimmed;
    if out.chars().count() > 150 {
        out = out.chars().take(150).collect();
    }
    let upper = out.split('.').next().unwrap_or("").to_ascii_uppercase();
    if RESERVED.contains(&upper.as_str()) {
        out.insert(0, '_');
    }
    out
}

/// Sanitize a stem; falls back to "image" when nothing usable is left.
pub fn sanitize_stem(s: &str) -> String {
    let v = sanitize_component(s);
    if v.is_empty() || v.chars().all(|c| c == '_' || c == '.') {
        "image".into()
    } else {
        v
    }
}

/// Join a *relative* path below `base`, rejecting absolute paths, drive prefixes and `..`.
pub fn safe_join(base: &Path, rel: &Path) -> Result<PathBuf> {
    let mut out = base.to_path_buf();
    for c in rel.components() {
        match c {
            Component::Normal(p) => out.push(sanitize_component(&p.to_string_lossy())),
            Component::CurDir => {}
            _ => return Err(Error::Invalid(format!("unsafe path component in {}", rel.display()))),
        }
    }
    Ok(out)
}

/// `rel` = path of `file`'s directory relative to `root`, if `file` is inside `root`.
pub fn relative_dir(root: &Path, file: &Path) -> Option<PathBuf> {
    let dir = file.parent()?;
    dir.strip_prefix(root).ok().map(|p| p.to_path_buf())
}

/// Windows paths compare case-insensitively.
pub fn same_path(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().to_lowercase();
    norm(a) == norm(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes() {
        assert_eq!(sanitize_component("a<b>:c"), "a_b__c");
        assert_eq!(sanitize_component("name. . "), "name");
        assert_eq!(sanitize_component("CON"), "_CON");
        assert_eq!(sanitize_component("con.txt"), "_con.txt");
        assert_eq!(sanitize_stem("???"), "image");
        assert_eq!(sanitize_stem(""), "image");
    }

    #[test]
    fn joins_safely() {
        let b = Path::new("C:\\out");
        assert!(safe_join(b, Path::new("a\\b")).is_ok());
        assert!(safe_join(b, Path::new("..\\x")).is_err());
        assert!(safe_join(b, Path::new("C:\\Windows")).is_err());
        assert!(safe_join(b, Path::new("\\\\server\\share")).is_err());
    }
}
