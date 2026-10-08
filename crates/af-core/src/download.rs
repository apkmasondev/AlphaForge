//! The only networking in AlphaForge: fetching model weights and the optional GPU runtime.
//!
//! Security properties:
//! * HTTPS only, and every hop of a redirect chain must be on [`ALLOWED_HOSTS`].
//! * Every file is verified against a SHA-256 pinned in the application before it is used;
//!   partial downloads live in `*.part` files and are only renamed after verification.
//! * Nothing is executed or deserialized from downloads except verified DLLs (GPU pack) and
//!   safetensors / ONNX data files.
//! * No cookies, no tokens, no telemetry: requests carry only a User-Agent.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::{CancelToken, Error, Result};

pub const ALLOWED_HOSTS: &[&str] = &[
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "files.pythonhosted.org",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "cas-bridge.xethub.hf.co",
    "cas-bridge-direct.xethub.hf.co",
    "transfer.xethub.hf.co",
];

fn host_allowed(host: &str) -> bool {
    let h = host.to_ascii_lowercase();
    ALLOWED_HOSTS.iter().any(|a| h == *a) || h.ends_with(".hf.co") || h.ends_with(".huggingface.co")
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_recv_body(Some(Duration::from_secs(120)))
        .max_redirects(0)
        .http_status_as_error(false)
        .user_agent(concat!("AlphaForge/", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

fn check_url(url: &str) -> Result<()> {
    let rest = url.strip_prefix("https://").ok_or_else(|| Error::Download(format!("refusing non-HTTPS URL {url}")))?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit('@').next().unwrap_or(host); // never allow userinfo tricks
    let host = host.split(':').next().unwrap_or(host);
    if !host_allowed(host) {
        return Err(Error::Download(format!("host {host} is not on the allow-list")));
    }
    Ok(())
}

fn map_transport(e: ureq::Error) -> Error {
    let s = e.to_string();
    let lower = s.to_ascii_lowercase();
    if lower.contains("dns") || lower.contains("resolve") || lower.contains("connect") || lower.contains("timed out") || lower.contains("timeout") || lower.contains("host") {
        Error::Offline(s)
    } else {
        Error::Download(s)
    }
}

/// GET with manual redirect following (allow-list enforced on every hop).
fn get(agent: &ureq::Agent, url: &str, range: Option<(u64, Option<u64>)>) -> Result<ureq::http::Response<ureq::Body>> {
    let mut current = url.to_string();
    for _ in 0..8 {
        check_url(&current)?;
        let mut req = agent.get(&current);
        if let Some((start, end)) = range {
            let v = match end {
                Some(e) => format!("bytes={start}-{e}"),
                None => format!("bytes={start}-"),
            };
            req = req.header("Range", v);
        }
        let resp = req.call().map_err(map_transport)?;
        let status = resp.status().as_u16();
        if (300..400).contains(&status) {
            let loc = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| Error::Download("redirect without location".into()))?;
            current = if loc.starts_with("https://") {
                loc.to_string()
            } else if let Some(path) = loc.strip_prefix('/') {
                let base = current.splitn(4, '/').take(3).collect::<Vec<_>>().join("/");
                format!("{base}/{path}")
            } else {
                return Err(Error::Download(format!("unsupported redirect {loc}")));
            };
            continue;
        }
        if status == 416 {
            return Err(Error::Download("range not satisfiable".into()));
        }
        if !(200..300).contains(&status) {
            return Err(Error::Download(format!("server answered HTTP {status}")));
        }
        return Ok(resp);
    }
    Err(Error::Download("too many redirects".into()))
}

/// Download `url` to `dest`, resuming a previous `dest.part`, verifying size and SHA-256.
/// `progress(done_bytes, total_bytes)`.
pub fn download_file(url: &str, dest: &Path, sha256: &str, size: u64, cancel: &CancelToken, progress: &dyn Fn(u64, u64)) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
        let have = part_path(dest).metadata().map(|m| m.len()).unwrap_or(0);
        crate::hw::ensure_space(parent, size.saturating_sub(have))?;
    }
    let part = part_path(dest);
    let agent = agent();
    let mut attempts = 0;
    loop {
        cancel.check()?;
        let have = part.metadata().map(|m| m.len()).unwrap_or(0);
        if have > size {
            std::fs::remove_file(&part)?;
            continue;
        }
        if have == size {
            break;
        }
        match fetch_into(&agent, url, &part, have, size, cancel, progress) {
            Ok(()) => {}
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => {
                attempts += 1;
                if attempts >= 6 {
                    return Err(e);
                }
                log::warn!("download retry {attempts}: {e}");
                std::thread::sleep(Duration::from_millis(800 * attempts));
            }
        }
    }
    progress(size, size);
    let got = crate::ai::template::sha256_file(&part)?;
    if !got.eq_ignore_ascii_case(sha256) {
        let _ = std::fs::remove_file(&part);
        return Err(Error::Checksum(dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
    }
    replace_file(&part, dest)?;
    Ok(())
}

fn fetch_into(agent: &ureq::Agent, url: &str, part: &Path, have: u64, size: u64, cancel: &CancelToken, progress: &dyn Fn(u64, u64)) -> Result<()> {
    let resp = get(agent, url, if have > 0 { Some((have, None)) } else { None })?;
    let partial = resp.status().as_u16() == 206;
    let mut file = std::fs::OpenOptions::new().create(true).append(partial).write(true).truncate(!partial).open(part)?;
    let mut done = if partial { have } else { 0 };
    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        cancel.check()?;
        let n = reader.read(&mut buf).map_err(|e| Error::Download(e.to_string()))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if done > size {
            return Err(Error::Download("server sent more data than expected".into()));
        }
        progress(done, size);
    }
    file.flush()?;
    if done < size {
        return Err(Error::Download("connection closed early".into()));
    }
    Ok(())
}

pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Rename with replace (Windows `MoveFileEx` semantics via std).
pub fn replace_file(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        std::fs::remove_file(to)?;
    }
    std::fs::rename(from, to)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Remote ZIP entry extraction (HTTP range requests)
// ---------------------------------------------------------------------------------------------

/// Location of one entry inside a remote ZIP archive.
#[derive(Debug, Clone)]
pub struct RemoteEntry {
    pub name: String,
    pub method: u16,
    pub compressed: u64,
    pub uncompressed: u64,
    pub local_header_offset: u64,
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn read_all(resp: ureq::http::Response<ureq::Body>, limit: u64) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    resp.into_body().into_reader().take(limit).read_to_end(&mut v).map_err(|e| Error::Download(e.to_string()))?;
    Ok(v)
}

/// Read the central directory of a remote ZIP. Requires HTTP range support.
pub fn remote_zip_index(url: &str) -> Result<Vec<RemoteEntry>> {
    let agent = agent();
    // Last 64 KiB contain the end-of-central-directory record (+ zip64 locator).
    let resp = get(&agent, url, Some((0, Some(0))))?;
    if resp.status().as_u16() != 206 {
        return Err(Error::Download("server does not support range requests".into()));
    }
    let total: u64 = resp
        .headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.rsplit('/').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Download("missing content-range".into()))?;
    drop(resp);
    let tail_len = total.min(65_557 + 20);
    let tail = read_all(get(&agent, url, Some((total - tail_len, Some(total - 1))))?, tail_len)?;
    let eocd = (0..tail.len().saturating_sub(21)).rev().find(|&i| le32(&tail, i) == 0x0605_4b50).ok_or_else(|| Error::Download("not a zip archive".into()))?;
    let mut cd_size = le32(&tail, eocd + 12) as u64;
    let mut cd_off = le32(&tail, eocd + 16) as u64;
    let mut count = le16(&tail, eocd + 10) as u64;
    if cd_off == 0xFFFF_FFFF || cd_size == 0xFFFF_FFFF || count == 0xFFFF {
        // zip64 locator sits right before the EOCD
        let loc = eocd.checked_sub(20).ok_or_else(|| Error::Download("bad zip64".into()))?;
        if le32(&tail, loc) != 0x0706_4b50 {
            return Err(Error::Download("bad zip64 locator".into()));
        }
        let z64_off = le64(&tail, loc + 8);
        let rec = read_all(get(&agent, url, Some((z64_off, Some(z64_off + 55))))?, 56)?;
        if rec.len() < 56 || le32(&rec, 0) != 0x0606_4b50 {
            return Err(Error::Download("bad zip64 record".into()));
        }
        count = le64(&rec, 32);
        cd_size = le64(&rec, 40);
        cd_off = le64(&rec, 48);
    }
    if cd_size > 64 << 20 {
        return Err(Error::Download("central directory too large".into()));
    }
    let cd = read_all(get(&agent, url, Some((cd_off, Some(cd_off + cd_size - 1))))?, cd_size)?;
    let mut out = Vec::with_capacity(count as usize);
    let mut p = 0usize;
    while p + 46 <= cd.len() && le32(&cd, p) == 0x0201_4b50 {
        let method = le16(&cd, p + 10);
        let mut csize = le32(&cd, p + 20) as u64;
        let mut usize_ = le32(&cd, p + 24) as u64;
        let nlen = le16(&cd, p + 28) as usize;
        let xlen = le16(&cd, p + 30) as usize;
        let clen = le16(&cd, p + 32) as usize;
        let mut lho = le32(&cd, p + 42) as u64;
        if p + 46 + nlen + xlen + clen > cd.len() {
            return Err(Error::Download("truncated zip directory".into()));
        }
        let name = String::from_utf8_lossy(&cd[p + 46..p + 46 + nlen]).into_owned();
        // zip64 extra field
        let mut x = p + 46 + nlen;
        let xend = x + xlen;
        while x + 4 <= xend {
            let id = le16(&cd, x);
            let sz = le16(&cd, x + 2) as usize;
            if x + 4 + sz > xend {
                break;
            }
            if id == 0x0001 {
                let mut q = x + 4;
                let end = x + 4 + sz;
                if usize_ == 0xFFFF_FFFF && q + 8 <= end {
                    usize_ = le64(&cd, q);
                    q += 8;
                }
                if csize == 0xFFFF_FFFF && q + 8 <= end {
                    csize = le64(&cd, q);
                    q += 8;
                }
                if lho == 0xFFFF_FFFF && q + 8 <= end {
                    lho = le64(&cd, q);
                }
            }
            x += 4 + sz;
        }
        out.push(RemoteEntry { name, method, compressed: csize, uncompressed: usize_, local_header_offset: lho });
        p += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

/// Stream one entry of a remote ZIP into `dest` (via `dest.part`), verifying size + SHA-256.
/// `progress(compressed_bytes_done)`.
pub fn extract_remote_entry(url: &str, e: &RemoteEntry, dest: &Path, sha256: &str, cancel: &CancelToken, progress: &dyn Fn(u64)) -> Result<()> {
    if e.method != 0 && e.method != 8 {
        return Err(Error::Download(format!("unsupported compression method {}", e.method)));
    }
    let agent = agent();
    let hdr = read_all(get(&agent, url, Some((e.local_header_offset, Some(e.local_header_offset + 29))))?, 30)?;
    if hdr.len() < 30 || le32(&hdr, 0) != 0x0403_4b50 {
        return Err(Error::Download("bad local file header".into()));
    }
    let data_off = e.local_header_offset + 30 + le16(&hdr, 26) as u64 + le16(&hdr, 28) as u64;
    let resp = get(&agent, url, Some((data_off, Some(data_off + e.compressed - 1))))?;
    if resp.status().as_u16() != 206 {
        return Err(Error::Download("server ignored the range request".into()));
    }
    let part = part_path(dest);
    let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&part)?);
    let counted = CountingReader { inner: resp.into_body().into_reader().take(e.compressed), n: 0, cancel, progress };
    let mut hasher = Sha256::new();
    let mut written = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    let mut reader: Box<dyn Read> = if e.method == 8 { Box::new(flate2::read::DeflateDecoder::new(counted)) } else { Box::new(counted) };
    loop {
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted && cancel.is_cancelled() => return Err(Error::Cancelled),
            Err(err) => return Err(Error::Download(err.to_string())),
        };
        if n == 0 {
            break;
        }
        written += n as u64;
        if written > e.uncompressed {
            return Err(Error::Download("entry larger than declared".into()));
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    drop(out);
    cancel.check()?;
    if written != e.uncompressed {
        let _ = std::fs::remove_file(&part);
        return Err(Error::Download("connection closed early".into()));
    }
    if !hex::encode(hasher.finalize()).eq_ignore_ascii_case(sha256) {
        let _ = std::fs::remove_file(&part);
        return Err(Error::Checksum(e.name.rsplit('/').next().unwrap_or(&e.name).to_string()));
    }
    replace_file(&part, dest)
}

struct CountingReader<'a, R> {
    inner: R,
    n: u64,
    cancel: &'a CancelToken,
    progress: &'a dyn Fn(u64),
}

impl<R: Read> Read for CountingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "cancelled"));
        }
        let n = self.inner.read(buf)?;
        self.n += n as u64;
        (self.progress)(self.n);
        Ok(n)
    }
}

/// Lightweight connectivity probe used to give a clear "offline" message before big downloads.
pub fn probe(url: &str) -> Result<()> {
    let agent = agent();
    get(&agent, url, Some((0, Some(0)))).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_disk_space_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        match crate::hw::ensure_space(dir.path(), u64::MAX / 4) {
            Err(Error::DiskSpace { .. }) => {}
            other => panic!("expected DiskSpace, got {other:?}"),
        }
    }

    #[test]
    #[ignore = "needs DNS; run with --ignored"]
    fn unreachable_host_reports_offline() {
        let dir = tempfile::tempdir().unwrap();
        let r = download_file("https://does-not-exist-alphaforge-test.hf.co/x.bin", &dir.path().join("x.bin"), "00", 10, &CancelToken::new(), &|_, _| {});
        assert!(matches!(r, Err(Error::Offline(_))), "{r:?}");
        assert!(!dir.path().join("x.bin").exists());
    }

    #[test]
    #[ignore = "needs network; run with --ignored"]
    fn checksum_mismatch_discards_file() {
        let dir = tempfile::tempdir().unwrap();
        let url = "https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/aa62cd87eafb9cc43056d08ef3615a14628b831d/config.json";
        let dest = dir.path().join("config.json");
        let r = download_file(url, &dest, &"0".repeat(64), 1, &CancelToken::new(), &|_, _| {});
        assert!(r.is_err());
        assert!(!dest.exists());
    }

    #[test]
    fn url_allow_list() {
        assert!(check_url("https://huggingface.co/ZhengPeng7/BiRefNet/resolve/x/model.safetensors").is_ok());
        assert!(check_url("https://files.pythonhosted.org/packages/x.whl").is_ok());
        assert!(check_url("http://huggingface.co/x").is_err());
        assert!(check_url("https://evil.example.com/x").is_err());
        assert!(check_url("https://huggingface.co.evil.com/x").is_err());
        assert!(check_url("https://user@evil.com/x").is_err());
    }
}
