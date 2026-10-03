//! Experimental spaced-seed recall channel (Rust port of `corpus_assay/spaced_seeds.py`).
//!
//! The exact channel matches contiguous n-grams by hash. The spaced channel matches a
//! gap-tolerant span-`span`/weight-`weight` seed and then *verifies* the candidate by a
//! token-level local alignment against the protected item's raw tokens. Because the
//! verifier is token-level, the index carries the protected-item token sequences, not
//! just hashes (see `docs/experimental-spaced-seeds.md`).
//!
//! Parity with the Python reference is bit-exact: seed hashes are blake2b-64-le over the
//! seed string `"s{pid}|tok ... tok"`, and the locus collapse / verify logic mirror
//! `spaced_candidates` / `verify`. The `.spaced` artifact is produced by
//! `corpus_assay/spaced_index.py`.

use crate::text::{hash_ngram_text, FIELD_SEPARATOR};
use byteorder::{LittleEndian, ReadBytesExt};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::{Arc, Mutex};

const SPACED_MAGIC: [u8; 8] = *b"CASPACE1";
const SPACED_FORMAT_VERSION: u32 = 1;

/// Selected token positions for each pattern (the indices of the `1`s).
fn offsets_from_patterns(patterns: &[String]) -> Vec<Vec<usize>> {
    patterns
        .iter()
        .map(|p| {
            p.bytes()
                .enumerate()
                .filter_map(|(i, b)| if b == b'1' { Some(i) } else { None })
                .collect()
        })
        .collect()
}

/// Build the seed string for pattern `pid` at window start `i`: `"s{pid}|tok tok ..."`.
/// Class-aware mapping is not applied (only `class_aware=none` is supported).
#[inline]
fn seed_string(toks: &[String], i: usize, pid: usize, offsets: &[usize]) -> String {
    let mut s = String::with_capacity(8 + offsets.len() * 6);
    s.push('s');
    s.push_str(&pid.to_string());
    s.push('|');
    for (k, &o) in offsets.iter().enumerate() {
        if k > 0 {
            s.push(' ');
        }
        s.push_str(&toks[i + o]);
    }
    s
}

/// Emit `(pos, pid, seed_hash)` for every seed window, mirroring `iter_spaced_seeds`:
/// windows touching a field separator are skipped and the cursor jumps past it.
fn for_each_seed<F: FnMut(usize, usize, u64)>(
    toks: &[String],
    span: usize,
    offsets: &[Vec<usize>],
    mut f: F,
) {
    let n = toks.len();
    if span == 0 || n < span {
        return;
    }
    let mut i = 0usize;
    while i + span <= n {
        // rightmost field separator in [i, i+span)
        let mut sep: Option<usize> = None;
        for off in (0..span).rev() {
            if toks[i + off] == FIELD_SEPARATOR {
                sep = Some(i + off);
                break;
            }
        }
        if let Some(s) = sep {
            i = s + 1;
            continue;
        }
        for (pid, offs) in offsets.iter().enumerate() {
            let h = hash_ngram_text(&seed_string(toks, i, pid, offs));
            f(i, pid, h);
        }
        i += 1;
    }
}

/// Loaded `.spaced` artifact plus the verify thresholds.
#[derive(Debug)]
pub struct SpacedIndex {
    offsets: Vec<Vec<usize>>,
    span: usize,
    item_tokens: Vec<Vec<String>>,
    postings: HashMap<u64, Vec<(u32, u32)>>,
    min_loci: usize,
    ver_min_span: usize,
    ver_identity: f64,
}

impl SpacedIndex {
    pub fn open(
        path: &str,
        min_loci: usize,
        ver_min_span: usize,
        ver_identity: f64,
    ) -> PyResult<Arc<Self>> {
        let file = File::open(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to open spaced index '{}': {}", path, e))
        })?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)
            .map_err(|e| PyValueError::new_err(format!("spaced header: {}", e)))?;
        if magic != SPACED_MAGIC {
            return Err(PyValueError::new_err("Invalid spaced index magic header."));
        }
        let version = read_u32(&mut r)?;
        if version != SPACED_FORMAT_VERSION {
            return Err(PyValueError::new_err(format!(
                "Unsupported spaced format version {}",
                version
            )));
        }
        let span = read_u32(&mut r)? as usize;
        let _weight = read_u32(&mut r)?;
        let n_patterns = read_u32(&mut r)? as usize;
        let _exact_n = read_u32(&mut r)?;
        let n_items = read_u32(&mut r)? as usize;
        let n_postings = read_u64(&mut r)? as usize;

        let mut patterns = Vec::with_capacity(n_patterns);
        for _ in 0..n_patterns {
            let mut buf = vec![0u8; span];
            r.read_exact(&mut buf)
                .map_err(|e| PyValueError::new_err(format!("spaced patterns: {}", e)))?;
            patterns.push(
                String::from_utf8(buf)
                    .map_err(|e| PyValueError::new_err(format!("spaced pattern utf8: {}", e)))?,
            );
        }
        let offsets = offsets_from_patterns(&patterns);

        // token intern table
        let n_tokens = read_u32(&mut r)? as usize;
        let mut vocab: Vec<String> = Vec::with_capacity(n_tokens.min(1_000_000));
        for _ in 0..n_tokens {
            let blen = read_u32(&mut r)? as usize;
            let mut buf = vec![0u8; blen];
            r.read_exact(&mut buf)
                .map_err(|e| PyValueError::new_err(format!("spaced token bytes: {}", e)))?;
            vocab.push(
                String::from_utf8(buf)
                    .map_err(|e| PyValueError::new_err(format!("spaced token utf8: {}", e)))?,
            );
        }

        let mut item_tokens: Vec<Vec<String>> = Vec::with_capacity(n_items.min(10_000_000));
        for _ in 0..n_items {
            let cnt = read_u32(&mut r)? as usize;
            let mut toks = Vec::with_capacity(cnt.min(100_000));
            for _ in 0..cnt {
                let id = read_u32(&mut r)? as usize;
                let tok = vocab.get(id).ok_or_else(|| {
                    PyValueError::new_err(format!("spaced token id {} out of range", id))
                })?;
                toks.push(tok.clone());
            }
            item_tokens.push(toks);
        }

        let mut postings: HashMap<u64, Vec<(u32, u32)>> =
            HashMap::with_capacity(n_postings.min(50_000_000));
        for _ in 0..n_postings {
            let hash = read_u64(&mut r)?;
            let n_entries = read_u32(&mut r)? as usize;
            let mut entries = Vec::with_capacity(n_entries.min(1_000_000));
            for _ in 0..n_entries {
                let item_id = read_u32(&mut r)?;
                let item_pos = read_u32(&mut r)?;
                // Bounds-check so a corrupt/hostile file can't panic the verifier's
                // `item_tokens[item_id]` lookup later.
                if item_id as usize >= item_tokens.len() {
                    return Err(PyValueError::new_err(format!(
                        "spaced posting item_id {} out of range (n_items={})",
                        item_id,
                        item_tokens.len()
                    )));
                }
                entries.push((item_id, item_pos));
            }
            postings.insert(hash, entries);
        }

        let mut trailing = [0u8; 1];
        if r.read(&mut trailing).unwrap_or(0) > 0 {
            return Err(PyValueError::new_err(
                "Unexpected trailing data in spaced index",
            ));
        }

        Ok(Arc::new(Self {
            offsets,
            span,
            item_tokens,
            postings,
            min_loci,
            ver_min_span,
            ver_identity,
        }))
    }

    /// Same-item local alignment on raw tokens; mirrors `verify`. `diag = doc_pos -
    /// item_pos`, so the aligned item index is `dp - diag`. `loci` is sorted ascending,
    /// so min/max are its first/last element.
    fn verify(&self, toks: &[String], item_toks: &[String], diag: i64, loci: &[usize]) -> bool {
        let (Some(&first), Some(&last)) = (loci.first(), loci.last()) else {
            return false;
        };
        let lo = first as i64;
        let hi = last as i64 + self.span as i64;
        let overlap_lo = lo.max(0).max(diag);
        let overlap_hi = hi.min(toks.len() as i64).min(item_toks.len() as i64 + diag);
        let total = overlap_hi - overlap_lo;
        if total < self.ver_min_span as i64 || total <= 0 {
            return false;
        }
        let mut matched = 0i64;
        for dp in overlap_lo..overlap_hi {
            if toks[dp as usize] == item_toks[(dp - diag) as usize] {
                matched += 1;
            }
        }
        (matched as f64) / (total as f64) >= self.ver_identity
    }

    /// Returns the id of a protected item that the spaced channel verifies against
    /// (`spaced_verified_flag`), or `None`. Mirrors `SpacedSeedDetector.spaced_flag`.
    pub fn verified_item(&self, toks: &[String]) -> Option<u32> {
        // Distinct doc positions per (item_id, diagonal). `for_each_seed` yields `pos`
        // in non-decreasing order, so appending while skipping a repeated last value
        // keeps each Vec sorted and deduplicated without a HashSet.
        let mut loci: HashMap<(u32, i64), Vec<usize>> = HashMap::new();
        for_each_seed(toks, self.span, &self.offsets, |pos, _pid, h| {
            if let Some(entries) = self.postings.get(&h) {
                for &(item_id, item_pos) in entries {
                    let diag = pos as i64 - item_pos as i64;
                    let v = loci.entry((item_id, diag)).or_default();
                    if v.last() != Some(&pos) {
                        v.push(pos);
                    }
                }
            }
        });
        // Deterministic selection: smallest (item_id, diagonal) that verifies, so the
        // reported `spaced_item` attribution is reproducible across runs.
        let mut candidates: Vec<(u32, i64)> = loci
            .iter()
            .filter(|(_, v)| v.len() >= self.min_loci)
            .map(|(k, _)| *k)
            .collect();
        candidates.sort_unstable();
        for (item_id, diag) in candidates {
            let positions = &loci[&(item_id, diag)];
            if self.verify(toks, &self.item_tokens[item_id as usize], diag, positions) {
                return Some(item_id);
            }
        }
        None
    }
}

static SPACED_CACHE: Lazy<Mutex<std::collections::HashMap<String, Arc<SpacedIndex>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Load (and cache by path) a spaced index with the given verify thresholds.
pub fn load_spaced_cached(
    path: &str,
    min_loci: usize,
    ver_min_span: usize,
    ver_identity: f64,
) -> PyResult<Arc<SpacedIndex>> {
    // Cache key includes the thresholds so two configs over the same file don't collide.
    let key = format!("{}|{}|{}|{}", path, min_loci, ver_min_span, ver_identity);
    {
        let cache = SPACED_CACHE
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock spaced cache"))?;
        if let Some(e) = cache.get(&key) {
            return Ok(Arc::clone(e));
        }
    }
    let idx = SpacedIndex::open(path, min_loci, ver_min_span, ver_identity)?;
    let mut cache = SPACED_CACHE
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock spaced cache"))?;
    Ok(Arc::clone(cache.entry(key).or_insert(idx)))
}

#[inline]
fn read_u32<R: Read>(r: &mut R) -> PyResult<u32> {
    r.read_u32::<LittleEndian>()
        .map_err(|e| PyValueError::new_err(format!("spaced read u32: {}", e)))
}

#[inline]
fn read_u64<R: Read>(r: &mut R) -> PyResult<u64> {
    r.read_u64::<LittleEndian>()
        .map_err(|e| PyValueError::new_err(format!("spaced read u64: {}", e)))
}

// --------------------------------------------------------------------------- //
// PyO3 parity entry points (used by the Python parity tests against the golden
// vectors; the production scan path calls SpacedIndex directly in scan.rs).
// --------------------------------------------------------------------------- //

/// The `(pos, pid, seed_hash)` stream for `tokens` under `patterns` (parity with
/// `iter_spaced_seeds`). Hashes are returned as u64.
#[pyfunction]
pub fn spaced_seed_hashes_rust(
    tokens: Vec<String>,
    patterns: Vec<String>,
) -> Vec<(usize, usize, u64)> {
    let span = patterns.first().map(|p| p.len()).unwrap_or(0);
    let offsets = offsets_from_patterns(&patterns);
    let mut out = Vec::new();
    for_each_seed(&tokens, span, &offsets, |pos, pid, h| {
        out.push((pos, pid, h))
    });
    out
}

/// Load `spaced_path` and return the verified item id (or `None`) for `tokens` — parity
/// with `SpacedSeedDetector.spaced_flag(verified=True)`.
#[pyfunction]
pub fn spaced_verified_item_rust(
    spaced_path: &str,
    tokens: Vec<String>,
    min_loci: usize,
    ver_min_span: usize,
    ver_identity: f64,
) -> PyResult<Option<u32>> {
    let idx = load_spaced_cached(spaced_path, min_loci, ver_min_span, ver_identity)?;
    Ok(idx.verified_item(&tokens))
}

// Note: behaviour is validated by the Python parity tests
// (tests/test_spaced_rust_parity.py), which exercise the seed stream, the verify
// thresholds, and the verified-flag verdict against the golden vectors and the live
// reference. `cargo test` cannot link the pyo3 extension-module here, so unit tests live
// on the Python side where they also run in CI.
