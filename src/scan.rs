use crate::formats::{write_header, FileType};
use crate::index::{get_attr_index, get_hash_index, get_index_ngram};
use crate::items_index::{get_items_index, items_sidecar_exists, ItemPostings};
use crate::mmap_index::MmapAttrIndex;
use crate::spaced::{load_spaced_cached, SpacedIndex};
use crate::stopgrams::StopGramsSet;
use crate::text::{
    doc_identity_rust, get_word_re, hash_ngram_text, join_ngram_window, ngram_allowed_rust,
    normalize_text_rust, FIELD_SEPARATOR,
};
use arrow::array::RecordBatchReader;
use once_cell::sync::Lazy;
use parquet::file::reader::ChunkReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use arrow::array::{Array, StringArray};
use byteorder::{LittleEndian, WriteBytesExt};
use hashbrown::{HashMap, HashSet};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Per-item attribution evidence for one flagged document. `hits` is the number
/// of distinct index n-grams matching the protected item; it partitions into
/// `unique_hits` (n-gram belongs to exactly one protected item — strongest
/// evidence), `shared_hits` (n-gram shared by a few items) and `boilerplate_hits`
/// (n-gram present in `>= boilerplate_threshold` items — near-worthless), by the
/// item fan-out of each matched hash.
#[derive(Serialize, Clone, Copy)]
struct ItemHit {
    item_id: u32,
    hits: usize,
    longest_run_tokens: usize,
    unique_hits: usize,
    shared_hits: usize,
    boilerplate_hits: usize,
}

#[derive(Serialize)]
struct FileScanResult {
    schema_version: u32,
    doc_id: String,
    match_count: usize,
    src_hits: Vec<(u32, usize)>,
    /// Item-level attribution, populated only in `item` gate mode: one `ItemHit`
    /// (with its unique/shared/boilerplate split) for every protected item that on
    /// its own meets `min_hits` (the per-item evidence that justified the flag).
    /// Empty in `union` mode.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    item_hits: Vec<ItemHit>,
    /// Protected item id that the spaced channel verified against, when the document was
    /// flagged by the spaced channel rather than the exact gate. `None` for exact flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    spaced_item: Option<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GateMode {
    /// Legacy: a doc is flagged when the *union* of distinct index hits across
    /// all benchmarks meets `min_hits`/`min_coverage`.
    Union,
    /// A doc is flagged only when a *single protected item* meets
    /// `min_hits`/`min_coverage` (and the optional `min_longest_run` locality
    /// clause), preventing hits scattered across unrelated items from combining.
    Item,
}

type ScanOutput = (usize, usize, Vec<String>, usize);

static STOPGRAM_CACHE: Lazy<Mutex<HashMap<String, Arc<StopGramsSet>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn load_stopgrams_cached(path: &str) -> PyResult<Arc<StopGramsSet>> {
    let cache_key = std::fs::canonicalize(path)
        .unwrap_or_else(|_| Path::new(path).to_path_buf())
        .to_string_lossy()
        .to_string();
    let mut cache = STOPGRAM_CACHE
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock stop-grams cache"))?;
    if let Some(entry) = cache.get(&cache_key) {
        return Ok(Arc::clone(entry));
    }
    let set = StopGramsSet::open(path)?;
    cache.insert(cache_key, Arc::clone(&set));
    Ok(set)
}

#[derive(Clone)]
struct PreparedScan {
    text_key: String,
    id_key: Option<String>,
    packed_doc_sep: String,
    packed_doc_sep_typo: Option<String>,
    backend: IndexBackend,
    out_hits_path: Option<String>,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    stopgrams: Option<Arc<StopGramsSet>>,
    gate_mode: GateMode,
    min_longest_run: usize,
    /// Item fan-out at/above which a matched n-gram is counted as boilerplate in
    /// the per-item evidence split.
    boilerplate_threshold: usize,
    item_postings: Option<Arc<ItemPostings>>,
    /// Optional spaced-seed recall channel: a doc is flagged when the exact gate fires
    /// OR the spaced verifier matches (union recall). `None` disables the channel.
    spaced: Option<Arc<SpacedIndex>>,
}

#[derive(Clone)]
enum IndexBackend {
    InMemory {
        hashes: Arc<HashSet<u64>>,
        attr: Arc<HashMap<u64, u64>>,
    },
    MmapAttr {
        attr: Arc<MmapAttrIndex>,
    },
}

impl PreparedScan {
    #[inline]
    fn mask_for_hash(&self, h: u64) -> Option<u64> {
        match &self.backend {
            IndexBackend::InMemory { hashes, attr } => {
                if hashes.contains(&h) {
                    Some(*attr.get(&h).unwrap_or(&0))
                } else {
                    None
                }
            }
            IndexBackend::MmapAttr { attr } => attr.mask_for_hash(h),
        }
    }
}

/// Item-level gate: a doc is contaminated only if a *single* protected item
/// accumulates `min_hits` distinct matching n-grams at `min_coverage` (and, when
/// `min_longest_run > 0`, a contiguous shared run of that many tokens).
///
/// Returns `(flagged, report_count, evidence)` where `report_count` is the
/// strongest single-item hit count and `evidence` is one `ItemHit` per item that
/// on its own clears the gate. Each item's distinct hits are split into
/// unique/shared/boilerplate by the item fan-out of each matched hash
/// (`hash_to_items.len()`): `1` → unique, `>= boilerplate_threshold` → boilerplate,
/// otherwise shared.
#[allow(clippy::too_many_arguments)]
fn item_gate(
    postings: &ItemPostings,
    indexed: &HashSet<u64>,
    positions: &[(u64, usize)],
    denom_all: f64,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    min_longest_run: usize,
    boilerplate_threshold: usize,
    // Scratch buffers owned by the caller and reused across sub-documents to
    // avoid per-doc heap allocation; cleared on entry. Value is the per-item
    // (unique_hits, shared_hits, boilerplate_hits) split; total hits is their sum.
    per_item_hits: &mut HashMap<u32, (usize, usize, usize)>,
    state: &mut HashMap<u32, (i64, usize, usize)>, // (last_pos, cur, best)
) -> (bool, usize, Vec<ItemHit>) {
    // Distinct matching hashes per protected item, bucketed by hash fan-out.
    // `indexed` already restricts to hashes present in the active index, so a hash
    // that exists only in a stale `.items` sidecar is ignored.
    per_item_hits.clear();
    for h in indexed.iter() {
        if let Some(items) = postings.hash_to_items.get(h) {
            let fan_out = items.len();
            for &it in items {
                let e = per_item_hits.entry(it).or_insert((0, 0, 0));
                if fan_out <= 1 {
                    e.0 += 1;
                } else if fan_out >= boilerplate_threshold {
                    e.2 += 1;
                } else {
                    e.1 += 1;
                }
            }
        }
    }
    if per_item_hits.is_empty() {
        return (false, 0, Vec::new());
    }

    // Longest contiguous shared run (in n-grams) per item, over the positional
    // stream; a gap (filtered/disallowed/out-of-index n-gram) ends a run.
    state.clear();
    for &(h, pos) in positions {
        if !indexed.contains(&h) {
            continue;
        }
        if let Some(items) = postings.hash_to_items.get(&h) {
            for &it in items {
                let e = state.entry(it).or_insert((-1, 0, 0));
                if e.1 > 0 && pos as i64 == e.0 + 1 {
                    e.1 += 1;
                } else {
                    e.1 = 1;
                }
                e.0 = pos as i64;
                if e.1 > e.2 {
                    e.2 = e.1;
                }
            }
        }
    }

    let mut best_hits = 0usize;
    let mut flagged = false;
    let mut evidence: Vec<ItemHit> = Vec::new();
    for (&item_id, &(unique_hits, shared_hits, boilerplate_hits)) in per_item_hits.iter() {
        let hits = unique_hits + shared_hits + boilerplate_hits;
        if hits > best_hits {
            best_hits = hits;
        }
        if hits < min_hits {
            continue;
        }
        if (hits as f64) / denom_all < min_coverage {
            continue;
        }
        let run_ngrams = state.get(&item_id).map(|s| s.2).unwrap_or(0);
        let longest_run_tokens = if run_ngrams > 0 {
            run_ngrams + n - 1
        } else {
            0
        };
        if min_longest_run > 0 && longest_run_tokens < min_longest_run {
            continue;
        }
        evidence.push(ItemHit {
            item_id,
            hits,
            longest_run_tokens,
            unique_hits,
            shared_hits,
            boilerplate_hits,
        });
        flagged = true;
    }
    // Deterministic ordering: strongest evidence first, then by item id.
    evidence.sort_unstable_by(|a, b| b.hits.cmp(&a.hits).then(a.item_id.cmp(&b.item_id)));
    (flagged, best_hits, evidence)
}

#[derive(Clone)]
struct ScanLimits {
    max_return_records: Option<usize>,
}

/// Scan a Parquet reader (must be Read+Seek) for contamination.
fn scan_reader<R>(reader: R, cfg: PreparedScan, limits: ScanLimits) -> Result<ScanOutput, String>
where
    R: ChunkReader + 'static,
{
    let re = get_word_re();
    let mut matched_hash_to_mask: HashMap<u64, u64> = HashMap::new();

    // Reused across sub-documents (cleared each iteration) to avoid per-doc
    // allocation in the hot scan loop.
    let mut doc_shingles: HashSet<u64> = HashSet::new();
    let mut positions: Vec<(u64, usize)> = Vec::new();
    let mut hits_by_src: HashMap<u32, usize> = HashMap::new();
    let mut indexed: HashSet<u64> = HashSet::new();
    let mut per_item_hits: HashMap<u32, (usize, usize, usize)> = HashMap::new();
    let mut item_run_state: HashMap<u32, (i64, usize, usize)> = HashMap::new();

    let mut arrow_reader = match ParquetRecordBatchReaderBuilder::try_new(reader) {
        Ok(b) => match b.with_batch_size(1024).build() {
            Ok(r) => r,
            Err(e) => return Err(e.to_string()),
        },
        Err(e) => return Err(e.to_string()),
    };

    // schema() is available on the built reader
    let schema = arrow_reader.schema();
    let text_col_idx = match schema.index_of(&cfg.text_key) {
        Ok(v) => v,
        Err(e) => {
            return Err(format!(
                "Text key '{}' not found in schema: {}",
                cfg.text_key, e
            ))
        }
    };

    let id_col_idx = cfg
        .id_key
        .as_ref()
        .and_then(|key| schema.index_of(key).ok());

    let mut total_scanned = 0usize;
    let mut total_contaminated = 0usize;
    let mut empty_after_filter = 0usize;
    let mut results_vec: Vec<String> = Vec::new();
    let mut returned_count = 0usize;
    let log_empty_filtered = std::env::var_os("CORPUS_ASSAY_SCAN_LOG_EMPTY_FILTERED").is_some();

    for batch_result in arrow_reader.by_ref() {
        // Pin the concrete item type so Rust stops guessing
        let batch_result: Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> =
            batch_result;

        let batch = batch_result.map_err(|e| e.to_string())?;

        let text_array = batch
            .column(text_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| "Text column is not a string type".to_string())?;

        let id_array: Option<&StringArray> =
            id_col_idx.and_then(|idx| batch.column(idx).as_any().downcast_ref::<StringArray>());

        for i in 0..text_array.len() {
            if !text_array.is_valid(i) {
                continue;
            }

            let packed_text_original: &str = text_array.value(i);
            if packed_text_original.is_empty() {
                continue;
            }

            let packed_text: String = if let Some(ref typo) = cfg.packed_doc_sep_typo {
                packed_text_original.replace(typo, &cfg.packed_doc_sep)
            } else {
                packed_text_original.to_string()
            };

            let original_id_val: Option<&str> = id_array.and_then(|arr| {
                if arr.is_valid(i) {
                    Some(arr.value(i))
                } else {
                    None
                }
            });

            for (sub_doc_index, sub_text) in
                packed_text.as_str().split(&cfg.packed_doc_sep).enumerate()
            {
                let sub_text: &str = sub_text;
                if sub_text.is_empty() || (sub_text.len() < cfg.n * 2) {
                    continue;
                }
                total_scanned += 1;

                let toks = normalize_text_rust(sub_text, re);
                if toks.len() < cfg.n {
                    continue;
                }

                // Positional n-gram stream (token order), reused for the unique
                // shingle set, the per-benchmark mask hits, and per-item runs.
                positions.clear();
                doc_shingles.clear();
                for (pos, win) in toks.windows(cfg.n).enumerate() {
                    if win.iter().any(|token| token == FIELD_SEPARATOR) {
                        continue;
                    }
                    let ngram = join_ngram_window(win);
                    if !ngram_allowed_rust(&ngram) {
                        continue;
                    }
                    let hash = hash_ngram_text(&ngram);
                    doc_shingles.insert(hash);
                    positions.push((hash, pos));
                }
                if doc_shingles.is_empty() {
                    continue;
                }
                let denom_all = doc_shingles.len().max(1) as f64;
                if let Some(stopgrams) = cfg.stopgrams.as_ref() {
                    doc_shingles.retain(|hash| !stopgrams.contains(*hash));
                    positions.retain(|(hash, _)| !stopgrams.contains(*hash));
                }
                if doc_shingles.is_empty() {
                    empty_after_filter += 1;
                    continue;
                }

                // Per-benchmark mask hits (used for the union gate, the per-source
                // reporting, and the matched-hash sidecar) — computed in both modes.
                // `indexed` is the subset of doc shingles actually present in the
                // active native/attr index; item gating is restricted to it so a
                // stale or mismatched `.items` sidecar cannot decide a scan.
                let mut distinct_hits = 0usize;
                hits_by_src.clear();
                indexed.clear();
                for h in doc_shingles.iter() {
                    if let Some(mask) = cfg.mask_for_hash(*h) {
                        distinct_hits += 1;
                        indexed.insert(*h);
                        if mask != 0 {
                            let mut m = mask;
                            while m != 0 {
                                let bit = m.trailing_zeros();
                                *hits_by_src.entry(bit).or_insert(0) += 1;
                                m &= m - 1;
                            }
                        }
                        let entry = matched_hash_to_mask.entry(*h).or_insert(0);
                        *entry |= mask;
                    }
                }

                // Decide flag + per-item evidence according to the gate mode.
                let (mut flagged, report_count, item_hits_vec) = match cfg.gate_mode {
                    GateMode::Union => {
                        let coverage = (distinct_hits as f64) / denom_all;
                        let flag = distinct_hits >= cfg.min_hits && coverage >= cfg.min_coverage;
                        // Gate on the union (max recall); when item postings are
                        // available, attach per-item attribution for flagged docs.
                        let evidence = if flag {
                            if let Some(postings) = cfg.item_postings.as_ref() {
                                item_gate(
                                    postings,
                                    &indexed,
                                    &positions,
                                    denom_all,
                                    cfg.n,
                                    cfg.min_hits,
                                    cfg.min_coverage,
                                    cfg.min_longest_run,
                                    cfg.boilerplate_threshold,
                                    &mut per_item_hits,
                                    &mut item_run_state,
                                )
                                .2
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        };
                        (flag, distinct_hits, evidence)
                    }
                    GateMode::Item => {
                        let postings = cfg
                            .item_postings
                            .as_ref()
                            .expect("item gate mode requires an item postings index");
                        item_gate(
                            postings,
                            &indexed,
                            &positions,
                            denom_all,
                            cfg.n,
                            cfg.min_hits,
                            cfg.min_coverage,
                            cfg.min_longest_run,
                            cfg.boilerplate_threshold,
                            &mut per_item_hits,
                            &mut item_run_state,
                        )
                    }
                };

                // Spaced-seed recall channel: union recall = exact gate OR spaced
                // verifier. Short-circuit when the exact gate already flagged (the flag
                // is decided; spaced attribution is only meaningful for spaced-only
                // flags). Runs on the already-normalised `toks`, so no re-tokenisation.
                let mut spaced_item: Option<u32> = None;
                if !flagged {
                    if let Some(spaced) = cfg.spaced.as_ref() {
                        spaced_item = spaced.verified_item(&toks);
                        if spaced_item.is_some() {
                            flagged = true;
                        }
                    }
                }

                if flagged {
                    total_contaminated += 1;

                    let sub_doc_id = if let Some(original_id) = original_id_val {
                        if !original_id.is_empty() {
                            format!("{}-part-{}", original_id, sub_doc_index)
                        } else {
                            doc_identity_rust(None, sub_text)
                        }
                    } else {
                        doc_identity_rust(None, sub_text)
                    };

                    let mut src_hits_vec: Vec<(u32, usize)> = Vec::with_capacity(hits_by_src.len());
                    // drain (not into_iter) so the reused buffer is emptied, not moved.
                    for (k, v) in hits_by_src.drain() {
                        if v > 0 {
                            src_hits_vec.push((k, v));
                        }
                    }

                    if limits
                        .max_return_records
                        .is_none_or(|limit| returned_count < limit)
                    {
                        let rec = FileScanResult {
                            schema_version: 2,
                            doc_id: sub_doc_id,
                            match_count: report_count,
                            src_hits: src_hits_vec,
                            item_hits: item_hits_vec,
                            spaced_item,
                        };
                        results_vec.push(serde_json::to_string(&rec).map_err(|e| e.to_string())?);
                        returned_count += 1;
                    }
                }
            }
        }
    }

    if let Some(path) = cfg.out_hits_path {
        let file = File::create(&path)
            .map_err(|e| format!("Failed to create per-hits file '{}': {}", path, e))?;
        let mut w = BufWriter::new(file);
        let mut sorted_pairs: Vec<(u64, u64)> = matched_hash_to_mask.into_iter().collect();
        sorted_pairs.sort_unstable_by_key(|(h, _)| *h);
        write_header(
            &mut w,
            FileType::HitsPairs,
            cfg.n as u32,
            sorted_pairs.len() as u64,
        )
        .map_err(|e| e.to_string())?;
        for (h, m) in sorted_pairs {
            w.write_u64::<LittleEndian>(h).map_err(|e| e.to_string())?;
            w.write_u64::<LittleEndian>(m).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }

    if log_empty_filtered && empty_after_filter > 0 {
        eprintln!(
            "[scan] {} docs had all shingles filtered (stop-grams)",
            empty_after_filter
        );
    }

    Ok((
        total_scanned,
        total_contaminated,
        results_vec,
        empty_after_filter,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_scan(
    text_key: &str,
    id_key: Option<&str>,
    index_path: &str,
    index_backend: &str,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    out_hits_path: Option<&str>,
    packed_doc_sep: &str,
    packed_doc_sep_typo: Option<&str>,
    stopgrams_path: Option<&str>,
    gate_mode: &str,
    min_longest_run: usize,
    boilerplate_threshold: usize,
    spaced_path: Option<&str>,
    spaced_min_loci: usize,
    spaced_ver_min_span: usize,
    spaced_ver_identity: f64,
) -> PyResult<PreparedScan> {
    if min_hits == 0 {
        return Err(PyValueError::new_err("min_hits must be >= 1"));
    }
    if boilerplate_threshold < 2 {
        return Err(PyValueError::new_err(
            "boilerplate_threshold must be >= 2 (a hash in <2 items cannot be boilerplate)",
        ));
    }
    if !(0.0..=1.0).contains(&min_coverage) {
        return Err(PyValueError::new_err("min_coverage must be in [0,1]"));
    }
    if packed_doc_sep.is_empty() {
        return Err(PyValueError::new_err("packed_doc_sep must be non-empty"));
    }
    let index_ngram = get_index_ngram(index_path)?;
    if index_ngram != n {
        return Err(PyValueError::new_err(format!(
            "Index ngram mismatch: index uses {}, scan requested {}. Rebuild the index or use a matching config.",
            index_ngram, n
        )));
    }

    let backend_mode = index_backend.to_lowercase();
    let backend = match backend_mode.as_str() {
        "auto" | "memory" => {
            let index_hashes = get_hash_index(index_path)?;
            let attr_index = get_attr_index(index_path)?;
            IndexBackend::InMemory {
                hashes: index_hashes,
                attr: attr_index,
            }
        }
        "mmap" => IndexBackend::MmapAttr {
            attr: MmapAttrIndex::open_attr(index_path)?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown index_backend '{}'; expected auto, memory, or mmap",
                other
            )))
        }
    };

    let stopgrams = if let Some(path) = stopgrams_path {
        let set = load_stopgrams_cached(path)?;
        if set.ngram() as usize != n {
            return Err(PyValueError::new_err(format!(
                "Stop-grams ngram mismatch: stop-grams uses {}, scan requested {}.",
                set.ngram(),
                n
            )));
        }
        Some(set)
    } else {
        None
    };

    // Union is the default *decision* gate: it maximises recall (item-level gating
    // is a strict subset of union flags) and pays a negligible false-positive cost
    // at n=13. The item gate is an opt-in stricter variant. Item postings, when
    // present, are loaded in either mode to provide per-item *attribution* for the
    // flagged documents.
    let gate_mode = match gate_mode.to_lowercase().as_str() {
        "union" | "auto" => GateMode::Union,
        "item" => GateMode::Item,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown gate_mode '{}'; expected auto, union, or item",
                other
            )))
        }
    };
    let item_postings = if items_sidecar_exists(index_path) {
        Some(get_items_index(index_path)?)
    } else if gate_mode == GateMode::Item {
        return Err(PyValueError::new_err(format!(
            "gate_mode='item' requires an item sidecar '{}.items'; rebuild the index with item postings or use gate_mode='union'.",
            index_path
        )));
    } else {
        None
    };

    let spaced = if let Some(path) = spaced_path {
        if spaced_min_loci == 0 {
            return Err(PyValueError::new_err("spaced_min_loci must be >= 1"));
        }
        if !(0.0..=1.0).contains(&spaced_ver_identity) {
            return Err(PyValueError::new_err(
                "spaced_ver_identity must be in [0,1]",
            ));
        }
        Some(load_spaced_cached(
            path,
            spaced_min_loci,
            spaced_ver_min_span,
            spaced_ver_identity,
        )?)
    } else {
        None
    };

    Ok(PreparedScan {
        text_key: text_key.to_string(),
        id_key: id_key.map(|s| s.to_string()),
        packed_doc_sep: packed_doc_sep.to_string(),
        packed_doc_sep_typo: match packed_doc_sep_typo {
            Some(typo) if !typo.is_empty() => Some(typo.to_string()),
            _ => None,
        },
        backend,
        out_hits_path: out_hits_path.map(|s| s.to_string()),
        n,
        min_hits,
        min_coverage,
        stopgrams,
        gate_mode,
        min_longest_run,
        boilerplate_threshold,
        item_postings,
        spaced,
    })
}

/// Stream a Python file-like object into a seekable temporary file under the GIL.
fn pyfile_to_tempfile(obj: &Bound<'_, PyAny>) -> PyResult<File> {
    const CHUNK_SIZE: usize = 8 * 1024 * 1024;
    let mut temp_file = tempfile::tempfile()
        .map_err(|e| PyValueError::new_err(format!("Failed to create tempfile: {}", e)))?;

    loop {
        // We expect a binary file-like object that supports read(size).
        let data_obj = obj.call_method1("read", (CHUNK_SIZE,))?;
        let bytes: &[u8] = data_obj.extract()?;
        if bytes.is_empty() {
            break;
        }
        temp_file
            .write_all(bytes)
            .map_err(|e| PyValueError::new_err(format!("Failed to write tempfile: {}", e)))?;
    }

    temp_file
        .seek(SeekFrom::Start(0))
        .map_err(|e| PyValueError::new_err(format!("Failed to rewind tempfile: {}", e)))?;
    Ok(temp_file)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (input, text_key, id_key, index_path, n, min_hits, min_coverage, index_backend="auto", out_hits_path=None, packed_doc_sep="<|endoftext|>", packed_doc_sep_typo="<|endoftext}>", max_return_records=Some(500), stopgrams_path=None, gate_mode="auto", min_longest_run=0, spaced_path=None, spaced_min_loci=3, spaced_ver_min_span=17, spaced_ver_identity=0.85, boilerplate_threshold=50))]
pub fn scan_stream_rust(
    py: Python<'_>,
    input: Bound<'_, PyAny>,
    text_key: &str,
    id_key: Option<&str>,
    index_path: &str,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    index_backend: &str,
    out_hits_path: Option<&str>,
    packed_doc_sep: &str,
    packed_doc_sep_typo: Option<&str>,
    max_return_records: Option<usize>,
    stopgrams_path: Option<&str>,
    gate_mode: &str,
    min_longest_run: usize,
    spaced_path: Option<&str>,
    spaced_min_loci: usize,
    spaced_ver_min_span: usize,
    spaced_ver_identity: f64,
    boilerplate_threshold: usize,
) -> PyResult<(usize, usize, Vec<String>, usize)> {
    let cfg = prepare_scan(
        text_key,
        id_key,
        index_path,
        index_backend,
        n,
        min_hits,
        min_coverage,
        out_hits_path,
        packed_doc_sep,
        packed_doc_sep_typo,
        stopgrams_path,
        gate_mode,
        min_longest_run,
        boilerplate_threshold,
        spaced_path,
        spaced_min_loci,
        spaced_ver_min_span,
        spaced_ver_identity,
    )?;

    let limits = ScanLimits { max_return_records };

    let temp_file = pyfile_to_tempfile(&input)?;
    let file_processing_result = py.detach(move || scan_reader(temp_file, cfg, limits));

    match file_processing_result {
        Ok(tuple) => Ok(tuple),
        Err(e_str) => Err(PyValueError::new_err(format!(
            "Rust worker failed for input stream: {}",
            e_str
        ))),
    }
}
