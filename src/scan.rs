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
use arrow::array::{ArrayRef, AsArray, RecordBatchReader};
use arrow::buffer::NullBuffer;
use arrow::compute::can_cast_types;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use once_cell::sync::Lazy;
use parquet::file::reader::ChunkReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyString};
use regex::Regex;

use arrow::array::{Array, StringArray};
use byteorder::{LittleEndian, WriteBytesExt};
use hashbrown::{HashMap, HashSet};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use std::borrow::Cow;
use std::fmt::Write as _;
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
///
/// `field_hits` lists `[field_idx, hits]` for each of the item's spec fields with
/// at least one hit (a hash in several fields of the item counts in each), and
/// `fields_matched` names those fields. Both are absent for `.items` v1 sidecars,
/// which carry no field information.
#[derive(Serialize, Clone)]
struct ItemHit {
    item_id: u32,
    hits: usize,
    longest_run_tokens: usize,
    unique_hits: usize,
    shared_hits: usize,
    boilerplate_hits: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    field_hits: Vec<(u32, usize)>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    fields_matched: Vec<String>,
}

/// Version of the flagged-document records (`FileScanResult`); mirrored by
/// `DOC_RESULT_SCHEMA_VERSION` in `python/corpus_assay/schemas/doc_result.py`.
/// v3 adds `field`, `role` and `turn`.
const DOC_RESULT_SCHEMA_VERSION: u32 = 3;

#[derive(Serialize)]
struct FileScanResult<'a> {
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
    /// Column the sub-document came from: a text key, or the messages key.
    #[serde(skip_serializing_if = "Option::is_none")]
    field: Option<&'a str>,
    /// Chat role of the turn (messages column only, when the turn has one).
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    /// Index of the turn in the messages list (messages column only).
    #[serde(skip_serializing_if = "Option::is_none")]
    turn: Option<u32>,
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
    /// Text columns; each is scanned as its own sub-document(s).
    text_keys: Vec<String>,
    /// Optional chat column (`list<struct{role, content}>`); each turn is its own
    /// sub-document(s).
    messages_key: Option<String>,
    id_key: Option<String>,
    /// Separator that packed records are split on. `None` disables splitting (and
    /// the typo replacement): each record is one document and its id gets no
    /// `-part-<k>` suffix.
    packed_doc_sep: Option<String>,
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
            for posting in items {
                let e = per_item_hits.entry(posting.item).or_insert((0, 0, 0));
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
            for posting in items {
                let e = state.entry(posting.item).or_insert((-1, 0, 0));
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
            field_hits: Vec::new(),
            fields_matched: Vec::new(),
        });
        flagged = true;
    }
    add_field_hits(postings, indexed, &mut evidence);
    // Deterministic ordering: strongest evidence first, then by item id.
    evidence.sort_unstable_by(|a, b| b.hits.cmp(&a.hits).then(a.item_id.cmp(&b.item_id)));
    (flagged, best_hits, evidence)
}

/// Fill `field_hits` / `fields_matched` of each evidence item: the item's
/// matched n-grams counted per field of the item they occur in. Only the few
/// evidence items are counted, in a second pass over the matched hashes.
fn add_field_hits(postings: &ItemPostings, indexed: &HashSet<u64>, evidence: &mut [ItemHit]) {
    if !postings.has_fields || evidence.is_empty() {
        return;
    }
    let mut counts: HashMap<u32, [usize; 32]> = evidence
        .iter()
        .map(|hit| (hit.item_id, [0usize; 32]))
        .collect();
    for h in indexed.iter() {
        let Some(items) = postings.hash_to_items.get(h) else {
            continue;
        };
        for posting in items {
            if let Some(per_field) = counts.get_mut(&posting.item) {
                let mut mask = posting.fields;
                while mask != 0 {
                    per_field[mask.trailing_zeros() as usize] += 1;
                    mask &= mask - 1;
                }
            }
        }
    }
    for hit in evidence.iter_mut() {
        let Some(per_field) = counts.get(&hit.item_id) else {
            continue;
        };
        hit.field_hits = per_field
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(field, &n)| (field as u32, n))
            .collect();
        // Names only when the meta names every matched field, so the two lists
        // line up.
        hit.fields_matched = hit
            .field_hits
            .iter()
            .map(|&(field, _)| postings.field_name(hit.item_id, field).map(str::to_string))
            .collect::<Option<Vec<String>>>()
            .unwrap_or_default();
    }
}

/// The parts of a packed record: split on `sep`, or the whole text when `sep` is
/// `None` (no splitting).
fn split_packed<'a>(text: &'a str, sep: Option<&'a str>) -> impl Iterator<Item = &'a str> + 'a {
    let mut whole = Some(text);
    let mut parts = sep.map(|sep| text.split(sep));
    std::iter::from_fn(move || match parts.as_mut() {
        Some(parts) => parts.next(),
        None => whole.take(),
    })
}

#[derive(Clone)]
struct ScanLimits {
    max_return_records: Option<usize>,
}

/// One text of a record, before packed splitting: a text column (`field` is the
/// column name) or one chat turn of the messages column (`field` is the messages
/// column name, with the turn's `role` and its index `turn` in the list).
struct Segment<'a> {
    field: &'a str,
    role: Option<&'a str>,
    turn: Option<u32>,
    text: &'a str,
}

/// Evidence for a flagged sub-document.
struct Flag {
    match_count: usize,
    src_hits: Vec<(u32, usize)>,
    item_hits: Vec<ItemHit>,
    spaced_item: Option<u32>,
}

enum SubDocOutcome {
    /// Empty or shorter than `2·n` bytes: not counted as scanned.
    Skipped,
    /// Every n-gram was a stop-gram.
    EmptyAfterFilter,
    Clean,
    Flagged(Flag),
}

/// Scores sub-documents. The buffers are reused across sub-documents (cleared
/// on each use) to avoid per-doc allocation in the hot scan loop;
/// `matched_hash_to_mask` accumulates over the whole shard for `hits.bin`.
#[derive(Default)]
struct Scorer {
    matched_hash_to_mask: HashMap<u64, u64>,
    doc_shingles: HashSet<u64>,
    positions: Vec<(u64, usize)>,
    hits_by_src: HashMap<u32, usize>,
    indexed: HashSet<u64>,
    per_item_hits: HashMap<u32, (usize, usize, usize)>,
    item_run_state: HashMap<u32, (i64, usize, usize)>,
}

impl Scorer {
    fn score(&mut self, cfg: &PreparedScan, re: &'static Regex, sub_text: &str) -> SubDocOutcome {
        if sub_text.is_empty() || (sub_text.len() < cfg.n * 2) {
            return SubDocOutcome::Skipped;
        }

        let toks = normalize_text_rust(sub_text, re);
        if toks.len() < cfg.n {
            return SubDocOutcome::Clean;
        }

        // Positional n-gram stream (token order), reused for the unique
        // shingle set, the per-benchmark mask hits, and per-item runs.
        self.positions.clear();
        self.doc_shingles.clear();
        for (pos, win) in toks.windows(cfg.n).enumerate() {
            if win.iter().any(|token| token == FIELD_SEPARATOR) {
                continue;
            }
            let ngram = join_ngram_window(win);
            if !ngram_allowed_rust(&ngram) {
                continue;
            }
            let hash = hash_ngram_text(&ngram);
            self.doc_shingles.insert(hash);
            self.positions.push((hash, pos));
        }
        if self.doc_shingles.is_empty() {
            return SubDocOutcome::Clean;
        }
        let denom_all = self.doc_shingles.len().max(1) as f64;
        if let Some(stopgrams) = cfg.stopgrams.as_ref() {
            self.doc_shingles.retain(|hash| !stopgrams.contains(*hash));
            self.positions
                .retain(|(hash, _)| !stopgrams.contains(*hash));
        }
        if self.doc_shingles.is_empty() {
            return SubDocOutcome::EmptyAfterFilter;
        }

        // Per-benchmark mask hits (used for the union gate, the per-source
        // reporting, and the matched-hash sidecar) — computed in both modes.
        // `indexed` is the subset of doc shingles actually present in the
        // active native/attr index; item gating is restricted to it so a
        // stale or mismatched `.items` sidecar cannot decide a scan.
        let mut distinct_hits = 0usize;
        self.hits_by_src.clear();
        self.indexed.clear();
        for h in self.doc_shingles.iter() {
            if let Some(mask) = cfg.mask_for_hash(*h) {
                distinct_hits += 1;
                self.indexed.insert(*h);
                let mut m = mask;
                while m != 0 {
                    let bit = m.trailing_zeros();
                    *self.hits_by_src.entry(bit).or_insert(0) += 1;
                    m &= m - 1;
                }
                *self.matched_hash_to_mask.entry(*h).or_insert(0) |= mask;
            }
        }

        // Decide flag + per-item evidence according to the gate mode.
        let (mut flagged, match_count, item_hits) = match cfg.gate_mode {
            GateMode::Union => {
                let coverage = (distinct_hits as f64) / denom_all;
                let flag = distinct_hits >= cfg.min_hits && coverage >= cfg.min_coverage;
                // Gate on the union (max recall); when item postings are
                // available, attach per-item attribution for flagged docs.
                let evidence = match cfg.item_postings.as_ref() {
                    Some(postings) if flag => self.item_gate(cfg, postings, denom_all).2,
                    _ => Vec::new(),
                };
                (flag, distinct_hits, evidence)
            }
            GateMode::Item => {
                let postings = cfg
                    .item_postings
                    .as_ref()
                    .expect("item gate mode requires an item postings index");
                self.item_gate(cfg, postings, denom_all)
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
                flagged = spaced_item.is_some();
            }
        }

        if !flagged {
            return SubDocOutcome::Clean;
        }
        // drain (not into_iter) so the reused buffer is emptied, not moved.
        let src_hits: Vec<(u32, usize)> =
            self.hits_by_src.drain().filter(|&(_, v)| v > 0).collect();
        SubDocOutcome::Flagged(Flag {
            match_count,
            src_hits,
            item_hits,
            spaced_item,
        })
    }

    fn item_gate(
        &mut self,
        cfg: &PreparedScan,
        postings: &ItemPostings,
        denom_all: f64,
    ) -> (bool, usize, Vec<ItemHit>) {
        item_gate(
            postings,
            &self.indexed,
            &self.positions,
            denom_all,
            cfg.n,
            cfg.min_hits,
            cfg.min_coverage,
            cfg.min_longest_run,
            cfg.boilerplate_threshold,
            &mut self.per_item_hits,
            &mut self.item_run_state,
        )
    }
}

/// Doc id of one sub-document.
///
/// With one text key and no messages key (the classic single-column scan) the id
/// is `<id>-part-<k>`, as before. With several text keys or a messages key it
/// names the field (and turn) too: `<id>/<field>-part-<k>` or
/// `<id>/<field>/<turn>-part-<k>`. Without splitting (`--no-split`) the
/// `-part-<k>` suffix is dropped. A record without an id falls back to a hash of
/// the sub-document text.
fn sub_doc_id(
    id: Option<&str>,
    name_fields: bool,
    seg: &Segment<'_>,
    part: usize,
    split: bool,
    sub_text: &str,
) -> String {
    let Some(id) = id else {
        return doc_identity_rust(None, sub_text);
    };
    let mut out = String::with_capacity(id.len() + seg.field.len() + 16);
    out.push_str(id);
    if name_fields {
        out.push('/');
        out.push_str(seg.field);
        if let Some(turn) = seg.turn {
            let _ = write!(out, "/{}", turn);
        }
    }
    if split {
        let _ = write!(out, "-part-{}", part);
    }
    out
}

/// Accumulates the per-shard counts and the returned records.
struct ShardScan<'c> {
    cfg: &'c PreparedScan,
    limits: ScanLimits,
    re: &'static Regex,
    /// Whether doc ids name the field (several text keys or a messages key).
    name_fields: bool,
    scorer: Scorer,
    scanned: usize,
    contaminated: usize,
    empty_after_filter: usize,
    results: Vec<String>,
}

impl ShardScan<'_> {
    fn scan_segment(&mut self, id: Option<&str>, seg: &Segment<'_>) -> Result<(), String> {
        if seg.text.is_empty() {
            return Ok(());
        }
        let cfg = self.cfg;
        let packed: Cow<'_, str> = match (&cfg.packed_doc_sep, &cfg.packed_doc_sep_typo) {
            (Some(sep), Some(typo)) if seg.text.contains(typo.as_str()) => {
                Cow::Owned(seg.text.replace(typo.as_str(), sep))
            }
            _ => Cow::Borrowed(seg.text),
        };
        for (part, sub_text) in split_packed(&packed, cfg.packed_doc_sep.as_deref()).enumerate() {
            let flag = match self.scorer.score(cfg, self.re, sub_text) {
                SubDocOutcome::Skipped => continue,
                SubDocOutcome::EmptyAfterFilter => {
                    self.scanned += 1;
                    self.empty_after_filter += 1;
                    continue;
                }
                SubDocOutcome::Clean => {
                    self.scanned += 1;
                    continue;
                }
                SubDocOutcome::Flagged(flag) => flag,
            };
            self.scanned += 1;
            self.contaminated += 1;
            if self
                .limits
                .max_return_records
                .is_some_and(|limit| self.results.len() >= limit)
            {
                continue;
            }
            let rec = FileScanResult {
                schema_version: DOC_RESULT_SCHEMA_VERSION,
                doc_id: sub_doc_id(
                    id,
                    self.name_fields,
                    seg,
                    part,
                    cfg.packed_doc_sep.is_some(),
                    sub_text,
                ),
                match_count: flag.match_count,
                src_hits: flag.src_hits,
                item_hits: flag.item_hits,
                spaced_item: flag.spaced_item,
                field: Some(seg.field),
                role: seg.role,
                turn: seg.turn,
            };
            self.results
                .push(serde_json::to_string(&rec).map_err(|e| e.to_string())?);
        }
        Ok(())
    }
}

/// A string column as a `StringArray`: `Utf8` as is; `LargeUtf8`, `Utf8View`,
/// string dictionaries and all-null columns are cast.
fn utf8_column(col: &ArrayRef, what: &str, name: &str) -> Result<StringArray, String> {
    let castable = match col.data_type() {
        DataType::Utf8 => return Ok(col.as_string::<i32>().clone()),
        DataType::LargeUtf8 | DataType::Utf8View | DataType::Null => true,
        DataType::Dictionary(_, values) => matches!(
            values.as_ref(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ),
        _ => false,
    };
    if !castable {
        return Err(format!(
            "{} column '{}' has type {}; expected a string column",
            what,
            name,
            col.data_type()
        ));
    }
    cast_to_utf8(col, what, name)
}

fn cast_to_utf8(col: &ArrayRef, what: &str, name: &str) -> Result<StringArray, String> {
    let cast = arrow::compute::cast(col, &DataType::Utf8)
        .map_err(|e| format!("Cannot read {} column '{}' as text: {}", what, name, e))?;
    Ok(cast.as_string::<i32>().clone())
}

/// The id column as strings. Non-string ids (integers and so on) are cast,
/// rather than silently ignored in favour of text hashes.
fn id_column(col: &ArrayRef, name: &str) -> Result<StringArray, String> {
    match col.data_type() {
        DataType::Utf8 => Ok(col.as_string::<i32>().clone()),
        dt if can_cast_types(dt, &DataType::Utf8) => cast_to_utf8(col, "Id", name),
        other => Err(format!(
            "Id column '{}' has type {}, which cannot be used as a doc id",
            name, other
        )),
    }
}

/// A chat column, `list<struct{role, content}>`, flattened for one batch.
struct MessagesColumn {
    /// Row `i` holds turns `offsets[i]..offsets[i + 1]`.
    offsets: Vec<usize>,
    row_nulls: Option<NullBuffer>,
    turn_nulls: Option<NullBuffer>,
    roles: Option<StringArray>,
    contents: StringArray,
}

impl MessagesColumn {
    fn new(col: &ArrayRef, name: &str) -> Result<Self, String> {
        let (offsets, values, row_nulls): (Vec<usize>, ArrayRef, Option<NullBuffer>) =
            match col.data_type() {
                DataType::List(_) => {
                    let list = col.as_list::<i32>();
                    (
                        list.value_offsets().iter().map(|&o| o as usize).collect(),
                        Arc::clone(list.values()),
                        list.nulls().cloned(),
                    )
                }
                DataType::LargeList(_) => {
                    let list = col.as_list::<i64>();
                    (
                        list.value_offsets().iter().map(|&o| o as usize).collect(),
                        Arc::clone(list.values()),
                        list.nulls().cloned(),
                    )
                }
                DataType::Null => {
                    return Ok(Self {
                        offsets: vec![0; col.len() + 1],
                        row_nulls: None,
                        turn_nulls: None,
                        roles: None,
                        contents: StringArray::from(Vec::<&str>::new()),
                    })
                }
                other => {
                    return Err(format!(
                "Messages column '{}' has type {}; expected a list of {{role, content}} structs",
                name, other
            ))
                }
            };
        let turns = values.as_struct_opt().ok_or_else(|| {
            format!(
                "Messages column '{}' holds {} items; expected {{role, content}} structs",
                name,
                values.data_type()
            )
        })?;
        let contents = turns.column_by_name("content").ok_or_else(|| {
            format!(
                "Messages column '{}' has no 'content' field in its items",
                name
            )
        })?;
        let contents = utf8_column(contents, "Messages content", name)?;
        let roles = turns
            .column_by_name("role")
            .map(|roles| utf8_column(roles, "Messages role", name))
            .transpose()?;
        Ok(Self {
            offsets,
            row_nulls,
            turn_nulls: turns.nulls().cloned(),
            roles,
            contents,
        })
    }

    /// `(turn index, role, content)` for each turn of `row` with content.
    fn turns(&self, row: usize) -> impl Iterator<Item = (u32, Option<&str>, &str)> + '_ {
        let (start, end) = if self.row_nulls.as_ref().is_some_and(|n| n.is_null(row)) {
            (0, 0)
        } else {
            (self.offsets[row], self.offsets[row + 1])
        };
        (start..end).filter_map(move |j| {
            if self.turn_nulls.as_ref().is_some_and(|n| n.is_null(j)) || self.contents.is_null(j) {
                return None;
            }
            let role = self
                .roles
                .as_ref()
                .filter(|roles| roles.is_valid(j))
                .map(|roles| roles.value(j));
            Some(((j - start) as u32, role, self.contents.value(j)))
        })
    }
}

/// Scan a Parquet reader (must be Read+Seek) for contamination.
fn scan_reader<R>(reader: R, cfg: PreparedScan, limits: ScanLimits) -> Result<ScanOutput, String>
where
    R: ChunkReader + 'static,
{
    let mut arrow_reader = ParquetRecordBatchReaderBuilder::try_new(reader)
        .and_then(|b| b.with_batch_size(1024).build())
        .map_err(|e| e.to_string())?;

    let schema = arrow_reader.schema();
    let mut text_cols: Vec<(&str, usize)> = Vec::with_capacity(cfg.text_keys.len());
    for key in &cfg.text_keys {
        let idx = schema
            .index_of(key)
            .map_err(|e| format!("Text key '{}' not found in schema: {}", key, e))?;
        text_cols.push((key.as_str(), idx));
    }
    let messages_col: Option<(&str, usize)> = match cfg.messages_key.as_deref() {
        Some(key) => Some((
            key,
            schema
                .index_of(key)
                .map_err(|e| format!("Messages key '{}' not found in schema: {}", key, e))?,
        )),
        None => None,
    };
    // A missing id column falls back to text hashes (the Python side warns).
    let id_col: Option<(&str, usize)> = cfg
        .id_key
        .as_deref()
        .and_then(|key| schema.index_of(key).ok().map(|idx| (key, idx)));

    let log_empty_filtered = std::env::var_os("CORPUS_ASSAY_SCAN_LOG_EMPTY_FILTERED").is_some();
    let mut shard = ShardScan {
        cfg: &cfg,
        limits,
        re: get_word_re(),
        name_fields: cfg.text_keys.len() != 1 || cfg.messages_key.is_some(),
        scorer: Scorer::default(),
        scanned: 0,
        contaminated: 0,
        empty_after_filter: 0,
        results: Vec::new(),
    };

    for batch_result in arrow_reader.by_ref() {
        let batch: RecordBatch = batch_result.map_err(|e| e.to_string())?;

        let texts: Vec<(&str, StringArray)> = text_cols
            .iter()
            .map(|&(key, idx)| Ok((key, utf8_column(batch.column(idx), "Text", key)?)))
            .collect::<Result<_, String>>()?;
        let messages: Option<(&str, MessagesColumn)> = match messages_col {
            Some((key, idx)) => Some((key, MessagesColumn::new(batch.column(idx), key)?)),
            None => None,
        };
        let ids: Option<StringArray> = match id_col {
            Some((key, idx)) => Some(id_column(batch.column(idx), key)?),
            None => None,
        };

        for row in 0..batch.num_rows() {
            let id: Option<&str> = ids
                .as_ref()
                .filter(|ids| ids.is_valid(row))
                .map(|ids| ids.value(row))
                .filter(|id| !id.is_empty());

            for (field, array) in &texts {
                if array.is_valid(row) {
                    let seg = Segment {
                        field,
                        role: None,
                        turn: None,
                        text: array.value(row),
                    };
                    shard.scan_segment(id, &seg)?;
                }
            }
            if let Some((field, messages)) = &messages {
                for (turn, role, text) in messages.turns(row) {
                    let seg = Segment {
                        field,
                        role,
                        turn: Some(turn),
                        text,
                    };
                    shard.scan_segment(id, &seg)?;
                }
            }
        }
    }

    if let Some(path) = cfg.out_hits_path.as_deref() {
        let file = File::create(path)
            .map_err(|e| format!("Failed to create per-hits file '{}': {}", path, e))?;
        let mut w = BufWriter::new(file);
        let mut sorted_pairs: Vec<(u64, u64)> =
            std::mem::take(&mut shard.scorer.matched_hash_to_mask)
                .into_iter()
                .collect();
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

    if log_empty_filtered && shard.empty_after_filter > 0 {
        eprintln!(
            "[scan] {} docs had all shingles filtered (stop-grams)",
            shard.empty_after_filter
        );
    }

    Ok((
        shard.scanned,
        shard.contaminated,
        shard.results,
        shard.empty_after_filter,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_scan(
    text_keys: Vec<String>,
    messages_key: Option<&str>,
    id_key: Option<&str>,
    index_path: &str,
    index_backend: &str,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    out_hits_path: Option<&str>,
    packed_doc_sep: Option<&str>,
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
    if text_keys.is_empty() && messages_key.is_none() {
        return Err(PyValueError::new_err(
            "at least one text key or a messages key is required",
        ));
    }
    {
        let mut seen: HashSet<&str> = HashSet::new();
        for key in text_keys.iter().map(String::as_str).chain(messages_key) {
            if key.is_empty() || !seen.insert(key) {
                return Err(PyValueError::new_err(format!(
                    "text and messages keys must be non-empty and distinct; got '{}' twice or empty",
                    key
                )));
            }
        }
    }
    if boilerplate_threshold < 2 {
        return Err(PyValueError::new_err(
            "boilerplate_threshold must be >= 2 (a hash in <2 items cannot be boilerplate)",
        ));
    }
    if !(0.0..=1.0).contains(&min_coverage) {
        return Err(PyValueError::new_err("min_coverage must be in [0,1]"));
    }
    if packed_doc_sep == Some("") {
        return Err(PyValueError::new_err(
            "packed_doc_sep must be non-empty (pass None to disable splitting)",
        ));
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
        text_keys,
        messages_key: messages_key.map(str::to_string),
        id_key: id_key.map(|s| s.to_string()),
        packed_doc_sep: packed_doc_sep.map(str::to_string),
        // The typo variant is only rewritten into the separator; with splitting
        // off there is nothing to rewrite it into.
        packed_doc_sep_typo: match (packed_doc_sep, packed_doc_sep_typo) {
            (Some(_), Some(typo)) if !typo.is_empty() => Some(typo.to_string()),
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
#[pyo3(signature = (input, text_key, id_key, index_path, n, min_hits, min_coverage, index_backend="auto", out_hits_path=None, packed_doc_sep=Some("<|endoftext|>"), packed_doc_sep_typo=Some("<|endoftext}>"), max_return_records=Some(500), stopgrams_path=None, gate_mode="auto", min_longest_run=0, spaced_path=None, spaced_min_loci=3, spaced_ver_min_span=17, spaced_ver_identity=0.85, boilerplate_threshold=50, messages_key=None))]
pub fn scan_stream_rust(
    py: Python<'_>,
    input: Bound<'_, PyAny>,
    text_key: Bound<'_, PyAny>,
    id_key: Option<&str>,
    index_path: &str,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    index_backend: &str,
    out_hits_path: Option<&str>,
    packed_doc_sep: Option<&str>,
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
    messages_key: Option<&str>,
) -> PyResult<(usize, usize, Vec<String>, usize)> {
    // `text_key` is one column name or a list of them.
    let text_keys: Vec<String> = if text_key.is_instance_of::<PyString>() {
        vec![text_key.extract()?]
    } else {
        text_key.extract()?
    };
    let cfg = prepare_scan(
        text_keys,
        messages_key,
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

    // `None` and `Some(0)` both mean "no cap": 0 used to return no records at all,
    // contradicting the documented "use 0 to disable".
    let limits = ScanLimits {
        max_return_records: max_return_records.filter(|&limit| limit > 0),
    };

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
