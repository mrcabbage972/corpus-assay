//! Text channels over the raw item text (`<index>.items.text.jsonl`):
//!
//! - the **exact-match channel**: an Aho-Corasick automaton over the
//!   light-normalized question (first field) of each eligible item, matched over
//!   each light-normalized sub-document with a word-boundary check;
//! - **choices near the question**: whether each other field of an item (each
//!   element, for list fields such as multiple-choice options) occurs within
//!   [`CHOICE_WINDOW_CHARS`] characters of the matched question.
//!
//! All of it works on [`light_normalize_rust`] text, which keeps numbers,
//! operators, punctuation and stop words.

use crate::index::cache_key;
use crate::items_index::ItemPostings;
use crate::text::light_normalize_rust;
use aho_corasick::AhoCorasick;
use hashbrown::{HashMap, HashSet};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};

/// Characters on either side of the matched question searched for its choices.
pub const CHOICE_WINDOW_CHARS: usize = 1000;

/// One field of an item as stored in the text sidecar: a string, or a list of
/// strings (a list-valued field such as multiple-choice options).
#[derive(Deserialize)]
#[serde(untagged)]
enum FieldValue {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ItemTextLine {
    item_idx: u32,
    fields: Vec<FieldValue>,
}

/// The light-normalized text of one item.
pub struct ItemText {
    /// The first field (the question). Empty when the item has none.
    pub question: String,
    /// Length of `question` in characters.
    question_chars: usize,
    /// Every other field, each list element separately; empty ones dropped.
    pub choices: Vec<String>,
}

impl ItemText {
    fn new(fields: Vec<FieldValue>) -> Self {
        let mut fields = fields.into_iter();
        let question = match fields.next() {
            Some(FieldValue::One(s)) => light_normalize_rust(&s),
            Some(FieldValue::Many(parts)) => light_normalize_rust(&parts.join(" ")),
            None => String::new(),
        };
        let mut choices = Vec::new();
        for field in fields {
            match field {
                FieldValue::One(s) => choices.push(light_normalize_rust(&s)),
                FieldValue::Many(parts) => {
                    choices.extend(parts.iter().map(|p| light_normalize_rust(p)))
                }
            }
        }
        choices.retain(|c| !c.is_empty());
        Self {
            question_chars: question.chars().count(),
            question,
            choices,
        }
    }
}

/// Every item's text, by item id.
pub struct ItemTexts {
    items: Vec<Option<ItemText>>,
}

impl ItemTexts {
    pub fn get(&self, item: u32) -> Option<&ItemText> {
        self.items.get(item as usize).and_then(Option::as_ref)
    }
}

pub fn items_text_path(index_path: &str) -> String {
    format!("{}.items.text.jsonl", index_path)
}

pub fn items_text_exists(index_path: &str) -> bool {
    std::path::Path::new(&items_text_path(index_path)).exists()
}

/// Upper bound on item ids accepted from the text sidecar (a dense table).
const MAX_ITEMS: usize = 1 << 28;

fn load_item_texts(index_path: &str) -> PyResult<ItemTexts> {
    let path = items_text_path(index_path);
    let file = File::open(&path).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to open item text sidecar '{}': {}",
            path, e
        ))
    })?;
    let mut items: Vec<Option<ItemText>> = Vec::new();
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| {
            PyValueError::new_err(format!("Error reading item text sidecar '{}': {}", path, e))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: ItemTextLine = serde_json::from_str(&line).map_err(|e| {
            PyValueError::new_err(format!(
                "Invalid line {} in item text sidecar '{}': {}",
                lineno + 1,
                path,
                e
            ))
        })?;
        let idx = rec.item_idx as usize;
        if idx >= MAX_ITEMS {
            return Err(PyValueError::new_err(format!(
                "Item id {} in item text sidecar '{}' exceeds the supported maximum",
                rec.item_idx, path
            )));
        }
        if idx >= items.len() {
            items.resize_with(idx + 1, || None);
        }
        items[idx] = Some(ItemText::new(rec.fields));
    }
    Ok(ItemTexts { items })
}

#[allow(clippy::type_complexity)]
static ITEM_TEXTS: Lazy<Mutex<std::collections::HashMap<String, Arc<ItemTexts>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

pub fn get_item_texts(index_path: &str) -> PyResult<Arc<ItemTexts>> {
    let key = cache_key(index_path);
    {
        let cache = ITEM_TEXTS
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock item-text cache"))?;
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.clone());
        }
    }
    let arc = Arc::new(load_item_texts(index_path)?);
    let mut cache = ITEM_TEXTS
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock item-text cache"))?;
    Ok(cache.entry(key).or_insert(arc).clone())
}

/// Which items the exact channel matches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExactMode {
    /// Items with fewer than `min_hits` indexed n-grams (the n-gram rule cannot
    /// flag them).
    Short,
    /// Every item.
    All,
}

/// The exact-match automaton: one pattern per distinct eligible question.
pub struct ExactMatcher {
    ac: Option<AhoCorasick>,
    /// Items per pattern id (items with the same question share a pattern).
    pattern_items: Vec<Vec<u32>>,
}

impl ExactMatcher {
    fn build(
        texts: &ItemTexts,
        postings: Option<&ItemPostings>,
        mode: ExactMode,
        min_chars: usize,
        min_hits: usize,
    ) -> PyResult<Self> {
        let mut patterns: Vec<&str> = Vec::new();
        let mut pattern_items: Vec<Vec<u32>> = Vec::new();
        let mut by_pattern: HashMap<&str, usize> = HashMap::new();
        for (idx, item) in texts.items.iter().enumerate() {
            let Some(item) = item else { continue };
            if item.question_chars < min_chars {
                continue;
            }
            // Items without n-grams have no postings at all: 0.
            let ngrams = postings.map_or(0, |p| p.ngrams_of(idx as u32));
            if mode == ExactMode::Short && ngrams >= min_hits {
                continue;
            }
            let pid = *by_pattern.entry(item.question.as_str()).or_insert_with(|| {
                patterns.push(item.question.as_str());
                pattern_items.push(Vec::new());
                patterns.len() - 1
            });
            pattern_items[pid].push(idx as u32);
        }
        let ac = if patterns.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&patterns).map_err(|e| {
                PyValueError::new_err(format!("Failed to build the exact-match automaton: {}", e))
            })?)
        };
        Ok(Self { ac, pattern_items })
    }

    /// `(item, start, end)` of the first word-bounded occurrence of each matched
    /// item's question in the light-normalized `text`, sorted by item.
    pub fn find(&self, text: &str) -> Vec<(u32, usize, usize)> {
        let Some(ac) = self.ac.as_ref() else {
            return Vec::new();
        };
        let mut seen: HashSet<usize> = HashSet::new();
        let mut out: Vec<(u32, usize, usize)> = Vec::new();
        for m in ac.find_overlapping_iter(text) {
            let pid = m.pattern().as_usize();
            if seen.contains(&pid) || !word_bounded(text, m.start(), m.end()) {
                continue;
            }
            seen.insert(pid);
            out.extend(
                self.pattern_items[pid]
                    .iter()
                    .map(|&item| (item, m.start(), m.end())),
            );
        }
        out.sort_unstable_by_key(|hit| hit.0);
        out
    }
}

#[allow(clippy::type_complexity)]
static EXACT_MATCHERS: Lazy<Mutex<std::collections::HashMap<String, Arc<ExactMatcher>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// The automaton for an index and settings, built once per process (shared by
/// every shard a worker scans).
pub fn get_exact_matcher(
    index_path: &str,
    texts: &ItemTexts,
    postings: Option<&ItemPostings>,
    mode: ExactMode,
    min_chars: usize,
    min_hits: usize,
) -> PyResult<Arc<ExactMatcher>> {
    let key = format!(
        "{}|{:?}|{}|{}",
        cache_key(index_path),
        mode,
        min_chars,
        min_hits
    );
    {
        let cache = EXACT_MATCHERS
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock exact-matcher cache"))?;
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.clone());
        }
    }
    let arc = Arc::new(ExactMatcher::build(
        texts, postings, mode, min_chars, min_hits,
    )?);
    let mut cache = EXACT_MATCHERS
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock exact-matcher cache"))?;
    Ok(cache.entry(key).or_insert(arc).clone())
}

fn joins(a: Option<char>, b: Option<char>) -> bool {
    matches!((a, b), (Some(a), Some(b)) if a.is_alphanumeric() && b.is_alphanumeric())
}

/// Whether `text[start..end]` sits on word boundaries: an alphanumeric
/// character of the match may not touch an alphanumeric character outside it
/// (so `3 apples` does not match inside `23 apples`; a match ending in `?` may
/// be followed by anything).
pub fn word_bounded(text: &str, start: usize, end: usize) -> bool {
    let inner = &text[start..end];
    !joins(text[..start].chars().next_back(), inner.chars().next())
        && !joins(inner.chars().next_back(), text[end..].chars().next())
}

/// First word-bounded occurrence of `needle` in `text[lo..hi]`.
pub fn find_bounded(text: &str, lo: usize, hi: usize, needle: &str) -> Option<(usize, usize)> {
    if needle.is_empty() {
        return None;
    }
    text[lo..hi]
        .match_indices(needle)
        .map(|(at, _)| (lo + at, lo + at + needle.len()))
        .find(|&(s, e)| word_bounded(text, s, e))
}

/// Byte range of `text` spanning `radius` characters on either side of
/// `start..end`.
fn char_window(text: &str, start: usize, end: usize, radius: usize) -> (usize, usize) {
    let lo = text[..start]
        .char_indices()
        .rev()
        .nth(radius.saturating_sub(1))
        .map_or(0, |(i, _)| i);
    let hi = text[end..]
        .char_indices()
        .nth(radius)
        .map_or(text.len(), |(i, _)| end + i);
    (lo, hi)
}

/// Choices of `item` found within [`CHOICE_WINDOW_CHARS`] of `span` (the matched
/// question) in the light-normalized `text`, or anywhere in it when the
/// question's location is unknown: `(found, total)`.
pub fn choices_near(item: &ItemText, text: &str, span: Option<(usize, usize)>) -> (usize, usize) {
    let (lo, hi) = match span {
        Some((start, end)) => char_window(text, start, end, CHOICE_WINDOW_CHARS),
        None => (0, text.len()),
    };
    let found = item
        .choices
        .iter()
        .filter(|choice| find_bounded(text, lo, hi, choice).is_some())
        .count();
    (found, item.choices.len())
}
