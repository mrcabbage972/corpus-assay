use blake2::digest::consts::{U16, U32, U8};
use blake2::digest::Digest;
use blake2::Blake2b;
use hashbrown::HashSet;
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use serde::Deserialize;
use sha2::Sha256;
use std::cell::RefCell;
use std::sync::Mutex;
use unicode_normalization::UnicodeNormalization;

static WORD_RE: OnceCell<Regex> = OnceCell::new();
static STOP_WORDS: OnceCell<HashSet<String>> = OnceCell::new();
static NGRAM_REJECT_REGEXES: OnceCell<Vec<Regex>> = OnceCell::new();
static CONFIG: OnceCell<NgramConfig> = OnceCell::new();
static CONFIG_HASH: OnceCell<String> = OnceCell::new();
static CONFIG_SHA256: OnceCell<String> = OnceCell::new();
pub const FIELD_SEPARATOR: &str = "<|field_sep|>";
pub const HASH_FUNCTION_NAME: &str = "blake2b-64-le";
pub const HASH_FUNCTION_VERSION: &str = "blake2 0.10";

thread_local! {
    static NORM_BUF: RefCell<String> = RefCell::new(String::with_capacity(1024));
}

#[derive(Deserialize)]
struct NgramConfig {
    version: String,
    stop_words: Vec<String>,
    word_re: String,
    ngram_reject_patterns: Vec<String>,
}

/// Config text staged from Python, awaiting first use.
///
/// The config path is owned entirely by Python — see the `corpus_assay`
/// package's single `DEFAULT_NGRAM_CONFIG_PATH` constant and the
/// `set_ngram_config_path` pyfunction. This crate never hardcodes a config
/// path. The staged value may be replaced repeatedly (e.g. the package default,
/// then a CLI override) up until the first use of any normalization function.
static PENDING_CONFIG: Mutex<Option<String>> = Mutex::new(None);

/// Raw JSON config text frozen for this process on first normalization use.
static CONFIG_RAW: OnceCell<String> = OnceCell::new();

/// Raw JSON config text in effect for this process.
///
/// Resolves once, from whatever Python last staged via `set_ngram_config_path`.
/// Every derived value — stop-words, word regex, reject patterns, and the
/// `ngram_config_hash` / `normalization_config_sha` fingerprints — flows through
/// this function, so the active config is reflected consistently everywhere.
/// Panics if no config was staged; the `corpus_assay` Python package
/// stages the bundled default on import, so that is unreachable in normal use.
fn config_raw() -> &'static str {
    CONFIG_RAW
        .get_or_init(|| {
            PENDING_CONFIG
                .lock()
                .expect("ngram config lock poisoned")
                .take()
                .expect(
                    "ngram config not set: call \
                     corpus_assay.normalization.set_ngram_config_path(path) before \
                     normalization (the Python package does this on import)",
                )
        })
        .as_str()
}

/// Stage `text` as the ngram config.
///
/// Validates that it parses as an `NgramConfig`. Callable repeatedly until the
/// config is frozen on first normalization use; after freezing, re-staging the
/// identical text is a no-op and staging different text is an error.
///
/// The `CONFIG_RAW` check is performed while holding the `PENDING_CONFIG` lock
/// to serialize against `config_raw`'s `get_or_init` closure (which also takes
/// that lock to consume the staged value). Without this, a `stage` call could
/// observe `CONFIG_RAW == None`, race with a concurrent freeze, and then write
/// to `PENDING_CONFIG` after the freeze has already consumed the prior value —
/// silently dropping the caller's config while returning `Ok`.
fn stage_ngram_config(text: String) -> PyResult<()> {
    serde_json::from_str::<NgramConfig>(&text)
        .map_err(|e| PyValueError::new_err(format!("invalid ngram config JSON: {e}")))?;
    let mut pending = PENDING_CONFIG.lock().expect("ngram config lock poisoned");
    if let Some(frozen) = CONFIG_RAW.get() {
        if frozen == &text {
            return Ok(());
        }
        return Err(PyRuntimeError::new_err(
            "ngram config is already frozen for this process (normalization has \
             already run); it cannot be changed",
        ));
    }
    *pending = Some(text);
    Ok(())
}

fn get_config() -> &'static NgramConfig {
    CONFIG.get_or_init(|| {
        serde_json::from_str(config_raw()).expect("Failed to parse ngram_config.json")
    })
}

fn get_stop_words() -> &'static HashSet<String> {
    STOP_WORDS.get_or_init(|| get_config().stop_words.iter().cloned().collect())
}

fn get_ngram_reject_regexes() -> &'static Vec<Regex> {
    NGRAM_REJECT_REGEXES.get_or_init(|| {
        get_config()
            .ngram_reject_patterns
            .iter()
            .map(|p| Regex::new(&format!("(?i){}", p)).expect("Invalid reject pattern"))
            .collect()
    })
}

pub fn ngram_config_hash_hex() -> &'static str {
    CONFIG_HASH.get_or_init(|| {
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(config_raw().as_bytes());
        let res: [u8; 32] = hasher.finalize().into();
        hex::encode(res)
    })
}

pub fn ngram_config_version_str() -> &'static str {
    &get_config().version
}

pub fn normalization_config_sha_hex() -> &'static str {
    CONFIG_SHA256.get_or_init(|| {
        let mut hasher = Sha256::new();
        hasher.update(config_raw().as_bytes());
        let res = hasher.finalize();
        hex::encode(res)
    })
}

pub fn get_word_re() -> &'static Regex {
    WORD_RE.get_or_init(|| {
        let config = get_config();
        Regex::new(&config.word_re).expect("Invalid word regex")
    })
}

fn normalize_text_fragment(s: &str, re: &'static Regex) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let stop_words = get_stop_words();
    NORM_BUF.with_borrow_mut(|norm_buf| {
        norm_buf.clear();
        s.nfkc().for_each(|c| norm_buf.push(c));
        let lower_s = norm_buf.to_lowercase();
        re.find_iter(&lower_s)
            .map(|m| m.as_str())
            .filter(|s| !stop_words.contains(*s))
            .map(|s| s.to_string())
            .collect()
    })
}

pub fn normalize_text_rust(s: &str, re: &'static Regex) -> Vec<String> {
    if !s.contains(FIELD_SEPARATOR) {
        return normalize_text_fragment(s, re);
    }

    let mut tokens: Vec<String> = Vec::new();
    for (idx, part) in s.split(FIELD_SEPARATOR).enumerate() {
        if idx > 0 {
            tokens.push(FIELD_SEPARATOR.to_string());
        }
        if part.is_empty() {
            continue;
        }
        tokens.extend(normalize_text_fragment(part, re));
    }
    tokens
}

pub fn hash_ngram_text(ngram: &str) -> u64 {
    let mut hasher = Blake2b::<U8>::new();
    hasher.update(ngram.as_bytes());
    let res: [u8; 8] = hasher.finalize().into();
    u64::from_le_bytes(res)
}

pub fn ngram_allowed_rust(ngram: &str) -> bool {
    ngram_filter_decision(ngram).0
}

pub fn ngram_filter_decision(ngram: &str) -> (bool, Option<usize>) {
    if ngram.trim().is_empty() {
        return (false, None);
    }

    // Hard boundary: never allow n-grams that cross a field boundary.
    // normalize_text_rust inserts FIELD_SEPARATOR as its own token, so any
    // cross-field window will include it.
    if ngram.contains(FIELD_SEPARATOR) {
        return (false, None);
    }

    for (idx, rx) in get_ngram_reject_regexes().iter().enumerate() {
        if rx.is_match(ngram) {
            return (false, Some(idx));
        }
    }
    (true, None)
}

pub fn join_ngram_window(win: &[String]) -> String {
    let mut ngram = String::with_capacity(win.iter().map(|w| w.len()).sum::<usize>() + win.len());
    for (j, word) in win.iter().enumerate() {
        if j > 0 {
            ngram.push(' ');
        }
        ngram.push_str(word);
    }
    ngram
}

pub fn doc_identity_rust(id_val: Option<&str>, text: &str) -> String {
    if let Some(id_str) = id_val {
        if !id_str.is_empty() {
            return id_str.to_string();
        }
    }
    let sample = if text.len() <= 1024 {
        text
    } else {
        let mut end_idx = 1024;
        while !text.is_char_boundary(end_idx) {
            end_idx -= 1;
        }
        &text[..end_idx]
    };
    let mut hasher = Blake2b::<U16>::new();
    hasher.update(sample.as_bytes());
    let res: [u8; 16] = hasher.finalize().into();
    hex::encode(res)
}

/// Set the ngram normalization config from a JSON file path.
///
/// Python owns the path — the bundled default plus any CLI override — and this
/// crate never hardcodes one. Must be called before any normalization runs;
/// the `corpus_assay` package calls it with the bundled default on
/// import. May be called again to override (e.g. from a CLI flag) up until the
/// first normalization, after which the config is frozen for the process.
#[pyfunction]
pub fn set_ngram_config_path(path: &str) -> PyResult<()> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| PyValueError::new_err(format!("failed to read ngram config '{path}': {e}")))?;
    stage_ngram_config(text)
}

#[pyfunction]
pub fn normalize_text(s: &str) -> PyResult<Vec<String>> {
    Ok(normalize_text_rust(s, get_word_re()))
}

#[pyfunction]
pub fn ngram_allowed(ngram: &str) -> PyResult<bool> {
    Ok(ngram_allowed_rust(ngram))
}

#[pyfunction]
pub fn hash_ngram(ngram: &str) -> PyResult<u64> {
    Ok(hash_ngram_text(ngram))
}

#[pyfunction]
pub fn ngram_filter(ngram: &str) -> PyResult<(bool, Option<usize>)> {
    Ok(ngram_filter_decision(ngram))
}

#[pyfunction]
pub fn ngram_config_version() -> PyResult<String> {
    Ok(ngram_config_version_str().to_string())
}

#[pyfunction]
pub fn ngram_config_hash() -> PyResult<String> {
    Ok(ngram_config_hash_hex().to_string())
}

#[pyfunction]
pub fn normalization_config_sha() -> PyResult<String> {
    Ok(normalization_config_sha_hex().to_string())
}

#[pyfunction]
pub fn hash_function_metadata(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("name", HASH_FUNCTION_NAME)?;
    dict.set_item("version", HASH_FUNCTION_VERSION)?;
    Ok(dict.into())
}
