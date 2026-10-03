use crate::formats::{read_header_optional, FileType, HEADER_SIZE};
use byteorder::{LittleEndian, ReadBytesExt};
use hashbrown::{HashMap, HashSet};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Deserialize, Serialize)]
struct HashFunctionMetadata {
    name: String,
    version: String,
}

// Caches are keyed by the canonical index path so a process can hold several
// distinct indexes at once (e.g. tests, or multi-index tooling) without the
// first-loaded one shadowing the rest.
#[allow(clippy::type_complexity)]
static HASH_INDEX: Lazy<Mutex<std::collections::HashMap<String, Arc<HashSet<u64>>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));
#[allow(clippy::type_complexity)]
static ATTR_INDEX: Lazy<Mutex<std::collections::HashMap<String, Arc<HashMap<u64, u64>>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));
static INDEX_NGRAM: Lazy<Mutex<std::collections::HashMap<String, usize>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

pub fn cache_key(path: &str) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| Path::new(path).to_path_buf())
        .to_string_lossy()
        .to_string()
}

#[derive(Debug, Deserialize, Serialize)]
struct IndexMetadata {
    #[serde(default)]
    schema_version: Option<u32>,
    ngram: usize,
    #[serde(default)]
    ngram_config_version: Option<String>,
    #[serde(default)]
    ngram_config_hash: Option<String>,
    #[serde(default)]
    normalization_config_sha: Option<String>,
    #[serde(default)]
    index_format_version: Option<u32>,
    #[serde(default)]
    hash_function: Option<HashFunctionMetadata>,
}

fn metadata_path(index_path: &str) -> String {
    format!("{}.meta.json", index_path)
}

fn load_index_from_native(path: &str) -> PyResult<HashSet<u64>> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open index file: {}", e)))?;
    let file_size = std::fs::metadata(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to stat index file: {}", e)))?
        .len();
    let mut reader = BufReader::new(file);
    let header = read_header_optional(&mut reader, Some(FileType::NativeHashes))
        .map_err(|e| PyValueError::new_err(format!("Error reading index header: {}", e)))?;
    let mut hashes = HashSet::new();
    if let Some(h) = header {
        let data_size = file_size.saturating_sub(HEADER_SIZE);
        let max_records = data_size / 8;
        if h.record_count > max_records {
            return Err(PyValueError::new_err(format!(
                "Index header record count {} exceeds max possible {}",
                h.record_count, max_records
            )));
        }
        if data_size != h.record_count * 8 {
            return Err(PyValueError::new_err(format!(
                "Index size {} does not match header record count {}",
                data_size, h.record_count
            )));
        }
        hashes.reserve(h.record_count as usize);
        for _ in 0..h.record_count {
            let hash = reader
                .read_u64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(format!("Error reading hash: {}", e)))?;
            hashes.insert(hash);
        }
        let mut extra = [0u8; 1];
        if reader
            .read(&mut extra)
            .map_err(|e| PyValueError::new_err(format!("Error checking trailing data: {}", e)))?
            > 0
        {
            return Err(PyValueError::new_err(
                "Unexpected trailing data in index file",
            ));
        }
        return Ok(hashes);
    }
    loop {
        match reader.read_u64::<LittleEndian>() {
            Ok(hash) => {
                hashes.insert(hash);
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Error reading index file: {}",
                    e
                )))
            }
        }
    }
    Ok(hashes)
}

fn load_attr_from_sidecar(index_path: &str) -> PyResult<HashMap<u64, u64>> {
    let attr_path = format!("{}.attr", index_path);
    let file = File::open(&attr_path).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to open attribution sidecar '{}': {}",
            attr_path, e
        ))
    })?;
    let file_size = std::fs::metadata(&attr_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to stat attr file: {}", e)))?
        .len();
    let mut reader = BufReader::new(file);
    let header = read_header_optional(&mut reader, Some(FileType::AttrMasks))
        .map_err(|e| PyValueError::new_err(format!("Error reading attr header: {}", e)))?;
    let mut map = HashMap::new();
    if let Some(h) = header {
        let data_size = file_size.saturating_sub(HEADER_SIZE);
        let max_records = data_size / 16;
        if h.record_count > max_records {
            return Err(PyValueError::new_err(format!(
                "Attr header record count {} exceeds max possible {}",
                h.record_count, max_records
            )));
        }
        if data_size != h.record_count * 16 {
            return Err(PyValueError::new_err(format!(
                "Attr size {} does not match header record count {}",
                data_size, h.record_count
            )));
        }
        for _ in 0..h.record_count {
            let hash = reader
                .read_u64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(format!("Error reading attr hash: {}", e)))?;
            let mask = reader
                .read_u64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(format!("Error reading attr mask: {}", e)))?;
            map.insert(hash, mask);
        }
        let mut extra = [0u8; 1];
        if reader
            .read(&mut extra)
            .map_err(|e| PyValueError::new_err(format!("Error checking trailing data: {}", e)))?
            > 0
        {
            return Err(PyValueError::new_err(
                "Unexpected trailing data in attr sidecar",
            ));
        }
        return Ok(map);
    }
    loop {
        let h = match reader.read_u64::<LittleEndian>() {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Error reading attr hash: {}",
                    e
                )))
            }
        };
        let mask = reader
            .read_u64::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(format!("Error reading attr mask: {}", e)))?;
        map.insert(h, mask);
    }
    Ok(map)
}

pub fn get_hash_index(path: &str) -> PyResult<Arc<HashSet<u64>>> {
    let key = cache_key(path);
    // Fast path: a cached entry. Drop the lock before slow disk I/O so concurrent
    // loads of other indexes are not serialized behind it.
    {
        let cache = HASH_INDEX
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock hash-index cache"))?;
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.clone());
        }
    }
    let arc = Arc::new(load_index_from_native(path)?);
    let mut cache = HASH_INDEX
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock hash-index cache"))?;
    // Double-checked: a racing thread may have inserted first; keep that one.
    Ok(cache.entry(key).or_insert(arc).clone())
}

pub fn get_attr_index(path: &str) -> PyResult<Arc<HashMap<u64, u64>>> {
    let key = cache_key(path);
    {
        let cache = ATTR_INDEX
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock attr-index cache"))?;
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.clone());
        }
    }
    let arc = Arc::new(load_attr_from_sidecar(path)?);
    let mut cache = ATTR_INDEX
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock attr-index cache"))?;
    Ok(cache.entry(key).or_insert(arc).clone())
}

fn load_index_ngram(path: &str) -> PyResult<usize> {
    let meta_path = metadata_path(path);
    let data = std::fs::read(&meta_path).map_err(|e| {
        PyValueError::new_err(format!(
            "Missing index metadata '{}': {}. Rebuild the native index.",
            meta_path, e
        ))
    })?;
    let meta: IndexMetadata = serde_json::from_slice(&data).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to parse index metadata '{}': {}",
            meta_path, e
        ))
    })?;
    Ok(meta.ngram)
}

pub fn get_index_ngram(path: &str) -> PyResult<usize> {
    let key = cache_key(path);
    {
        let cache = INDEX_NGRAM
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock index-ngram cache"))?;
        if let Some(entry) = cache.get(&key) {
            return Ok(*entry);
        }
    }
    let ngram = load_index_ngram(path)?;
    let mut cache = INDEX_NGRAM
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock index-ngram cache"))?;
    Ok(*cache.entry(key).or_insert(ngram))
}
