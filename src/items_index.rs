use crate::formats::{read_header_optional, FileType};
use crate::index::cache_key;
use byteorder::{LittleEndian, ReadBytesExt};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::{Arc, Mutex};

/// In-memory item-postings index: hash -> sorted list of protected item ids.
///
/// Built by the Python index builder as `<index>.items` and consumed by the
/// scanner's item-aware gate. Each record is `(u64 hash, u32 n_items, n_items ×
/// u32 item_id)`. `n_items_total` is the number of distinct protected items, used
/// only for sanity/metadata.
#[derive(Debug)]
pub struct ItemPostings {
    pub hash_to_items: HashMap<u64, Vec<u32>>,
}

#[allow(clippy::type_complexity)]
static ITEMS_INDEX: Lazy<Mutex<std::collections::HashMap<String, Arc<ItemPostings>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

fn items_path(index_path: &str) -> String {
    format!("{}.items", index_path)
}

pub fn items_sidecar_exists(index_path: &str) -> bool {
    std::path::Path::new(&items_path(index_path)).exists()
}

fn load_items_from_sidecar(index_path: &str) -> PyResult<ItemPostings> {
    let path = items_path(index_path);
    let file = File::open(&path).map_err(|e| {
        PyValueError::new_err(format!("Failed to open item sidecar '{}': {}", path, e))
    })?;
    let mut reader = BufReader::new(file);
    let header = read_header_optional(&mut reader, Some(FileType::ItemPostings))
        .map_err(|e| PyValueError::new_err(format!("Error reading item sidecar header: {}", e)))?
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Item sidecar '{}' is missing a format header; rebuild the index.",
                path
            ))
        })?;

    // Cap the preallocation: `record_count` comes from the file header, so a
    // corrupt/hostile value must not trigger a huge up-front allocation. The map
    // still grows as records are read.
    let mut hash_to_items: HashMap<u64, Vec<u32>> =
        HashMap::with_capacity((header.record_count as usize).min(1_000_000));
    for _ in 0..header.record_count {
        let hash = reader
            .read_u64::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(format!("Error reading item hash: {}", e)))?;
        let n_items = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(format!("Error reading item count: {}", e)))?;
        // Cap the per-record preallocation for the same reason; grows if needed.
        let mut ids = Vec::with_capacity((n_items as usize).min(1024));
        for _ in 0..n_items {
            ids.push(
                reader
                    .read_u32::<LittleEndian>()
                    .map_err(|e| PyValueError::new_err(format!("Error reading item id: {}", e)))?,
            );
        }
        hash_to_items.insert(hash, ids);
    }
    let mut extra = [0u8; 1];
    if reader
        .read(&mut extra)
        .map_err(|e| PyValueError::new_err(format!("Error checking trailing data: {}", e)))?
        > 0
    {
        return Err(PyValueError::new_err(
            "Unexpected trailing data in item sidecar",
        ));
    }
    Ok(ItemPostings { hash_to_items })
}

pub fn get_items_index(index_path: &str) -> PyResult<Arc<ItemPostings>> {
    let key = cache_key(index_path);
    // Drop the lock before slow disk I/O so concurrent loads are not serialized.
    {
        let cache = ITEMS_INDEX
            .lock()
            .map_err(|_| PyValueError::new_err("Failed to lock items-index cache"))?;
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.clone());
        }
    }
    let arc = Arc::new(load_items_from_sidecar(index_path)?);
    let mut cache = ITEMS_INDEX
        .lock()
        .map_err(|_| PyValueError::new_err("Failed to lock items-index cache"))?;
    // Double-checked: keep the first inserted entry if a thread raced us.
    Ok(cache.entry(key).or_insert(arc).clone())
}
