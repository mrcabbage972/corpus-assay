use crate::formats::{read_header_optional, FileType};
use crate::index::cache_key;
use byteorder::{LittleEndian, ReadBytesExt};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::{Arc, Mutex};

/// One posting of an n-gram hash: the protected item it occurs in, and the
/// fields of that item it occurs in (bit k = the item's k-th spec field; 0 when
/// the sidecar predates field masks, format v1).
#[derive(Debug, Clone, Copy)]
pub struct Posting {
    pub item: u32,
    pub fields: u32,
}

/// In-memory item-postings index: hash -> postings sorted by item id.
///
/// Built by the Python index builder as `<index>.items` and consumed by the
/// scanner's item-aware gate. Format v1 (file type `ItemPostings`) records are
/// `(u64 hash, u32 n_items, n_items × u32 item_id)`; format v2 (file type
/// `ItemPostingsV2`) records are `(u64 hash, u32 n_items, n_items × (u32 item_id,
/// u32 field_mask))`.
#[derive(Debug)]
pub struct ItemPostings {
    pub hash_to_items: HashMap<u64, Vec<Posting>>,
    /// Whether the postings carry field masks (format v2).
    pub has_fields: bool,
    /// Field names for the field-mask bits, from `.items.meta.json` (empty when
    /// the sidecar has no field masks, or the meta lacks the names).
    field_names: FieldNames,
}

/// The field names of each item's benchmark, as listed in `.items.meta.json`.
#[derive(Debug, Default)]
struct FieldNames {
    per_benchmark: Vec<Vec<String>>,
    /// Index into `per_benchmark` per item id (`u32::MAX`: unknown).
    item_benchmark: Vec<u32>,
}

impl ItemPostings {
    /// Name of field `field` of `item`'s benchmark, when the meta lists it.
    pub fn field_name(&self, item: u32, field: u32) -> Option<&str> {
        let bench = *self.field_names.item_benchmark.get(item as usize)?;
        self.field_names
            .per_benchmark
            .get(bench as usize)?
            .get(field as usize)
            .map(String::as_str)
    }
}

#[derive(Deserialize)]
struct ItemsMetaFile {
    #[serde(default)]
    fields: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    items: Vec<ItemsMetaEntry>,
}

#[derive(Deserialize)]
struct ItemsMetaEntry {
    #[serde(default)]
    benchmark: String,
}

/// Field names per item from `<index>.items.meta.json`. Best effort: a missing
/// or unreadable meta only means records carry field ids without names.
fn load_field_names(index_path: &str) -> FieldNames {
    let path = format!("{}.items.meta.json", index_path);
    let Ok(file) = File::open(&path) else {
        return FieldNames::default();
    };
    let Ok(meta) = serde_json::from_reader::<_, ItemsMetaFile>(BufReader::new(file)) else {
        return FieldNames::default();
    };
    if meta.fields.is_empty() {
        return FieldNames::default();
    }
    let mut names: Vec<&String> = meta.fields.keys().collect();
    names.sort();
    let index: std::collections::HashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i as u32))
        .collect();
    let item_benchmark = meta
        .items
        .iter()
        .map(|item| {
            index
                .get(item.benchmark.as_str())
                .copied()
                .unwrap_or(u32::MAX)
        })
        .collect();
    let per_benchmark = names
        .iter()
        .map(|name| meta.fields[*name].clone())
        .collect();
    FieldNames {
        per_benchmark,
        item_benchmark,
    }
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

fn read_u32(reader: &mut impl Read, what: &str) -> PyResult<u32> {
    reader
        .read_u32::<LittleEndian>()
        .map_err(|e| PyValueError::new_err(format!("Error reading {}: {}", what, e)))
}

fn load_items_from_sidecar(index_path: &str) -> PyResult<ItemPostings> {
    let path = items_path(index_path);
    let file = File::open(&path).map_err(|e| {
        PyValueError::new_err(format!("Failed to open item sidecar '{}': {}", path, e))
    })?;
    let mut reader = BufReader::new(file);
    let header = read_header_optional(&mut reader, None)
        .map_err(|e| PyValueError::new_err(format!("Error reading item sidecar header: {}", e)))?
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Item sidecar '{}' is missing a format header; rebuild the index.",
                path
            ))
        })?;
    let has_fields = if header.file_type == FileType::ItemPostingsV2 as u32 {
        true
    } else if header.file_type == FileType::ItemPostings as u32 {
        false
    } else {
        return Err(PyValueError::new_err(format!(
            "Item sidecar '{}' has file type {}, expected {} or {}",
            path,
            header.file_type,
            FileType::ItemPostings as u32,
            FileType::ItemPostingsV2 as u32
        )));
    };

    // Cap the preallocation: `record_count` comes from the file header, so a
    // corrupt/hostile value must not trigger a huge up-front allocation. The map
    // still grows as records are read.
    let mut hash_to_items: HashMap<u64, Vec<Posting>> =
        HashMap::with_capacity((header.record_count as usize).min(1_000_000));
    for _ in 0..header.record_count {
        let hash = reader
            .read_u64::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(format!("Error reading item hash: {}", e)))?;
        let n_items = read_u32(&mut reader, "item count")?;
        // Cap the per-record preallocation for the same reason; grows if needed.
        let mut postings = Vec::with_capacity((n_items as usize).min(1024));
        for _ in 0..n_items {
            let item = read_u32(&mut reader, "item id")?;
            let fields = if has_fields {
                read_u32(&mut reader, "field mask")?
            } else {
                0
            };
            postings.push(Posting { item, fields });
        }
        hash_to_items.insert(hash, postings);
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
    let field_names = if has_fields {
        load_field_names(index_path)
    } else {
        FieldNames::default()
    };
    Ok(ItemPostings {
        hash_to_items,
        has_fields,
        field_names,
    })
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
