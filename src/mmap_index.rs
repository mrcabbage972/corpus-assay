use crate::formats::{read_header_optional, FileType, HEADER_SIZE};
use memmap2::{Mmap, MmapOptions};
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex, Weak};
use std::time::UNIX_EPOCH;

#[derive(Debug)]
pub struct MmapAttrIndex {
    mmap: Mmap,
    data_offset: usize,
    len: usize,
}

#[derive(Hash, Eq, PartialEq)]
struct MmapCacheKey {
    path: String,
    len: u64,
    modified_ns: u128,
}

static MMAP_CACHE: OnceCell<Mutex<HashMap<MmapCacheKey, Weak<MmapAttrIndex>>>> = OnceCell::new();

fn cache() -> &'static Mutex<HashMap<MmapCacheKey, Weak<MmapAttrIndex>>> {
    MMAP_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl MmapAttrIndex {
    pub fn open_attr(index_path: &str) -> PyResult<Arc<Self>> {
        let attr_path = format!("{}.attr", index_path);
        let file = File::open(&attr_path).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to open attribution sidecar '{}': {}",
                attr_path, e
            ))
        })?;
        let metadata = file
            .metadata()
            .map_err(|e| PyValueError::new_err(format!("Failed to stat attr file: {}", e)))?;
        let file_size = metadata.len();
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|ts| ts.duration_since(UNIX_EPOCH).ok())
            .map(|dur| dur.as_nanos())
            .unwrap_or(0);

        let key = MmapCacheKey {
            path: attr_path.clone(),
            len: file_size,
            modified_ns,
        };

        if let Some(existing) = cache().lock().expect("mmap cache lock").get(&key) {
            if let Some(shared) = existing.upgrade() {
                return Ok(shared);
            }
        }

        let mut reader = BufReader::new(&file);
        let header = read_header_optional(&mut reader, Some(FileType::AttrMasks))
            .map_err(|e| PyValueError::new_err(format!("Error reading attr header: {}", e)))?;
        let header = header.ok_or_else(|| {
            PyValueError::new_err(
                "Attr sidecar missing header; mmap backend requires headered .attr files",
            )
        })?;

        let data_size = file_size
            .checked_sub(HEADER_SIZE)
            .ok_or_else(|| PyValueError::new_err("Attr file shorter than header"))?;
        let max_records = data_size / 16;
        if header.record_count > max_records {
            return Err(PyValueError::new_err(format!(
                "Attr header record count {} exceeds max possible {}",
                header.record_count, max_records
            )));
        }
        if data_size != header.record_count * 16 {
            return Err(PyValueError::new_err(format!(
                "Attr size {} does not match header record count {}",
                data_size, header.record_count
            )));
        }
        let len = usize::try_from(header.record_count)
            .map_err(|_| PyValueError::new_err("Attr record count exceeds addressable size"))?;
        let data_offset = usize::try_from(HEADER_SIZE)
            .map_err(|_| PyValueError::new_err("Header offset exceeds addressable size"))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| PyValueError::new_err(format!("Failed to mmap attr file: {}", e)))?
        };
        if data_offset
            .checked_add(len.saturating_mul(16))
            .is_none_or(|end| end > mmap.len())
        {
            return Err(PyValueError::new_err(
                "Attr mmap size does not cover expected records",
            ));
        }

        let index = Arc::new(Self {
            mmap,
            data_offset,
            len,
        });

        cache()
            .lock()
            .expect("mmap cache lock")
            .insert(key, Arc::downgrade(&index));

        Ok(index)
    }

    #[inline]
    fn read_u64_at(&self, offset: usize) -> u64 {
        let ptr = unsafe { self.mmap.as_ptr().add(offset) as *const u64 };
        let raw = unsafe { std::ptr::read_unaligned(ptr) };
        u64::from_le(raw)
    }

    #[inline]
    fn hash_at(&self, i: usize) -> u64 {
        let offset = self.data_offset + i * 16;
        self.read_u64_at(offset)
    }

    #[inline]
    fn mask_at(&self, i: usize) -> u64 {
        let offset = self.data_offset + i * 16 + 8;
        self.read_u64_at(offset)
    }

    pub fn mask_for_hash(&self, needle: u64) -> Option<u64> {
        let mut lo = 0usize;
        let mut hi = self.len;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let h = self.hash_at(mid);
            if h < needle {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < self.len && self.hash_at(lo) == needle {
            Some(self.mask_at(lo))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::write_header;
    use byteorder::{LittleEndian, WriteBytesExt};
    use tempfile::tempdir;

    #[test]
    fn mmap_lookup_returns_masks() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("index.native");
        let attr_path = index_path.with_extension("native.attr");
        let mut f = File::create(&attr_path).unwrap();
        let pairs = vec![(10u64, 1u64), (20u64, 5u64), (30u64, 9u64)];
        write_header(&mut f, FileType::AttrMasks, 13, pairs.len() as u64).unwrap();
        for (h, m) in &pairs {
            f.write_u64::<LittleEndian>(*h).unwrap();
            f.write_u64::<LittleEndian>(*m).unwrap();
        }
        drop(f);

        let index = MmapAttrIndex::open_attr(index_path.to_str().unwrap()).unwrap();
        assert_eq!(index.mask_for_hash(10), Some(1));
        assert_eq!(index.mask_for_hash(20), Some(5));
        assert_eq!(index.mask_for_hash(30), Some(9));
        assert_eq!(index.mask_for_hash(25), None);
    }

    #[test]
    fn mmap_rejects_mismatched_size() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("index.native");
        let attr_path = index_path.with_extension("native.attr");
        let mut f = File::create(&attr_path).unwrap();
        write_header(&mut f, FileType::AttrMasks, 13, 2).unwrap();
        f.write_u64::<LittleEndian>(10).unwrap();
        f.write_u64::<LittleEndian>(1).unwrap();
        drop(f);

        let err = MmapAttrIndex::open_attr(index_path.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("Attr size"));
    }
}
