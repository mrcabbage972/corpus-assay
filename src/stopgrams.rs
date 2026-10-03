use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

const STOPGRAM_MAGIC: [u8; 8] = *b"CASTOP1\0";
const STOPGRAM_FORMAT_VERSION: u32 = 1;
const STOPGRAM_HEADER_SIZE: usize = 24;
const STOPGRAM_RECORD_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct StopGramHeader {
    pub ngram: u32,
    pub record_count: u64,
}

fn read_header<R: Read>(reader: &mut R) -> Result<StopGramHeader, String> {
    let mut buf = [0u8; STOPGRAM_HEADER_SIZE];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read stop-grams header: {}", e))?;
    if buf[..8] != STOPGRAM_MAGIC {
        return Err("Invalid stop-grams magic header.".to_string());
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != STOPGRAM_FORMAT_VERSION {
        return Err(format!("Unsupported stop-grams format version {}", version));
    }
    let ngram = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let record_count = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    Ok(StopGramHeader {
        ngram,
        record_count,
    })
}

#[derive(Debug)]
pub struct StopGramsSet {
    mmap: Mmap,
    data_offset: usize,
    len: usize,
    ngram: u32,
}

impl StopGramsSet {
    pub fn open(path: &str) -> PyResult<Arc<Self>> {
        let file = File::open(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to open stop-grams '{}': {}", path, e))
        })?;
        let metadata = file
            .metadata()
            .map_err(|e| PyValueError::new_err(format!("Failed to stat stop-grams: {}", e)))?;
        let file_size = metadata.len() as usize;
        let mut reader = std::io::BufReader::new(&file);
        let header = read_header(&mut reader)
            .map_err(|e| PyValueError::new_err(format!("Stop-grams header error: {}", e)))?;
        let data_size = file_size
            .checked_sub(STOPGRAM_HEADER_SIZE)
            .ok_or_else(|| PyValueError::new_err("Stop-grams file shorter than header"))?;
        let max_records = data_size / STOPGRAM_RECORD_SIZE;
        if header.record_count > max_records as u64 {
            return Err(PyValueError::new_err(format!(
                "Stop-grams header record count {} exceeds max possible {}",
                header.record_count, max_records
            )));
        }
        if data_size != (header.record_count as usize) * STOPGRAM_RECORD_SIZE {
            return Err(PyValueError::new_err(format!(
                "Stop-grams size {} does not match header record count {}",
                data_size, header.record_count
            )));
        }

        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| PyValueError::new_err(format!("Failed to mmap stop-grams: {}", e)))?
        };
        Ok(Arc::new(Self {
            mmap,
            data_offset: STOPGRAM_HEADER_SIZE,
            len: header.record_count as usize,
            ngram: header.ngram,
        }))
    }

    pub fn ngram(&self) -> u32 {
        self.ngram
    }

    #[inline]
    fn read_u64_at(&self, offset: usize) -> u64 {
        let ptr = unsafe { self.mmap.as_ptr().add(offset) as *const u64 };
        let raw = unsafe { std::ptr::read_unaligned(ptr) };
        u64::from_le(raw)
    }

    #[inline]
    fn hash_at(&self, i: usize) -> u64 {
        let offset = self.data_offset + i * STOPGRAM_RECORD_SIZE;
        self.read_u64_at(offset)
    }

    pub fn contains(&self, needle: u64) -> bool {
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
        lo < self.len && self.hash_at(lo) == needle
    }
}
