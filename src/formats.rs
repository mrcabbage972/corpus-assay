use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom};

pub const MAGIC: [u8; 8] = *b"CAFMTv1\0";
pub const FORMAT_VERSION: u32 = 1;
pub const ENDIANNESS_MARKER: u32 = 0x01020304;
// 8 (magic) + 5 * 4 (u32 fields) + 8 (u64 record_count)
pub const HEADER_SIZE: u64 = 36;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_matches_layout() {
        let mut buf = Vec::new();
        write_header(&mut buf, FileType::NativeHashes, 0, 0).unwrap();
        assert_eq!(buf.len() as u64, HEADER_SIZE);
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileType {
    NativeHashes = 1,
    AttrMasks = 2,
    HitsPairs = 3,
    /// Item-postings sidecar (`<index>.items`): per hash, the list of protected
    /// evaluation-item ids it belongs to. Enables item-level (rather than
    /// benchmark-level) hit gating in the scanner.
    ItemPostings = 4,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct Header {
    pub file_type: u32,
    pub version: u32,
    pub endianness_marker: u32,
    pub ngram_n: u32,
    pub reserved: u32,
    pub record_count: u64,
}

pub fn write_header<W: WriteBytesExt>(
    writer: &mut W,
    file_type: FileType,
    ngram_n: u32,
    record_count: u64,
) -> std::io::Result<()> {
    writer.write_all(&MAGIC)?;
    writer.write_u32::<LittleEndian>(file_type as u32)?;
    writer.write_u32::<LittleEndian>(FORMAT_VERSION)?;
    writer.write_u32::<LittleEndian>(ENDIANNESS_MARKER)?;
    writer.write_u32::<LittleEndian>(ngram_n)?;
    writer.write_u32::<LittleEndian>(0)?;
    writer.write_u64::<LittleEndian>(record_count)?;
    Ok(())
}

fn parse_header<R: ReadBytesExt>(
    reader: &mut R,
    expected_file_type: Option<FileType>,
) -> std::io::Result<Header> {
    let file_type = reader.read_u32::<LittleEndian>()?;
    let version = reader.read_u32::<LittleEndian>()?;
    let endianness_marker = reader.read_u32::<LittleEndian>()?;
    let ngram_n = reader.read_u32::<LittleEndian>()?;
    let reserved = reader.read_u32::<LittleEndian>()?;
    let record_count = reader.read_u64::<LittleEndian>()?;

    if version != FORMAT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported format version {}", version),
        ));
    }
    if endianness_marker != ENDIANNESS_MARKER {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Unsupported endianness marker",
        ));
    }
    if let Some(expected) = expected_file_type {
        if file_type != expected as u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unexpected file type {} (expected {})",
                    file_type, expected as u32
                ),
            ));
        }
    }
    Ok(Header {
        file_type,
        version,
        endianness_marker,
        ngram_n,
        reserved,
        record_count,
    })
}

pub fn read_header_optional<R: Read + Seek>(
    reader: &mut R,
    expected_file_type: Option<FileType>,
) -> std::io::Result<Option<Header>> {
    let mut magic_buf = [0u8; 8];
    reader.read_exact(&mut magic_buf)?;
    if magic_buf != MAGIC {
        reader.seek(SeekFrom::Start(0))?;
        return Ok(None);
    }
    let header = parse_header(reader, expected_file_type)?;
    Ok(Some(header))
}
