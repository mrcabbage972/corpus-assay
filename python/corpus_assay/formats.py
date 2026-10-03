from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import BinaryIO

MAGIC = b"CAFMTv1\x00"
FORMAT_VERSION = 1
ENDIANNESS_MARKER = 0x01020304

_HEADER_STRUCT = struct.Struct("<8sIIIIIQ")
HEADER_SIZE = _HEADER_STRUCT.size


class FileType(IntEnum):
    NATIVE_HASHES = 1
    ATTR_MASKS = 2
    HITS_PAIRS = 3
    # Item-postings sidecar (`<index>.items`): per hash, the protected
    # evaluation-item ids it belongs to. Enables item-level hit gating.
    ITEM_POSTINGS = 4


@dataclass(frozen=True)
class BinaryHeader:
    file_type: int
    version: int
    endianness_marker: int
    ngram_n: int
    reserved: int
    record_count: int


def write_header(
    writer: BinaryIO,
    *,
    file_type: FileType,
    ngram_n: int = 0,
    record_count: int = 0,
) -> None:
    writer.write(
        _HEADER_STRUCT.pack(
            MAGIC,
            int(file_type),
            FORMAT_VERSION,
            ENDIANNESS_MARKER,
            int(ngram_n),
            0,
            int(record_count),
        )
    )


def _parse_header(
    data: bytes, *, expected_file_type: FileType | None = None
) -> BinaryHeader:
    magic, file_type, version, endianness, ngram_n, reserved, record_count = (
        _HEADER_STRUCT.unpack(data)
    )
    if magic != MAGIC:
        raise ValueError("Invalid format magic header.")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format version {version}.")
    if endianness != ENDIANNESS_MARKER:
        raise ValueError("Unsupported endianness marker in header.")
    if expected_file_type is not None and file_type != int(expected_file_type):
        raise ValueError(
            f"Unexpected file_type {file_type} (expected {int(expected_file_type)})."
        )
    return BinaryHeader(
        file_type=file_type,
        version=version,
        endianness_marker=endianness,
        ngram_n=ngram_n,
        reserved=reserved,
        record_count=record_count,
    )


def read_header(
    reader: BinaryIO, *, expected_file_type: FileType | None = None
) -> BinaryHeader:
    data = reader.read(HEADER_SIZE)
    if len(data) != HEADER_SIZE:
        raise ValueError(
            f"Truncated header: expected {HEADER_SIZE} bytes, got {len(data)}."
        )
    return _parse_header(data, expected_file_type=expected_file_type)


def read_header_optional(
    reader: BinaryIO, *, expected_file_type: FileType | None = None
) -> BinaryHeader | None:
    data = reader.read(len(MAGIC))
    if len(data) != len(MAGIC):
        raise ValueError(
            f"Truncated header prefix: expected {len(MAGIC)} bytes, got {len(data)}."
        )
    if data != MAGIC:
        reader.seek(0)
        return None
    rest = reader.read(HEADER_SIZE - len(MAGIC))
    if len(rest) != HEADER_SIZE - len(MAGIC):
        raise ValueError("Truncated header data after magic prefix.")
    return _parse_header(data + rest, expected_file_type=expected_file_type)
