import struct
from collections import defaultdict
from typing import Dict, Iterator

from pydantic import BaseModel, Field

from corpus_assay import formats


class AttributionLegend(BaseModel):
    """Typed legend metadata for attribution sidecars."""

    name_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_name: list[str] = Field(default_factory=list)
    per_source_total: dict[int, int] | None = None


class AttributionIndex:
    """Centralize attribution sidecar, legend, and mask utilities."""

    MAX_SOURCES = 64
    ATTR_FILE_TYPE = formats.FileType.ATTR_MASKS
    HITS_FILE_TYPE = formats.FileType.HITS_PAIRS

    def __init__(self) -> None:
        """Initialize empty attribution state."""
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: list[str] = []
        self.hash2mask: dict[int, int] = {}
        # Item-level postings: hash -> set of protected-item indices, plus
        # per-item metadata (benchmark, item_id, distinct hash count). Populated
        # only when the builder tracks items; enables item-level scan gating.
        self.hash2items: dict[int, set[int]] = {}
        self.items_meta: list[dict] = []

    def register_item(self, benchmark: str, item_id: str | None = None) -> int:
        """Register a protected evaluation item, returning its integer index."""
        idx = len(self.items_meta)
        self.items_meta.append(
            {
                "benchmark": benchmark,
                "item_id": item_id if item_id is not None else f"{benchmark}#{idx}",
                "n_protected_hashes": 0,
            }
        )
        return idx

    def add_hash_for_item(self, hash_value: int, item_idx: int) -> None:
        """Record that ``hash_value`` occurs in protected item ``item_idx``."""
        bucket = self.hash2items.setdefault(hash_value, set())
        if item_idx not in bucket:
            bucket.add(item_idx)
            self.items_meta[item_idx]["n_protected_hashes"] += 1

    def get_source_id(self, name: str) -> int:
        """Return the numeric source id for a name, registering if needed."""
        if name not in self.name_to_id:
            if len(self.id_to_name) >= self.MAX_SOURCES:
                raise ValueError(
                    f"Attribution supports at most {self.MAX_SOURCES} sources "
                    f"(attempted to add '{name}' as source #{len(self.id_to_name) + 1})."
                )
            self.name_to_id[name] = len(self.id_to_name)
            self.id_to_name.append(name)
        return self.name_to_id[name]

    def add_hash_for_source(self, hash_value: int, source_name: str) -> None:
        """Add a hash entry for a named source."""
        sid = self.get_source_id(source_name)
        self.add_hash_mask(hash_value, 1 << sid)

    def add_hash_mask(self, hash_value: int, mask: int) -> None:
        """Merge a mask into the attribution map for a hash."""
        self.hash2mask[hash_value] = self.hash2mask.get(hash_value, 0) | mask

    def write_sidecar(self, base_path: str, *, ngram: int | None = None) -> str:
        """Write the attribution sidecar for the current hash map."""
        attr_path = base_path + ".attr"
        with open(attr_path, "wb") as bf:
            formats.write_header(
                bf,
                file_type=formats.FileType.ATTR_MASKS,
                ngram_n=ngram or 0,
                record_count=len(self.hash2mask),
            )
            for h in sorted(self.hash2mask):
                mask = self.hash2mask[h]
                bf.write(struct.pack("<QQ", h, mask))
        return attr_path

    def has_items(self) -> bool:
        """Whether any item postings were recorded."""
        return bool(self.hash2items)

    def write_items_sidecar(self, base_path: str, *, ngram: int | None = None) -> str:
        """Write the item-postings sidecar (`<base>.items`) + metadata JSON.

        Record layout (after the standard format header): for each hash, sorted
        ascending, ``<u64 hash, u32 n_items, n_items × u32 item_idx>``. The paired
        ``<base>.items.meta.json`` carries the per-item ``(benchmark, item_id,
        n_protected_hashes)`` table so reports can name the specific leaked items.
        """
        items_path = base_path + ".items"
        with open(items_path, "wb") as bf:
            formats.write_header(
                bf,
                file_type=formats.FileType.ITEM_POSTINGS,
                ngram_n=ngram or 0,
                record_count=len(self.hash2items),
            )
            for h in sorted(self.hash2items):
                ids = sorted(self.hash2items[h])
                bf.write(struct.pack("<QI", h, len(ids)))
                if ids:
                    bf.write(struct.pack(f"<{len(ids)}I", *ids))

        meta_path = base_path + ".items.meta.json"
        import json

        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {"n_items": len(self.items_meta), "items": self.items_meta},
                mf,
                indent=2,
            )
        return items_path

    @staticmethod
    def read_items_sidecar(path: str) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(hash, [item_idx, ...])`` records from an item-postings sidecar."""
        head = struct.calcsize("<QI")
        with open(path, "rb") as f:
            header = formats.read_header_optional(
                f, expected_file_type=formats.FileType.ITEM_POSTINGS
            )
            if header is None:
                raise ValueError(f"{path} is missing an item-postings header.")
            for _ in range(header.record_count):
                chunk = f.read(head)
                if len(chunk) != head:
                    raise ValueError(f"Truncated item record in {path}.")
                h, n_items = struct.unpack("<QI", chunk)
                # Guard against a corrupt/hostile count driving a huge read.
                if n_items > 1_000_000:
                    raise ValueError(
                        f"Unreasonably large item count ({n_items}) in {path}."
                    )
                ids_chunk = f.read(4 * n_items)
                if len(ids_chunk) != 4 * n_items:
                    raise ValueError(f"Truncated item-id list in {path}.")
                ids = list(struct.unpack(f"<{n_items}I", ids_chunk)) if n_items else []
                yield h, ids

    def per_source_totals(self) -> Dict[int, int]:
        """Return unique hash counts per source id from the current map."""
        totals: Dict[int, int] = defaultdict(int)
        for _hash, mask in self.hash2mask.items():
            for sid in self.iter_mask_source_ids(mask):
                totals[sid] += 1
        return dict(totals)

    def write_legend(
        self, base_path: str, per_source_total: Dict[int, int] | None = None
    ) -> str:
        """Write the source legend JSON (typed with Pydantic)."""
        meta_path = base_path + ".sources.json"
        legend = AttributionLegend(
            name_to_id=self.name_to_id,
            id_to_name=self.id_to_name,
            per_source_total=per_source_total,
        )
        with open(meta_path, "w") as mf:
            mf.write(legend.model_dump_json(indent=2))
        return meta_path

    @staticmethod
    def load_legend(path: str) -> AttributionLegend:
        """Load the source legend JSON into a validated Pydantic model."""
        with open(path, "r") as f:
            payload = f.read()
        return AttributionLegend.model_validate_json(payload)

    @staticmethod
    def iter_mask_source_ids(mask: int) -> Iterator[int]:
        """Yield source ids set in the provided bitmask."""
        mm = mask
        while mm:
            sid = (mm & -mm).bit_length() - 1
            yield sid
            mm &= mm - 1

    @staticmethod
    def read_hash_mask_pairs(
        path: str, *, expected_file_type: formats.FileType | None = None
    ) -> Iterator[tuple[int, int]]:
        """Read <u64 hash, u64 mask> records from a sidecar file."""
        rec_size = struct.calcsize("<QQ")
        with open(path, "rb") as f:
            header = formats.read_header_optional(
                f, expected_file_type=expected_file_type
            )
            if header is not None:
                for _ in range(header.record_count):
                    chunk = f.read(rec_size)
                    if len(chunk) != rec_size:
                        raise ValueError(
                            f"Truncated record in {path}: expected {rec_size} bytes, got {len(chunk)}"
                        )
                    yield struct.unpack("<QQ", chunk)
                extra = f.read(1)
                if extra:
                    raise ValueError(f"Extra trailing data in {path} after records.")
                return

            while chunk := f.read(rec_size):
                if len(chunk) != rec_size:
                    raise ValueError(
                        f"Truncated record in {path}: expected {rec_size} bytes, got {len(chunk)}"
                    )
                yield struct.unpack("<QQ", chunk)


__all__ = ["AttributionIndex", "AttributionLegend"]
