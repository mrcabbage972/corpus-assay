mod formats;
mod index;
mod items_index;
mod mmap_index;
mod scan;
mod spaced;
mod stopgrams;
mod text;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
#[pyo3(name = "_native")]
fn corpus_assay(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Export scan functions
    m.add_function(wrap_pyfunction!(scan::scan_stream_rust, m)?)?;

    // Export text normalization utilities
    m.add_function(wrap_pyfunction!(text::set_ngram_config_path, m)?)?;
    m.add_function(wrap_pyfunction!(text::normalize_text, m)?)?;
    m.add_function(wrap_pyfunction!(text::ngram_allowed, m)?)?;
    m.add_function(wrap_pyfunction!(text::hash_ngram, m)?)?;
    m.add_function(wrap_pyfunction!(text::ngram_filter, m)?)?;
    m.add_function(wrap_pyfunction!(text::ngram_config_version, m)?)?;
    m.add_function(wrap_pyfunction!(text::ngram_config_hash, m)?)?;
    m.add_function(wrap_pyfunction!(text::normalization_config_sha, m)?)?;
    m.add_function(wrap_pyfunction!(text::hash_function_metadata, m)?)?;

    // Spaced-seed channel parity entry points (production path uses scan_*_rust).
    m.add_function(wrap_pyfunction!(spaced::spaced_seed_hashes_rust, m)?)?;
    m.add_function(wrap_pyfunction!(spaced::spaced_verified_item_rust, m)?)?;

    Ok(())
}
