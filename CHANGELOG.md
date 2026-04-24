# Changelog

All notable changes to `muvera-fde` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2025-05-01

### Added
- Initial release.
- `MUVERAEncoder` high-level class with pre-cached per-repetition parameters.
- `generate_query_fde` and `generate_document_fde` low-level functional API.
- `FDEConfig` Pydantic v2 immutable configuration model.
- `ProjectionType` enum (`DEFAULT_IDENTITY`, `AMS_SKETCH` / Count Sketch).
- `fill_empty_partitions` support (document side) via SimHash Hamming-distance
  nearest-neighbour fill, operating in batches to bound peak memory.
- Optional final Count-Sketch compression (`final_projection_dimension`).
- Full `py.typed` marker for PEP 561 inline type annotations.
- GitHub Actions CI matrix across Python 3.10–3.13 and three OSes.
- Trusted PyPI publishing via OIDC (no stored API tokens).
- Faithful attribution to Google's graph-mining Apache 2.0 upstream.

[Unreleased]: https://github.com/smarthi/muvera-fde/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/smarthi/muvera-fde/releases/tag/v0.1.0
