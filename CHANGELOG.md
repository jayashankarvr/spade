# Changelog

All notable changes to SPADE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-06-05

### Fixed
- **Affine verifier crash**: `_solve_huber` could raise `UnboundLocalError` on a singular system (e.g. constant patch with regularization disabled) because `rss` was assigned only after the matrix inversion; it is now initialized from the L2 solution and fails gracefully.
- **Non-deterministic sharding**: `ShardedIndex` used the built-in `hash()`, which is salted per process (`PYTHONHASHSEED`), so the same `image_id` could map to different shards across runs. Now uses a stable md5-based hash.
- **CLI duplicate indexing**: `spade index` globbed both lowercase and uppercase patterns, indexing every image twice on case-insensitive filesystems. Globbed files are now de-duplicated.
- **API decompression-bomb gap**: the megapixel check only inspected the header before decoding; PIL's `Image.MAX_IMAGE_PIXELS` is now set so the cap is enforced at decode time too.
- **API error mapping**: the `/match` endpoint reported genuine server faults as HTTP 400; it now returns 400 only for `ValueError` and 500 otherwise, matching `/index`.

### Changed
- Validation exceptions (`ConfigurationError`, `InvalidImageError`, `ImageSizeError`, `ImageFormatError`) now subclass `ValueError` in addition to `SPADEError`, restoring backward-compatible `except ValueError` handling.
- Renamed the internal `IndexError` exception to `IndexStoreError` to stop shadowing the Python built-in.

### Known limitations
- In `scoring_mode="bayes"`, `max_targets` LRU eviction is not enforced (the Bayes engine keeps its own target stores). To be addressed in the v1 redesign.

## [Unreleased]

### Added
- **Image Pyramids**: Multi-scale analysis with configurable pyramid levels (1x, 0.5x, 0.25x, 0.125x) for scale-invariant matching
- **256-dimensional descriptors**: Upgraded from 128-dim for better discrimination
- **LSH Pre-filtering**: Optional Locality Sensitive Hashing for sublinear search time on large indexes (requires `datasketch`)
- **REST API**: FastAPI-based web service with endpoints for indexing, matching, and target management
- **Spatial Coherence Verification**: Clusters matches by translation consistency to dramatically reduce false positives
- `spade serve` CLI command to start the API server
- Hybrid index combining LSH pre-filtering with FAISS search
- `CoherentRegion` dataclass with bounding boxes and confidence scores

### Changed
- Default descriptor dimension increased from 128 to 256
- Improved coordinate mapping for pyramid-indexed targets
- Enhanced match results include coherent regions

### Technical Details
- `ImagePyramid` class for multi-resolution image processing
- `PyramidPatchCollection` with original coordinate mapping
- `LSHPreFilter` using MinHash for fast candidate selection
- `HybridIndex` combining LSH + FAISS for optimal speed/accuracy tradeoff
- FastAPI endpoints with Pydantic models for type safety

## [0.1.0] - Unreleased

### Added
- Core patch extraction with entropy-based filtering
- Photometric-invariant descriptors (difference vectors, chromaticity, LBP, rank-order, gradient histogram)
- FAISS-based approximate nearest neighbor search
- Affine color transform verification with Huber loss (robust to JPEG artifacts)
- Chi-square probability model for match scoring
- Spatial aggregation into forensic heatmaps
- CLI commands: `match`, `index`, `search`
- Python API via `ForensicsEngine`
- Auto-train PCA for descriptor dimensionality reduction
- Support for patch sizes 3x3 through 6x6
- JSON metadata storage (secure, replaces pickle)

### Technical Details
- 128-dimensional descriptors (configurable)
- Vectorized patch extraction using numpy stride tricks
- Optimized IRLS solver for Huber loss
- Memory-efficient target storage (patches only, not full images)
