# Changelog

All notable changes to SPADE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Image Pyramids**: Multi-scale analysis with configurable pyramid levels (1×, 0.5×, 0.25×, 0.125×) for scale-invariant matching
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
- Support for patch sizes 3×3 through 6×6
- JSON metadata storage (secure, replaces pickle)

### Technical Details
- 128-dimensional descriptors (configurable)
- Vectorized patch extraction using numpy stride tricks
- Optimized IRLS solver for Huber loss
- Memory-efficient target storage (patches only, not full images)
