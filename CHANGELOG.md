# Changelog

All notable changes to SPADE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release reworks the matching pipeline. On the synthetic benchmark
(`python -m spade.bench.runner --n 20 --negatives 20`) localization IoU goes from
0.48 to 0.85, detection AUC from 0.74 to 0.96, and time per image from 41 s to
0.6 s. With JPEG q85 it reaches IoU 0.78 / AUC 0.95.

### Fixed
- **Targets indexed before PCA training were unmatchable.** PCA was fitted part-way through indexing, so earlier targets kept random-projection descriptors while queries used PCA. The descriptor no longer has any trained state.
- **`spade index` + `spade search` compared descriptors from different spaces**, because the trained projection wasn't saved with the index. Indexes are now saved as a bundle (`ForensicsEngine.save` / `ForensicsEngine.load`) that includes the config.
- **Deleting a target didn't delete anything.** `remove_target`, LRU eviction and `DELETE /targets/{id}` left the vectors in FAISS, where they kept using nearest-neighbour slots. The index now uses `IndexIDMap2` and really removes them.
- **The default entropy filter dropped most patches**, and values of 3.0 or more (as in the old README) dropped all of them, because a 3x3 patch can't reach 3 bits with 8 bins. Entropy filtering is now off by default, and an unreachable threshold is rejected.
- **Per-patch "probability" decided matches**, but a 12-parameter color fit on 27 numbers passes for about half of unrelated textured 3x3 pairs. Decisions are now made per region (see NFA below). The JSON report renames the field to `fit_p_value`.
- **Recolors that clip at 0 or 255** broke the affine model on exactly the copied pixels. Clipped pixel pairs are now left out of the fits and tests.
- **Coherent regions mixed target images** (one offset across different targets formed one cluster) and could chain slowly drifting offsets into one huge cluster. Regions are now formed per target and per pyramid scale, around offset peaks.
- The chi-square tail used `1 - cdf`, which rounds to 0 long before the true value. It now uses the survival function.
- The REST API ran indexing and matching on the event loop, which blocked health checks; they now run in a worker thread. Auto-generated image ids (`img_{count}`) collided after deletions; they're now random.
- Index files are written with `os.replace` from the same directory (actually atomic). The old version deleted the existing files first, and loading still supported pickle.
- API docs and README sent `image_id`, `target_ids` and `return_heatmap` as form fields, which the API ignored; they're query parameters.

### Added
- **Region-level verification with an a-contrario NFA** (`spade.verification.nfa`). Each cluster gets one 3x3 color transform fitted over all its pixels, keeps the patches that agree with it, and is scored by its Number of False Alarms. `MatchResult.significant_regions`, `CoherentRegion.log10_nfa`, `.significant`, `.color_M`, `.color_b`, `Config.max_nfa`.
- `MatchResult.localization`: footprint of the significant regions in source coordinates.
- `Match.target_scale`: pyramid level a match came from. Pyramid matches cluster in that level's coordinates, so downscaled copies form proper regions.
- `ANNIndex.remove`, batched `ANNIndex.search`.

### Changed
- **Descriptor**: normalized luminance pattern (`LuminanceDescriptor`), vectorized. It replaces the five-part composite plus 256-d random projection or PCA. It beat the composite under JPEG in the comparison behind this change and costs far less. Flat patches are always dropped.
- The per-patch pre-filter fits grey-level gain and offset (2 parameters) in one batched call instead of a Python IRLS loop per candidate.
- Defaults chosen by benchmark: `patch_size=5`, `k_neighbors=8`, `distance_threshold=0.5`, `min_probability=0.01`, `entropy_threshold=None`.
- JSON report schema 2.0: `fit_p_value`, per-region `log10_nfa` and `significant`, `result.detected`; `localization.detection_score` is now `-log10(NFA)` of the best region (area moved to `area_fraction`).
- Heatmaps paint each match's footprint instead of a single pixel.
- Entropy is computed with one `bincount` instead of a per-patch loop.
- CLI: `spade index -o PREFIX` writes a bundle; `spade search` loads it and reloads patches from the indexed files. `match`/`search` print significant regions with their NFA.
- CLI `-e/--entropy` now defaults to the config (so `SPADE_ENTROPY_THRESHOLD` applies); pass `-e 0` to turn the filter off explicitly.

### Removed
- **Bayes-factor pipeline** (`scoring_mode="bayes"`: `engine_bayes`, Bayes-factor scorer, natural-image priors, 4-D Hough, persistence peaks, loopy BP, RG-trajectory term, Zernike and persistence descriptors, `scripts/train_priors.py`, `scripts/benchmark.py`, `scripts/synthetic_corpus.py`). The "calibrated" Bayes factor compared densities from different spaces with no Occam penalty, its BP graph construction was quadratic in Python, and the NFA replaces it.
- **LSH pre-filter** (`HybridIndex`, `datasketch`): MinHash/Jaccard is the wrong similarity for real-valued vectors, and exact FAISS search on 25-dimensional descriptors is faster anyway.
- **`ShardedIndex`**: local mode spun up a thread pool per query patch, and remote mode had no server to talk to.
- **`LearnedDescriptor`** (ImageNet ResNet on upscaled 3x3 patches; no photometric invariance) and the `learned`/`phase3` extras.
- **GPU path** (`spade.acceleration`, `use_gpu`): only the entropy step ran on the GPU.
- Config fields `descriptor_dim`, `descriptor_spatial_pooling`, `auto_train_pca`, `pca_train_samples`, `scale_weights`, `lsh_*`, `use_gpu`, `use_learned_descriptors`, `learned_model`, `distributed_enabled`, `num_shards`, `scoring_mode`, `aggregation_mode`, `priors_dir`. `SpatialAggregator.find_regions`, `SpatialCoherenceVerifier(min_cluster_probability=...)`.
- Index files from earlier versions can't be loaded (they were in a different descriptor space anyway); re-index.

### Added (measurement and evidence)
- **Real-data benchmarks** (`spade.bench.datasets`): splices between real photos (scikit-image's public-domain set), CASIA v2 cross-image splices with their actual donor images (donor vs host decided from pixels, never by SPADE), and a CSV manifest for your own data. `python -m spade.bench.download casia2` fetches CASIA v2 from Kaggle.
- **NFA calibration** (`spade.bench.calibration`, `runner --calibrate`): false regions per image pair against the NFA threshold, next to recall, from a single pass.
- **False discovery rate across searches**: `spade.verification.nfa.fdr_select` (Benjamini-Hochberg over every region hypothesis) and `spade scan INDEX IMAGES... --fdr Q`. Regions carry `n_tests` and `log10_p`; results carry `stats["n_tests"]`.
- Tried and dropped because they didn't hold up on the calibration benchmark: per-patch "rarity" chance rates in the NFA (either lost half the recall on JPEG photos or broke the false-alarm guarantee) and a sensor-noise residual check (JPEG removes chroma noise; luminance residuals are already correlated by the matching itself).

### Added (geometry and photometry)
- **Copy-move within one image**: `ForensicsEngine.match_self(image)`. A region-level ratio test (the best competing offset's support sets the chance rate) keeps repetitive texture from passing as copies: on real-photo copy-moves at JPEG 90, false regions fell from 1.4 to 0.02 per image at NFA < 10^-3, detection AUC rose from 0.86 to 0.94, recall is about 0.5. Synthetic benchmark: IoU 0.83, detection AUC 1.0. Two-image search doesn't use the ratio test (there it only cost recall).
- **JPEG-aware noise**: `spade.jpeg.quantization_sigma` reads a file's quantization tables; pairs are tested against base noise plus both sides' rounding error. `index_target(..., jpeg_sigma=)` (read from `image_path` when omitted), `match(..., jpeg_sigma=)`; CLI and API do it automatically. Photo splices at JPEG 75: IoU 0.72 -> 0.76 with no false regions.
- **Rotation-invariant matching (experimental, off by default)**: `Config(rotation_invariant=True)` - canonical patch orientation (intensity centroid) plus rigid-transform clustering; `Match.rotation`, `CoherentRegion.rotation`. About 2/3 of arbitrarily rotated photo copies found, 5-30 s per image (rigid clustering made ~4x faster: no per-seed de-duplication, 64 seeds per round).
- Benchmark: `--task copymove`, `--rotation`, `--flip`, `--curve` (tone curves) options; CASIA v2 `Tp_S` copy-moves.
- Tried and dropped: isotonic tone-curve fitting (the affine fit already handles strong Curves/gamma edits as well or better).

### Added (collections and review)
- **Mirrored copies, on by default**: the mirrored query is searched too (`Match.mirrored`, `CoherentRegion.mirrored`; `Config(flip_invariant=False)` to disable). Photo splices with half the fragments mirrored: recall 0.45 -> 0.90, AUC 0.80 -> 0.96, false regions unchanged; copy-move on real photos: no extra false alarms. Doubles search time.
- **Donor pyramid on by default**, at scales 1, 0.75 and 0.5 (was off, at 1/0.5/0.25/0.125). CASIA v2, 60 + 60 pairs: IoU 0.27 -> 0.37, recall 0.29 -> 0.39, AUC 0.72 -> 0.77, false regions unchanged, +17% time. `spade serve` no longer forces it off unless `--no-pyramid` is given.
- `Config.query_scales`: also search the questioned image shrunk, for copies that were enlarged before pasting (`Match.source_scale`). Off by default: on 14 resized CASIA v2 splices it found 5 instead of 4 at 1.7x the time.
- **Visual HTML report**: `spade.visual.render_html`, `spade match --html report.html` - verdict, both images with matching region outlines, region table with NFA and color transform, heatmap, cues, input hashes; self-contained, light/dark.
- **Provenance graph**: `spade.provenance.build_provenance` and `spade provenance DIR [--copy-moves] [--json] [--dot]` - which images share regions, under FDR control across the collection. Links are undirected on purpose.
- **Compressed index for large collections, on by default**: `index_type="ivfpq"` - FAISS IVF-PQ, trained once `ivf_train_size` (50,000) patches are in, exact until then; deletes and save/load supported. 200k descriptors: 5x smaller, 22x faster search. Real-photo splices with it forced on: identical IoU, recall and false-alarm rate to the exact index. `index_type="flat"` stays exact.
- Tried and removed: a sparse "seed" index (every 4th target position, regions filled in densely). Sparsifying the target changes the statistics the NFA relies on, and none of three chance models held up on real photos (2-13 false regions per pair against 0 for the dense index). Compressing the dense index keeps the statistics intact instead.
- Targets now store their pixels (uint8, per pyramid level) instead of a float32 patch per position: about 100x less memory per indexed image.
- Copy-move's ratio test vectorized (about 3x faster `match_self`).
- Saved bundles now refuse overrides of `index_type` and `rotation_invariant` on load (they shape the stored index).
- Tried and removed: a closed-form learned linear descriptor (LDA-style projection of the luminance pattern). It won on isolated patch pairs (recall at 0.1% false matches 0.64 -> 0.75 at 5x5) but not end to end: 8 dimensions lost recall (0.90 -> 0.825 at JPEG 75), 16 tied. Nearest-neighbour retrieval, not pair separation, is what the engine needs.
- Donor-free tampering cues (TruFor, Noiseprint++) not integrated: their licence forbids profit-oriented use and their weights host was unreachable, so nothing could be tested.

### Measured on real data
- **CASIA v2** cross-image splices (60 + 60 pairs): SPADE IoU 0.27, precision 0.98, recall 0.29, AUC 0.72; RootSIFT + RANSAC IoU 0.18, precision 0.95, recall 0.19, AUC 0.82. One false region across 120 pairs. Much harder than the generated benchmarks; resized/rotated splices are the likely gap.
- Tried and not kept: a copy-move ratio test restricted to a region's own agreeing patches (photo copy-moves: recall 0.47 -> 0.37, AUC 0.96 -> 0.88).

### Added (before the rework)
- **JSON forensic report** (`spade.report`): versioned, audit-defensible report with input SHA-256s, config, localization, the recovered affine color transform `(M,b)`, and image-level `cues`. Written via `spade match -j report.json`. Schema 1.1.
- **Scale-inconsistency forensic cue** (`spade.scale`): native-scale (effective-resolution) estimation plus a resize/resampling-detection signal, surfaced in the report under `cues.scale_inconsistency` with an anomaly bounding box.
- **Spatial-density localization** (`spade.aggregation.localize`): keep the largest connected component of the matched footprints; the component's area fraction is the image-level detection score.
- **Larger-patch support**: `Config.descriptor_spatial_pooling` (compute descriptors directly on the whole patch instead of pooling 3x3 sub-patches - much faster at larger sizes); `patch_size` range widened from 3-6 to 3-16.
- **Benchmark harness** (`spade.bench`): synthetic recolored-splice generator, localization/detection metrics, RootSIFT+RANSAC baseline, and a runner (`python -m spade.bench.runner`). Requires the `bench` extra.

### Changed (before the rework)
- `Match` now carries the recovered affine color transform `(M, b)` (previously computed for scoring then discarded).
- CLI `match`/`serve` `--patch-size` now accepts 3-16.

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

## [0.2.0]

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
