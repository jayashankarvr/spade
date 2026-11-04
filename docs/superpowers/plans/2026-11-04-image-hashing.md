# Image Hashing (PDQ, pHash, PhotoDNA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `spade hash` produces bit-exact PDQ and pHash hashes (and calls PhotoDNA for licensees) and compares them against near-duplicate hash lists.

**Architecture:** A new `spade/hashing.py` holds one small class per algorithm behind a common `compute(image, dihedral) -> list[ImageHash]` interface, plus hash-list loading and vectorised matching. `spade/photodna.py` is the network adapter, kept apart because it is the only code that sends images out. CLI, REST API and the JSON report call into these; nothing else in SPADE changes.

**Tech Stack:** Python 3.10+, numpy, Pillow; optional `pdqhash` and `imagehash` (extra `hash`); libpHash via `ctypes` when the user installs it; `requests` for PhotoDNA (already a dependency of the `bench` extra; add to `hash`).

**Spec:** `docs/superpowers/specs/2026-11-04-image-hashing-design.md`

## Global Constraints

- Every hash is produced by the reference implementation (`pdqhash`, `imagehash`, libpHash); no re-implementations.
- Hashes carry their algorithm; comparing across algorithms raises `ValueError`.
- Algorithm names: `pdq`, `phash`, `phash-lib`, `photodna`.
- Default distances: `pdq` 31 (of 256 bits); `phash` and `phash-lib` 10 (of 64), revised only from Task 7's measurements.
- `min_quality` default 50 (PDQ quality); query hashes below it are skipped and reported.
- Dihedral variant names, in this order: `original`, `rot90`, `rot180`, `rot270`, `flip_x`, `flip_y`, `flip_diag`, `flip_antidiag` (Meta's order for `pdqhash.compute_dihedral`).
- PhotoDNA: off unless named; never in "all" or in reports; credentials only from `SPADE_PHOTODNA_KEY` / `SPADE_PHOTODNA_ENDPOINT`, never logged, stored or echoed.
- libpHash is GPLv3: never bundled, vendored or listed as a dependency; only loaded if the user installed it.
- Missing optional packages raise `spade.exceptions.DependencyError` naming `pip install spade-forensics[hash]`.
- Report schema bumps to `"2.2"`.

## Review Focus

1. A grayscale, palette or RGBA image (e.g. a PNG with alpha) - converted to RGB the way the reference tools expect, not crashing or silently hashing garbage. Test in Task 1.
2. A hash list with Windows line endings, blank lines, upper-case hex or a BOM - parsed, not rejected. Test in Task 3.
3. A list mixing algorithms when the query used only one algorithm - entries of other algorithms ignored, not compared. Test in Task 3.
4. An image smaller than the hash grid (e.g. 4x4) - still hashes (PDQ reports low quality), no crash. Test in Task 1.
5. The PhotoDNA service timing out or returning an error page - a clear error without the key in it. Test in Task 5.

---

### Task 1: PDQ and pHash with reference vectors

**Files:**
- Create: `src/spade/hashing.py`
- Create: `tests/data/hashes/` (Meta's 8 `bridge-*.jpg` from `pdq/data/reg-test-input/dih/` of github.com/facebook/ThreatExchange, plus `README.md` naming the source, commit and licence, and `expected.csv` with `filename,hash,quality` from `pdq/cpp/reg_test/expected/out`)
- Modify: `pyproject.toml` (extra `hash = ["pdqhash>=0.2.8", "imagehash>=4.3", "requests>=2.25.0"]`; add the same to `full`)
- Test: `tests/test_hashing.py`

**Interfaces:**
- Produces:
  - `@dataclass ImageHash: algorithm: str; hex: str; quality: Optional[int]; variant: str = "original"` with property `bits -> np.ndarray` (packed uint8, `len(hex) // 2` bytes)
  - `VARIANTS: tuple[str, ...]` (the eight names above)
  - `DEFAULT_DISTANCE: dict[str, int]` = `{"pdq": 31, "phash": 10, "phash-lib": 10}`
  - `available_algorithms() -> list[str]` (local ones only, never `photodna`)
  - `hash_image(image: np.ndarray, algorithms: Sequence[str] = None, dihedral: bool = False) -> list[ImageHash]` (default `available_algorithms()`; `photodna` raises `ValueError("photodna isn't hashed locally; use spade.photodna")`)
  - `hamming(a: ImageHash | str, b: ImageHash | str) -> int`
  - `dihedral_images(image: np.ndarray) -> list[np.ndarray]` (the eight transforms in `VARIANTS` order, matching Meta's convention)

- [ ] **Step 1: Vendor the reference data.** Fetch the 8 images and the expected hashes (sparse checkout of `pdq/`); write `tests/data/hashes/README.md` and `expected.csv`. Check the repo licence allows redistribution of the images; if it doesn't, keep the expected hashes only and download the images in a test fixture that skips when offline, and say so in the README.
- [ ] **Step 2: Write failing tests** in `tests/test_hashing.py`:
  - `test_pdq_matches_meta_reference`: for each row of `expected.csv`, `hash_image(load_image(path), ["pdq"])[0]` has `.hex == hash` and `.quality == quality`.
  - `test_pdq_dihedral_matches_rotated_files`: `hash_image(original, ["pdq"], dihedral=True)` yields 8 hashes whose `.variant` follow `VARIANTS` and whose hex equal the expected hashes of `bridge-1..8` in that order.
  - `test_dihedral_images_follow_meta_order`: `hash_image(dihedral_images(original)[i], ["pdq"])[0].hex == expected[i]` for all i (pins the transform convention that pHash reuses).
  - `test_phash_matches_imagehash`: equals `str(imagehash.phash(PIL.Image.open(path).convert("RGB")))` for every vendored image.
  - `test_odd_image_modes_hash` (Review Focus 1): a grayscale `(H, W)` array, an RGBA PNG and a palette PNG loaded via `load_image` hash without error, and the RGBA one equals the hash of its RGB conversion.
  - `test_tiny_image` (Review Focus 4): a 4x4 image hashes; PDQ quality < 50.
  - `test_hamming`: same hash -> 0; one flipped bit -> 1; `hamming(pdq_hash, phash_hash)` raises `ValueError`.
  - `test_missing_dependency`: with `pdqhash` import patched to fail, `hash_image(img, ["pdq"])` raises `DependencyError` mentioning `spade-forensics[hash]`.
- [ ] **Step 3: Run** `pytest tests/test_hashing.py -v` -> FAIL (module missing).
- [ ] **Step 4: Implement** `src/spade/hashing.py`: one private class per algorithm with `name`, `bits`, `compute(image, dihedral) -> list[ImageHash]`. PDQ via `pdqhash.compute` / `compute_dihedral` on the RGB uint8 array; convert the returned bit vector to Meta's hex order (the reference test decides the order; if the hex differs from Meta's only by a few bits on the JPEGs, check whether JPEG decoding differs from Meta's tool by hashing a lossless PNG of the same pixels with Meta's own algorithm description, and report the finding instead of loosening the test). pHash via `imagehash.phash` on a PIL image built from the array; for `dihedral=True`, hash each of `dihedral_images(image)`.
- [ ] **Step 5: Run** `pytest tests/test_hashing.py -v` -> PASS.
- [ ] **Step 6: Commit** `feat(hashing): bit-exact PDQ and pHash with Meta's reference vectors`.

### Task 2: libpHash (`phash-lib`)

**Files:**
- Modify: `src/spade/hashing.py`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Consumes: the algorithm class shape and `ImageHash` from Task 1.
- Produces: `phash-lib` in `available_algorithms()` when `ctypes.util.find_library("pHash")` (or `SPADE_PHASH_LIB=/path/libpHash.so`) loads; otherwise absent, and requesting it raises `DependencyError("libpHash isn't installed ... (GPLv3; install it separately, e.g. apt install libphash0)")`.

- [ ] **Step 1: Failing tests:**
  - `test_phash_lib_missing_is_clear`: with the loader patched to find nothing, `available_algorithms()` lacks `phash-lib` and `hash_image(img, ["phash-lib"])` raises `DependencyError` mentioning GPLv3.
  - `test_phash_lib_matches_library` (skipped unless the library loads): for the vendored images, equals `ph_dct_imagehash` called directly on the file, formatted as 16 lowercase hex digits.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** the `phash-lib` class: `ctypes` call `int ph_dct_imagehash(const char *file, uint64_t *hash)`; it takes a file path, so write the array to a temporary PNG (lossless) and remove it after; non-zero return raises `RuntimeError` naming the image.
- [ ] **Step 4: Run** -> PASS (library test skipped here unless installed; try `apt-get install libphash0` in the dev container and run it once, noting the result in the commit message).
- [ ] **Step 5: Commit** `feat(hashing): libpHash via the user's own install`.

### Task 3: Hash lists and matching

**Files:**
- Modify: `src/spade/hashing.py`
- Test: `tests/test_hash_lists.py`

**Interfaces:**
- Consumes: `ImageHash`, `hamming`, `DEFAULT_DISTANCE` from Task 1.
- Produces:
  - `@dataclass HashList: algorithms: np.ndarray (object); hexes: np.ndarray (object); ids: np.ndarray (object)` with `__len__`
  - `load_hash_list(path: str, algorithm: Optional[str] = None) -> HashList`
  - `@dataclass HashMatch: id: str; algorithm: str; hex: str; distance: int; variant: str`
  - `@dataclass MatchReport: matches: list[HashMatch]; skipped_low_quality: list[ImageHash]`
  - `match_hashes(hashes: Sequence[ImageHash], hash_list: HashList, max_distance: Optional[int] = None, min_quality: int = 50) -> MatchReport`

- [ ] **Step 1: Failing tests:**
  - `test_formats`: files with `algorithm,hash,id`; `hash,id` + `algorithm="pdq"`; bare hashes (ids `"1"`, `"2"`, ... by line); a header line - all load the same entries.
  - `test_messy_files_parse` (Review Focus 2): CRLF endings, blank lines, upper-case hex and a UTF-8 BOM load.
  - `test_bad_lines`: a 63-hex-digit PDQ hash, a non-hex character, and a hash-only file without `algorithm=` raise `ValueError` naming the line number.
  - `test_nearest_first_within_distance`: entries at distances 0, 5, 31, 32 from a PDQ query -> matches `[0, 5, 31]` in that order; `max_distance=5` -> `[0, 5]`.
  - `test_other_algorithms_ignored` (Review Focus 3): a list holding `phash` and `pdq` entries; a PDQ-only query matches only PDQ entries.
  - `test_quality_gate`: a query with quality 40 is skipped (in `skipped_low_quality`, no matches); `min_quality=0` matches it; pHash hashes (quality `None`) are never skipped.
  - `test_variant_reported`: a list entry equal to the `rot90` variant reports `variant == "rot90"`.
  - `test_million_entries_fast`: 1,000,000 random PDQ entries, one query, `match_hashes` under 1.0 s.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement:** parse with `csv` after stripping a BOM; store per-algorithm packed byte matrices built once per list (cache on the `HashList`); distances via `np.unpackbits(np.bitwise_xor(matrix, query), axis=1).sum(axis=1)` or a popcount table; same algorithm only.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(hashing): hash lists and vectorised matching`.

### Task 4: `spade hash` CLI

**Files:**
- Modify: `src/spade/cli.py`
- Test: `tests/test_cli_hash.py`

**Interfaces:**
- Consumes: `hash_image`, `available_algorithms`, `load_hash_list`, `match_hashes` (Tasks 1-3); `PhotoDnaClient` (Task 5) only when `--algorithm photodna` is given (import inside the branch).
- Produces: `spade hash IMAGE... [--algorithm NAME]... [--against LIST] [--max-distance N] [--min-quality 50] [--thorough] [--json OUT]`.

- [ ] **Step 1: Failing tests** (click `CliRunner`, vendored images):
  - `test_prints_hashes`: output has one line per image and algorithm with the hex and, for PDQ, `quality=`.
  - `test_against_list`: a list containing the original's PDQ hash -> output names that id with `distance=0`; a list without it -> `no match`; exit code 0 both times.
  - `test_thorough_matches_rotation`: a list holding only `bridge-2-rotate-90`'s hash: no match by default, a match with `--thorough` naming variant `rot90`.
  - `test_json_output`: `--json` writes `{"images": [{"path", "hashes": [...], "matches": [...], "skipped_low_quality": [...]}]}`.
  - `test_errors`: unreadable image, malformed list, `--algorithm phash-lib` without the library -> non-zero exit with a message (no traceback).
  - `test_all_excludes_photodna`: `--algorithm all` never calls PhotoDNA.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** the command; `all` expands to `available_algorithms()`; `--thorough` sets `dihedral=True`.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(cli): spade hash`.

### Task 5: PhotoDNA adapter (licensees)

**Files:**
- Create: `src/spade/photodna.py`
- Modify: `src/spade/cli.py` (the `photodna` branch of `spade hash`)
- Test: `tests/test_photodna.py`

**Interfaces:**
- Produces: `class PhotoDnaClient: __init__(self, key: Optional[str] = None, endpoint: Optional[str] = None, timeout: float = 30.0)` (defaults from `SPADE_PHOTODNA_KEY` / `SPADE_PHOTODNA_ENDPOINT`); `match(self, image_bytes: bytes) -> dict` (the service's JSON verdict, passed through); `PhotoDnaError(Exception)`.

- [ ] **Step 1: Failing tests** (the HTTP layer mocked with `unittest.mock.patch` on `requests.post`):
  - `test_no_key`: constructing without a key raises `PhotoDnaError` saying a Microsoft PhotoDNA licence is needed and where to apply; the message holds no key.
  - `test_sends_image_and_key_header`: the request carries the key in the header Microsoft's API documentation names, the image bytes in the body, the configured endpoint and the timeout; returns the response JSON.
  - `test_service_error_hides_key` (Review Focus 5): a timeout, an HTTP 500 and an HTML error page each raise `PhotoDnaError`; `str(error)` and the log output never contain the key.
  - `test_cli_needs_explicit_name`: `spade hash img --algorithm photodna` with a mocked client prints the verdict; plain `spade hash img` never constructs a client.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** against Microsoft's published PhotoDNA Cloud Service API reference for the request (endpoint path, key header, body format); the module docstring states it is tested only against a mock until a licensed key is available.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(photodna): Cloud Service adapter for licensees`.

### Task 6: REST `/hash` and report hashes

**Files:**
- Modify: `src/spade/api.py`, `src/spade/cli.py` (`serve --hash-list`), `src/spade/report.py`
- Test: `tests/test_api.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: Tasks 1-3, 5.
- Produces: `POST /hash?algorithm=NAME` (repeatable) -> `{"hashes": [{"algorithm", "hash", "quality", "variant"}], "matches": [...]?}`; `serve --hash-list PATH` / `SPADE_HASH_LIST`; report input entries gain `"hashes": {"pdq": {"hash", "quality"}, "phash": {"hash"}}`; `SCHEMA_VERSION = "2.2"`.

- [ ] **Step 1: Failing tests:**
  - `test_hash_endpoint`: upload a vendored image -> PDQ hex equals the reference; no `matches` key without a server list.
  - `test_hash_endpoint_with_list`: app created with a hash list -> `matches` contains the reference id at distance 0.
  - `test_hash_endpoint_missing_algorithm`: `?algorithm=phash-lib` without the library -> 501 with the install hint.
  - `test_photodna_needs_server_key`: `?algorithm=photodna` without `SPADE_PHOTODNA_KEY` on the server -> 403 with an explanation.
  - `test_report_records_hashes`: `build_report(...)` for an input image includes `hashes.pdq.hash` equal to `hash_image(...)` and never a `photodna` key; `schema_version == "2.2"`.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement**; hashing in the API runs in the thread pool like matching; the report computes hashes only when the optional packages are installed.
- [ ] **Step 4: Run** -> PASS (plus `pytest -q` for the whole suite).
- [ ] **Step 5: Commit** `feat(api,report): /hash endpoint and hashes in the forensic report`.

### Task 7: Measured thresholds and documentation

**Files:**
- Create: `src/spade/bench/hash_robustness.py`
- Modify: `src/spade/hashing.py` (`DEFAULT_DISTANCE` only if the measurement says so), `README.md`, `CHANGELOG.md`, `docs/API.md`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Produces: `python -m spade.bench.hash_robustness` printing, per algorithm, the largest distance between each benchmark photo and its edited versions (resize 0.5x and 2x, JPEG quality 70, colour grade 0.15, 5% crop) and the smallest distance between unrelated photos.

- [ ] **Step 1: Failing tests:**
  - `test_edited_copies_within_distance`: for every scikit-image benchmark photo, resize 0.5x/2x, JPEG 70 and colour grade 0.15 stay within `DEFAULT_DISTANCE` for `pdq` and `phash`.
  - `test_unrelated_photos_beyond_distance`: every pair of different benchmark photos is further apart than `DEFAULT_DISTANCE`.
- [ ] **Step 2: Run the measurement script** and record its table; if the pHash default of 10 fails either test, choose the value that separates them with the largest margin on both sides and explain it in the README (note that the benchmark is small).
- [ ] **Step 3: Run** `pytest tests/test_hashing.py -v` -> PASS.
- [ ] **Step 4: Documentation:** README section "Image hashing" (purpose of each algorithm; commands; distances and the quality rule with the measured table; licensing: libpHash GPLv3 installed separately, PhotoDNA needs a Microsoft licence and sends the image to Microsoft and is tested only against a mock; what perceptual hashes don't survive - heavy cropping, arbitrary-angle rotation, large local edits - and that SPADE's region search covers those); CHANGELOG entry; `docs/API.md` entry for `/hash`.
- [ ] **Step 5: Commit** `docs(hashing): measured thresholds, README, API docs`.
