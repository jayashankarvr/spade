# PDQ image hashing for near-duplicate lookup

## Goal

Let people check an image against near-duplicate hash lists shared between
organisations (Meta ThreatExchange, NCMEC/StopNCII-style lists). Those lists
use PDQ, Meta's 256-bit perceptual hash, so SPADE must produce PDQ hashes that
are bit-for-bit identical to Meta's reference implementation, and must be able
to compare them against a list.

This sits next to SPADE's region matching, not in place of it: PDQ answers
"is this the same picture, give or take resizing, recompression and small
edits?"; SPADE's region search answers "where did this piece come from?".

## Decisions

- **Implementation:** the `pdqhash` package (BSD), a wrapper around Meta's
  reference C++ code. Bit-exact by construction; also gives the 8 dihedral
  variants (rotations by 90/180/270 degrees and mirror images). Optional extra:
  `pip install spade-forensics[hash]`. Re-implementing PDQ was rejected: any
  rounding difference breaks interoperability.
- **Scope:** produce the hash and compare against a hash list. Running SPADE's
  region match on list hits is out of scope for now.

## Components

### `spade/hashing.py`

- `PdqHash` (dataclass): `hex` (64 lowercase hex digits), `quality` (int,
  0-100), `bits` (numpy uint8 array of 32 bytes, for fast comparison).
- `pdq_hash(image, dihedral=False) -> PdqHash | list[PdqHash]`: image is an RGB
  uint8 array (as `spade.engine.load_image` returns). With `dihedral=True`,
  returns the 8 variants, the original first.
- `hamming(a, b) -> int`: bit distance between two hashes (hex strings or
  `PdqHash`).
- `load_hash_list(path) -> HashList`: reads a CSV or text file. Each line holds
  a 64-hex-digit hash and optionally an id (CSV columns `hash,id` with or
  without a header; or `hash<whitespace>id`; or the hash alone, with the line
  number as id). Lines that don't parse raise `ValueError` naming the line.
- `match_hashes(hashes, hash_list, max_distance=31, min_quality=50) -> list[HashMatch]`:
  every list entry within `max_distance` of any of `hashes`, nearest first.
  `HashMatch` holds the list id, its hash, the distance and which query hash
  (variant index) matched. Comparison is vectorised (XOR + popcount on packed
  bytes): a million-entry list compares in well under a second.
- Quality: PDQ's guidance is that hashes with quality below 50 (near-blank or
  flat images) are unreliable. `match_hashes` skips query hashes below
  `min_quality` and says so in the result (`skipped_low_quality`); callers can
  pass `min_quality=0` to compare anyway.
- If `pdqhash` isn't installed, `pdq_hash` raises `DependencyError` with the
  install command (same pattern as the API and bench extras).

### Defaults: quick and thorough

- Default: the hash of the image as given.
- `thorough`: the 8 dihedral variants are compared, so rotated-by-90 and
  mirrored copies match too.
- `max_distance` default 31, PDQ's published match threshold.

### CLI

`spade hash IMAGE... [--against LIST] [--max-distance 31] [--min-quality 50] [--thorough] [--json OUT]`

- Without `--against`: prints each image's hash and quality.
- With `--against`: also prints the matching list entries (id, distance,
  variant), or "no match"; exit code 0 either way, errors (unreadable image,
  bad list line) exit non-zero with a clear message.
- `--json`: writes the same information as JSON.

### REST API

- `POST /hash` (file upload): returns `{"hash", "quality", "matches"}`.
  `matches` is present when the server was started with `--hash-list PATH`
  (or `SPADE_HASH_LIST`), so the list stays on the server; `thorough` follows
  the server's config. Without `pdqhash` installed, the endpoint returns 501
  with the install hint.

### JSON forensic report

Each input image entry gains `pdq: {"hash", "quality"}` next to the SHA-256 it
already records, when `pdqhash` is installed (omitted otherwise). Report
schema version bumps to 2.2.

## Accuracy checks (tests)

1. **Bit-exactness:** hashes of Meta's published reference test images match
   Meta's published hashes exactly (vectors vendored into `tests/data/pdq/`
   with their source noted). This is the acceptance test for the feature.
2. **Robustness:** for the scikit-image benchmark photos, resized (0.5x, 2x),
   JPEG-recompressed (quality 70) and slightly recolored versions stay within
   distance 31 of the original; 90-degree rotations and mirror images match
   only with `thorough`.
3. **Separation:** hashes of unrelated benchmark photos are all further apart
   than 31 (record the minimum observed distance in the test).
4. **Hash-list parsing:** each accepted format, malformed lines, duplicates.
5. **Quality gate:** a flat image hashes with quality below 50 and is skipped
   in matching unless `min_quality=0`.

## Documentation

README section "Image hashing (PDQ)": what it is for, the commands, the
threshold and quality rule, and what PDQ doesn't survive (heavy cropping,
arbitrary-angle rotation, large local edits), pointing to SPADE's region
search for those cases. CHANGELOG entry. `docs/API.md` entry for `/hash`.

## Out of scope

- Other hash families (pHash, PhotoDNA, TMK video hashes).
- Running SPADE's region match automatically on list hits.
- Hash databases larger than memory or indexed search (FAISS binary index) -
  revisit if lists beyond ~10 million entries come up.
