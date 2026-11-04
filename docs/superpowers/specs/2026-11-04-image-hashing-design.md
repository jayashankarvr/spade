# Perceptual image hashing for near-duplicate lookup (PDQ, pHash, PhotoDNA)

## Goal

Let people check an image against near-duplicate hash lists - the lists
organisations share (Meta ThreatExchange, NCMEC/StopNCII-style lists) and
older databases built with pHash. A hash is only useful for this if it is
bit-for-bit identical to what built the database, so every algorithm uses the
reference implementation, never a re-implementation. SPADE produces the hashes
and compares them against a list.

This sits next to SPADE's region matching, not in place of it: a perceptual
hash answers "is this the same picture, give or take resizing, recompression
and small edits?"; SPADE's region search answers "where did this piece come
from?".

## Algorithms

| name | what | bits | implementation | how it's obtained | default match distance |
|---|---|---|---|---|---|
| `pdq` | Meta's PDQ | 256 | `pdqhash` (BSD), wraps Meta's reference C++ | optional extra `spade-forensics[hash]` | 31 (PDQ's published threshold) |
| `phash` | DCT pHash as in the Python `imagehash` library | 64 | `imagehash` (BSD) | optional extra `spade-forensics[hash]` | 10 |
| `phash-lib` | `ph_dct_imagehash` from libpHash (phash.org) | 64 | the user's own libpHash install, called through `ctypes` | installed separately by the user: libpHash is GPLv3, so SPADE (Apache-2.0) never ships or bundles it | 10 |
| `photodna` | Microsoft PhotoDNA | - | Microsoft's PhotoDNA Cloud Service | the user's own PhotoDNA licence and API key | decided by the service |

The pHash thresholds (10 of 64 bits) are starting values: the accuracy tests
below measure robustness and separation on our benchmark photos, and the
defaults are set from those measurements and recorded in the README.

### PhotoDNA

PhotoDNA is proprietary; Microsoft licenses it only to vetted organisations,
usually through its PhotoDNA Cloud Service with an approved key. There is no
legitimate open implementation, and SPADE will not use reverse-engineered
ones. SPADE provides an adapter that sends an image to the Cloud Service with
the user's own credentials and reports what the service returns. Rules:

- Off unless asked for explicitly (`--algorithm photodna` / `algorithms=["photodna"]`);
  never part of "all" defaults, because it sends the image to an outside
  service.
- Credentials from the environment only (`SPADE_PHOTODNA_KEY`, and
  `SPADE_PHOTODNA_ENDPOINT` if the licence uses a non-default endpoint); never
  written to disk, logs, reports or error messages.
- Without a key, the command explains that PhotoDNA needs a Microsoft licence
  and where to apply.
- The request and response fields follow Microsoft's API documentation that
  comes with the licence. Until a real key is available, the adapter is tested
  against a mocked service only, and the README says so.

## Components

### `spade/hashing.py`

One small interface, one class per algorithm (each a few dozen lines):

- `ImageHash` (dataclass): `algorithm`, `hex` (lowercase hex), `bits`,
  `quality` (0-100 where the algorithm has one, else `None`), `variant`
  (`"original"` or the dihedral transform: `rot90`, `rot180`, `rot270`,
  `flip`, `flip_rot90`, ...).
- `hash_image(image, algorithms=("pdq", "phash"), dihedral=False) -> list[ImageHash]`:
  image is an RGB uint8 array (as `spade.engine.load_image` returns). With
  `dihedral=True`, each algorithm also hashes the 7 rotated/mirrored variants
  (`pdq` uses its own dihedral routine; the others hash transformed copies).
  `photodna` isn't hashed locally; see the adapter.
- `available_algorithms() -> list[str]`: which ones are usable here.
- `hamming(a, b) -> int`: bit distance; refuses hashes of different
  algorithms.
- `load_hash_list(path, algorithm=None) -> HashList`: CSV or text. Lines are
  `algorithm,hash,id`, `hash,id` or a bare hash (line number as id); a header
  line is allowed. Files without an algorithm column need `algorithm=`
  (CLI `--algorithm`), and a hash whose length doesn't fit the algorithm is an
  error naming the line.
- `match_hashes(hashes, hash_list, max_distance=None, min_quality=50) -> MatchReport`:
  entries within each algorithm's distance (or `max_distance`), nearest first,
  compared only within the same algorithm. Vectorised XOR + popcount on packed
  bytes: a million entries in well under a second. Query hashes with a quality
  below `min_quality` (PDQ's guidance: near-blank or flat images) are skipped
  and listed in `skipped_low_quality`; `min_quality=0` compares anyway.
- `PhotoDnaClient`: `match(image) -> dict` (the service's verdict, passed
  through with its fields), configured from the environment as above.
- A missing optional package raises `DependencyError` with the install
  command; a missing libpHash says how to install it.

### Quick and thorough

- Default: hashes of the image as given.
- `thorough`: also the 7 rotated/mirrored variants, so 90-degree rotations
  and mirror images match.

### CLI

`spade hash IMAGE... [--algorithm pdq|phash|phash-lib|photodna|all] [--against LIST] [--max-distance N] [--min-quality 50] [--thorough] [--json OUT]`

- `--algorithm` may be repeated; default is every available local algorithm
  (`pdq`, `phash`, and `phash-lib` when installed). `all` means the same;
  `photodna` must always be named on its own.
- Without `--against`: prints each image's hashes (and quality).
- With `--against`: also prints matching entries (algorithm, id, distance,
  variant) or "no match". Exit code 0 either way; unreadable images, bad list
  lines or missing dependencies exit non-zero with a clear message.

### REST API

- `POST /hash?algorithm=...` (file upload): returns the hashes, plus
  `matches` when the server was started with `--hash-list PATH`
  (`SPADE_HASH_LIST`), so the list stays on the server. `thorough` follows the
  server config. A requested algorithm that isn't installed returns 501 with
  the install hint. `photodna` is allowed only if the server operator enabled
  it (`SPADE_PHOTODNA_KEY` set on the server).

### JSON forensic report

Each input image entry gains `hashes: {"pdq": {"hash", "quality"}, "phash": {"hash"}}`
for the local algorithms installed, next to the SHA-256 it already records.
Never PhotoDNA (it would send the image out). Report schema 2.2.

## Accuracy checks (tests)

1. **Bit-exactness, the acceptance test:**
   - `pdq`: Meta's published reference images and hashes (from the
     ThreatExchange repository), vendored into `tests/data/hashes/` with their
     source noted; all must match exactly.
   - `phash`: hashes of the vendored images compared with `imagehash` run
     directly (guards against our image preparation drifting, e.g. colour
     conversion), plus hashes published by `imagehash`'s own tests where
     available.
   - `phash-lib`: compared with libpHash's own command-line output when the
     library is installed (test skipped otherwise).
   - If reference vectors can't be obtained, that is reported, not worked
     around with weaker tests.
2. **Robustness:** for the scikit-image benchmark photos, resized (0.5x, 2x),
   JPEG-recompressed (quality 70) and slightly recolored versions stay within
   each algorithm's distance; 90-degree rotations and mirror images match only
   with `thorough`.
3. **Separation:** unrelated benchmark photos stay further apart than the
   distance; the smallest observed distance is recorded. Robustness and
   separation together set the pHash defaults.
4. **Hash lists:** every accepted format, algorithm column vs `--algorithm`,
   wrong-length hashes, malformed lines, mixing algorithms refused.
5. **Quality gate:** a flat image hashes with PDQ quality below 50 and is
   skipped unless `min_quality=0`.
6. **PhotoDNA adapter:** against a mocked service - request shape, key taken
   from the environment and never echoed, clear errors for no key / HTTP
   failures, off unless named.

## Documentation

README section "Image hashing": what each algorithm is for, the commands, the
thresholds and quality rule, the licensing situation (libpHash is GPLv3 and
installed separately; PhotoDNA needs a Microsoft licence and sends the image
to Microsoft), and what perceptual hashes don't survive (heavy cropping,
arbitrary-angle rotation, large local edits), pointing to SPADE's region
search for those. CHANGELOG entry; `docs/API.md` entry for `/hash`.

## Out of scope

- Other hash families (aHash, dHash, wavelet hash, TMK video hashes).
- Running SPADE's region match automatically on list hits.
- Hashing sub-regions of an image (e.g. to find a known image pasted into a
  collage or screenshot). Shared lists hold whole-image hashes, so it would
  mean hashing a grid of windows with a threshold corrected for the number of
  windows, checked on a collage benchmark. Possible later, if needed.
- Hash databases larger than memory or indexed search (FAISS binary index);
  revisit beyond ~10 million entries.
