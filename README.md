# SPADE

Sub-Patch Analysis & Detection Engine - finds where a small region of an image was copied from, across a collection of images or within one image, even after recoloring, JPEG recompression and shrinking. Every reported region comes with a significance score (an a-contrario NFA) you can put in a report.

The default search covers the common edits and is quick; `--thorough` also checks mirrored, enlarged and rotated copies at about 13x the time (see [Quick or Thorough](#quick-or-thorough)).

## Install

```bash
pip install -e .
# or when published:
pip install spade-forensics

# With optional features:
pip install spade-forensics[api]    # REST API
pip install spade-forensics[bench]  # Benchmark harness (OpenCV baseline)
pip install spade-forensics[hash]   # Perceptual image hashes (PDQ, pHash)
pip install spade-forensics[full]   # Everything
```

Requires: numpy, scipy, pillow, scikit-learn, faiss-cpu, click

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start the API server
docker-compose up

# Index an image
curl -X POST "http://localhost:8000/index?image_id=target_001" \
  -F "file=@target.jpg"

# Match a source image
curl -X POST "http://localhost:8000/match?return_heatmap=true" \
  -F "file=@source.jpg"
```

### Option 2: Python Module

```bash
# Start API server
python -m spade serve --port 8000

# Show version
python -m spade version

# Run tests
python -m spade test
```

### Option 3: Makefile

```bash
make install      # Install in development mode
make test         # Run tests
make serve        # Start API server
make docker-build # Build Docker image
```

## Usage

### CLI

```bash
# Match source against target
spade match source.jpg target.jpg

# With options: patch size, heatmap, JSON forensic report
spade match source.jpg target.jpg -p 6 -o heatmap.png -j report.json -v

# Also check mirrored, enlarged and rotated copies (slower)
spade match source.jpg target.jpg --thorough

# Index a directory (writes myindex.index, myindex.meta.json, myindex.engine.json;
# with --thorough also myindex.rotated.*)
spade index images/ -o myindex

# Search against the saved index; patches are reloaded from the indexed files
spade search query.jpg myindex -k 5

# Scan many questioned images, holding the false discovery rate at 5%
spade scan myindex questioned/*.jpg --fdr 0.05

# Visual report a person can review (plus the JSON record)
spade match source.jpg target.jpg --html report.html -j report.json

# Which images in a folder share copied regions (and, optionally, copy-moves)
spade provenance photos/ --copy-moves --json links.json --dot links.dot
```

### Python API

#### Basic Usage

```python
from spade import ForensicsEngine
from spade.engine import load_image

engine = ForensicsEngine()
engine.index_target(load_image("suspect_photo.jpg"), "suspect_001")
engine.index_target(load_image("crime_scene.jpg"), "scene_001")

result = engine.match(load_image("questioned.jpg"))

for region in result.significant_regions:
    print(f"Copied from {region.image_id}")
    print(f"  source bbox {region.source_bbox} -> target bbox {region.target_bbox}")
    print(f"  {len(region.matches)} agreeing patches, log10 NFA {region.log10_nfa:.1f}")
    print(f"  color transform M={region.color_M.round(3)}, b={region.color_b.round(3)}")

# Where in the questioned image the copied pixels are (bool mask)
mask = result.localization.mask
```

`log10_nfa` is the log of the expected number of regions at least this good in
images with no copy at all. -5 means you'd expect one such coincidence in
100,000 comparable searches. Regions are reported as significant when NFA < 1
(`Config.max_nfa`).

Individual `Match` objects carry a `probability`, which is a per-patch
chi-square goodness-of-fit p-value. It's a pre-filter, not evidence on its own:
small patches fit each other by chance all the time.

#### Custom Configuration

```python
from spade import ForensicsEngine, Config

config = Config(
    patch_size=6,            # 3-16; 5 is the benchmarked default
    stride=2,                # skip positions for speed
    noise_sigma=0.03,        # expected noise; raise for heavily compressed images
    max_nfa=0.01,            # stricter significance
    pyramid_enabled=False,   # index targets at full size only (faster)
    max_targets=100,         # evict least recently indexed targets past this
)
engine = ForensicsEngine(config)
```

#### Copy-Move Within One Image

```python
result = engine.match_self(load_image("questioned.jpg"))
for region in result.significant_regions:
    print(region.source_bbox, "duplicated at", region.target_bbox, region.log10_nfa)
```

The image is matched against itself, ignoring offsets shorter than twice the
patch size. Each duplicated pair is reported once; `localization` covers both
copies. Repetitive texture (brick, gravel, lattices) matches itself at many
offsets, so a region only counts when its offset clearly beats every other
offset its patches support - a region-level ratio test. On 60 recolored
real-photo copy-moves plus 60 clean crops it reports no false regions at any
threshold and finds 70-82% of the copies (NFA 1e-4 to 10). Content that really
is duplicated is still reported: scikit-image's `grass` and `gravel` photos
each contain a cloned strip, and SPADE finds them, so the benchmark leaves
those photos out.

With mirroring on (`flip_invariant`, part of `thorough`), mirrored copies are
searched too, which raises a trap: a left-right symmetric
object (a face, a facade, the round edge of a microscope or fundus photo)
matches its own mirror image. Its two halves sit side by side on the same
rows, so mirrored regions shaped like that are dropped. The price is a
mirrored copy pasted right beside its source at the same height, which looks
exactly like a symmetric object: 1 of 40 copies in the mirrored copy-move
benchmark (found at NFA 1e-3: 72.5% -> 70%, still no false regions).

#### JPEG-Aware Noise

A JPEG file's quantization tables say how much rounding error to expect. The
CLI and API read them automatically; from Python pass them yourself:

```python
from spade.jpeg import quantization_sigma

engine.index_target(load_image(path), path, image_path=path)   # read from the file
result = engine.match(image, jpeg_sigma=quantization_sigma("questioned.jpg") or 0.0)
```

#### Quick or Thorough

Most copied fragments are cut out (cropped), recolored, recompressed or
shrunk, and the default search covers exactly those. Mirrored, enlarged and
rotated copies are rarer and cost several extra searches each, so they are
checked only when asked for:

```python
engine = ForensicsEngine(Config(thorough=True))   # CLI: spade match --thorough ...
```

| Edit to the copied fragment | default | `thorough` |
|---|---|---|
| cut out / cropped, moved | yes | yes |
| recolored (brightness, contrast, colour grade, tone curve) | yes | yes |
| JPEG recompression | yes | yes |
| shrunk | yes, cross-image (target pyramid) | yes, cross-image |
| mirrored (left-right; up-down too, as mirror plus 180 degrees) | - | yes |
| enlarged | - | yes, cross-image (`query_scales`) |
| rotated | - | yes, cross-image (a second, ring-based search) |

Not checked even with `thorough`: a copy that is both rotated and rescaled;
rotated or rescaled copies within one image (copy-move gets mirroring only);
non-uniform scaling, shear or perspective. `thorough` takes about 13x as long
as the default and holds a second index (about twice the memory); a thorough
`spade index` writes `.rotated.*` files next to the usual three, which belong
with it.

`thorough` combines its searches into one result and multiplies every region's
NFA by the number of searches combined, so the NFA keeps its meaning. With
`spade index`, decide when indexing: `spade search` and `spade scan` follow
the index. The pieces can also be switched on one by one
(`flip_invariant`, `query_scales`, `rotation_invariant`).

#### Mirrored Copies

`Config(flip_invariant=True)` (part of `thorough`) also searches the mirrored
query, so a fragment that was flipped before pasting is found
(`Match.mirrored`, `CoherentRegion.mirrored`). On photo splices where half the
fragments are mirrored it doubles recall (0.45 -> 0.90); on copy-move in real
photos it adds no false alarms. It takes about 1.5x as long (the mirrored pass
runs on a second thread).

#### Rotated Copies (experimental)

`Config(rotation_invariant=True)` compares patches as colour samples on rings
around their centres instead of as squares, and groups matches by rotation +
translation (it indexes targets at full size only; the pyramid isn't supported
yet). The ring descriptor doesn't change when a patch turns, and each match's
angle is measured by aligning the two sets of rings, so no per-patch
orientation has to be estimated. On 20 photo splices rotated by up to 180
degrees (plus 20 unrelated pairs) it finds 60% at NFA < 10^-3 with IoU 0.51,
AUC 0.995 and no false regions, against 50% / 0.34 / 0.84 with the earlier
design that turned each patch to its own estimated orientation. It's off by
default because it is slower (about 20 s per 192 px pair) and has limits:

- The rings read about twice the patch width, so a copy narrower than about
  17 px (patch 5) can't be found.
- Smooth, gently varying content has little structure around a ring to
  match on.
- Within one image, a round object with a radial pattern matches itself
  turned; regions that turn about a point inside both copies are dropped
  for that reason.
- Rotated copy-moves (a region copied and turned within the same image) are
  mostly missed. The repeated-texture test copy-move needs compares plain
  offsets, which a rotation spreads out; without it, edges matching edges at
  any angle raise alarms on over a third of clean photos, and a version
  comparing whole rigid transforms didn't separate the two on our test
  images. Cross-image search (a rotated fragment from another image) is
  what this mode handles. Switched on by hand for copy-move, it also raises
  more alarms on clean images: on CoMoFoD (25 copies + 25 originals each) it
  found 28% of rotated copies with 1 original flagged, and 36% of plain
  translated copies with 2 originals flagged, where the default finds 40% and
  flags none. `thorough` leaves it out of copy-move for that reason.

#### Large Collections

The index (`index_type`, default `"ivfpq"`) stores descriptors with product
quantization in an inverted file once `ivf_train_size` patches (default
50,000) are indexed; below that, search is exact. On 200,000 descriptors it is
5x smaller (4.3 vs 21.6 MB) and searches 22x faster; on real-photo splices
forced to use it, results matched the exact index (IoU 0.818, recall 0.975,
no false regions). Only candidate generation is approximate - every
candidate is still verified against real pixels. `index_type="flat"` keeps
search exact at any size.

Targets keep their pixels once (uint8) rather than a patch per position,
which is most of the memory regardless of the index.

#### Multi-Scale Matching

Targets are indexed at full, 3/4 and 1/2 size by default, so a copy that was
shrunk before pasting is still found. Matches record the level they came from
in `target_scale`. On 60 spliced + 60 unrelated CASIA v2 pairs this took IoU
from 0.27 to 0.37 and recall from 0.29 to 0.39 with no extra false regions
(1 in 120 pairs either way), for about 17% more time per image.

```python
config = Config(pyramid_scales=[1.0, 0.5, 0.25])   # or pyramid_enabled=False
engine = ForensicsEngine(config)
```

The pyramid only covers copies that got smaller. For copies that were enlarged,
`query_scales=[1.0, 0.75, 0.5]` also searches the questioned image shrunk to
those sizes. Each extra scale is another full search, and the NFA is
multiplied by the number of passes. On 40 + 40 CASIA v2 pairs it raised the
share of splices found from 47.5% to 50% (57.5% at NFA < 1) at over 4x the
time, so it's off by default and part of `thorough`.

#### Heatmap and Report

```python
from spade.report import build_report, write_report

result = engine.match(source, return_heatmap=True)
result.heatmap          # (H, W) in [0, 1] over the best region's target image

report = build_report(result=result, config=engine.config,
                      source_path="questioned.jpg", source_image=source)
write_report(report, "report.json")
```

#### Save and Load an Index

```python
engine = ForensicsEngine()
for path in paths:
    engine.index_target(load_image(path), path, image_path=path)
engine.save("myindex")

# Later, possibly in another process. Matching settings can be overridden;
# patch size, stride and pyramid always come from the saved bundle.
engine = ForensicsEngine.load("myindex", max_nfa=0.1)
result = engine.match(source)
```

Patches aren't stored in the bundle. They're re-extracted from `image_path` the
first time a target is needed, so keep the indexed files where they were.

### REST API

```bash
# Start the API server
python -m spade serve --port 8000

# Or using Docker
docker-compose up

# Or with custom config via environment variables
export SPADE_PATCH_SIZE=4
export SPADE_MAX_TARGETS=200
python -m spade serve
```

**API Endpoints:**
- `GET /` - Health check
- `GET /health` - Comprehensive health info
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/ready` - Kubernetes readiness probe
- `GET /config` - Get current configuration
- `POST /index` - Upload and index an image
- `POST /match` - Match an image against indexed targets
- `GET /targets` - List indexed images
- `DELETE /targets/{id}` - Remove an indexed image

**Interactive Docs:** `http://localhost:8000/docs`
**API Documentation:** See [docs/API.md](docs/API.md) for details

#### Example: Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Index a target
with open("target.jpg", "rb") as f:
    response = requests.post(f"{BASE_URL}/index", params={"image_id": "target_001"}, files={"file": f})
    print(response.json())

# Match a source
with open("source.jpg", "rb") as f:
    result = requests.post(f"{BASE_URL}/match", files={"file": f}).json()

if result["detected"]:
    for region in result["coherent_regions"]:
        if region["significant"]:
            print(region["image_id"], region["source_bbox"], region["log10_nfa"])
```

### Environment Variables

Configure SPADE via environment variables:

```bash
export SPADE_PATCH_SIZE=5             # Patch size (3-16)
export SPADE_STRIDE=1                 # Patch stride
export SPADE_K_NEIGHBORS=8            # Nearest neighbours per source patch
export SPADE_DISTANCE_THRESHOLD=0.5   # Max squared descriptor distance
export SPADE_MIN_PROBABILITY=0.01     # Per-patch fit p-value floor
export SPADE_NOISE_SIGMA=0.02         # Expected noise (0-1 scale)
export SPADE_MAX_NFA=1.0              # Significance threshold for regions
export SPADE_ENTROPY_THRESHOLD=       # Optional texture floor in bits (unset = off)
export SPADE_PYRAMID_ENABLED=false    # Index targets at full size only
export SPADE_THOROUGH=true            # Also check mirrored, enlarged and rotated copies
export SPADE_MAX_TARGETS=1000         # Evict least recently indexed targets
```

### Deployment

#### Docker

```bash
# Build image
docker build -t spade-forensics .

# Run container
docker run -p 8000:8000 -v ./data:/data spade-forensics

# Or use docker-compose
docker-compose up

# With Nginx reverse proxy
docker-compose --profile with-nginx up
```

#### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spade-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spade
  template:
    metadata:
      labels:
        app: spade
    spec:
      containers:
      - name: spade
        image: spade-forensics:latest
        ports:
        - containerPort: 8000
        env:
        - name: SPADE_MAX_TARGETS
          value: "100"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: spade-data
          mountPath: /data
      volumes:
      - name: spade-data
        persistentVolumeClaim:
          claimName: spade-pvc
```

## Image hashing

`spade hash` computes perceptual hashes and checks them against lists of known
hashes, such as the lists organisations share through Meta's ThreatExchange or
older databases built with pHash. A hash answers "is this the same picture,
give or take resizing, recompression and small edits?". It doesn't say where a
piece of a picture came from; that is what SPADE's region search is for.

A list only works with hashes made the same way, bit for bit, as the ones in
it, so every algorithm here comes from its reference library:

| Algorithm | What it is | Bits | Comes from | Match at distance |
|---|---|---|---|---|
| `pdq` | Meta's PDQ, the format of ThreatExchange lists | 256 | `pdqhash` (MIT, bundling Meta's BSD C++), `pip install spade-forensics[hash]` | 31 or less |
| `phash` | DCT pHash as computed by the Python `imagehash` library | 64 | `imagehash`, same extra | 10 or less |
| `phash-lib` | `ph_dct_imagehash` from libpHash (phash.org), for databases built with the C library | 64 | your own libpHash install | 10 or less |
| `photodna` | Microsoft PhotoDNA | - | Microsoft's PhotoDNA Cloud Service | decided by the service |

`pdq` and `phash` hashes of the same image are different numbers and are
never compared with each other; neither is `phash` with `phash-lib`.

```bash
# Print the hashes of some images (every local algorithm installed here)
spade hash photo.jpg other.png

# Match against a list; rows are algorithm,hash,id (a header line is fine)
spade hash photo.jpg --against known.csv

# Also hash the eight rotations and flips, so rotated or mirrored copies match
spade hash photo.jpg --against known.csv --thorough

# One algorithm, a stricter distance, and the result as JSON
spade hash photo.jpg -a pdq --against known.csv --max-distance 20 --json out.json
```

```
photo.jpg pdq d8f8f0cee0f4a84f0637022a078f67f0b36e2ed596621e1d33e6339c4e9c9b22 quality=100
photo.jpg phash aca29c1c33dc23d7
  match bridge (pdq) distance=0 variant=original
```

A list can also leave out the algorithm column (`hash,id`, or one bare hash
per line) if you name its algorithm with a single `--algorithm`, e.g.
`spade hash photo.jpg -a pdq --against pdq-list.txt`.

In Python, `spade.hashing.hash_image`, `load_hash_list` and `match_hashes` do
the same (`load_hash_list(path, algorithm="pdq")` for lists without an
algorithm column). On the server, `POST /hash` does the same (see
[docs/API.md](docs/API.md#post-hash)), with the list given to
`spade serve --hash-list known.csv` (or `SPADE_HASH_LIST`) so it stays on the
server. The JSON forensic report records each input image's hashes, for the
local algorithms installed, next to its SHA-256.

**Quality.** PDQ gives each hash a quality from 0 to 100. Near-blank and flat
images score low, and their hashes match too much by chance, so PDQ hashes
below quality 50 aren't matched (`--min-quality 0` matches them anyway). They
are still printed, and listed as skipped. pHash has no quality score.

### How the distances were chosen

`python -m spade.bench.hash_robustness` hashes the 13 scikit-image benchmark
photos and edited copies of each, and prints the largest distance between a
photo and its edited copy, and the smallest distance between two different
photos:

| | `pdq` | `phash` |
|---|---:|---:|
| resized to 0.5x (largest) | 58 (brick; next is grass, 28) | 2 |
| resized to 2x (largest) | 10 | 2 |
| JPEG quality 70 (largest) | 2 | 2 |
| colour grade 0.15 (largest) | 6 | 2 |
| 5% cropped (largest) | 110 | 22 |
| two different photos (smallest) | 108 | 20 |
| distance used | 31 | 10 |

For `phash`, 10 sits well inside the gap: edited copies move 2 bits at most
and different photos are at least 20 apart. For `pdq`, 31 is the threshold
Meta publishes for PDQ, and we keep it. The one
miss is the brick photo halved in size (58 bits): it is an even, fine texture
with little large-scale structure, and PDQ's hash of such an image changes a
lot when it is shrunk, whatever resampling filter is used. Ordinary photos
stayed under 31. Some tools (python-threatexchange, HMA) hash the full image
without the 512 px downsample SPADE applies, so their hashes of large images can
differ from SPADE's by a few bits, well inside 31. The benchmark is small, so
treat these numbers as a sanity check rather than error rates; `--max-distance` changes the cutoff. The
`phash-lib` default copies `phash`'s; it hasn't been measured, because
libpHash isn't installed where these were run.

### What hashes don't survive

Perceptual hashes describe the whole picture, so they fail when the picture as
a whole changes: cropping, rotation by angles other than 90 degrees, and
large local edits such as pasting in or removing a big object. Even a 5% crop
pushed all 13 photos above past the PDQ threshold, and 2 of them past the pHash
one. Rotations by 90 degrees and mirror images match only with
`--thorough`. For those cases, and for finding where part of an image came
from, use `spade match` and `spade search`, which compare regions rather than
whole images.

### Licensing and privacy

- `pdqhash` (MIT, bundling Meta's BSD C++) and `imagehash` (BSD) are installed
  with the `hash` extra.
- libpHash is GPLv3. SPADE is Apache-2.0 and doesn't ship it or link to it at
  build time; `phash-lib` only works if you install libpHash yourself (for
  example `apt install libphash0`, or set `SPADE_PHASH_LIB` to the `.so`).
  SPADE's tests check `phash-lib` only against a mocked library, not the real
  libpHash, so check a few hashes against libpHash's own output before relying
  on it.
- PhotoDNA needs a licence from Microsoft and a PhotoDNA Cloud Service key in
  `SPADE_PHOTODNA_KEY`. Using it sends the image to Microsoft. SPADE never
  does that unless `photodna` is asked for by name, and the report never uses
  it. The PhotoDNA client is tested only against a mocked service.

## How it works

A diagram of the architecture and the workflow is in [docs/architecture.drawio](docs/architecture.drawio)
(two pages; open it in draw.io / diagrams.net).

1. Extract every patch (default 5x5) and drop flat ones. Targets are also
   indexed at 3/4 and 1/2 size, for copies that were shrunk.
2. Describe each patch by its normalized luminance pattern: grey levels minus
   their mean, scaled to unit length. Brightness and contrast drop out, and mild
   color grades barely move it. Luminance is used because JPEG keeps it at full
   resolution while it subsamples color.
3. Nearest-neighbour search (FAISS): exact for small collections, a compressed
   inverted-file index past 50,000 patches.
4. Per-patch pre-filter: fit grey-level gain and offset, chi-square test the
   residual, with the noise level read from the JPEG quantization tables when
   there are any. Pixels clipped at 0 or 255 are ignored, since a recolor that
   saturates breaks the model there.
5. Cluster the surviving matches by translation, per target image and scale,
   and (across images) split each cluster into spatially connected parts, so
   chance matches elsewhere at the same offset don't dilute a real copy.
6. For each cluster, fit one 3x3 color transform over all its pixels, keep the
   patches whose luminance agrees with it, and compute the region's NFA:
   how many regions this consistent you'd expect by chance, given how often
   patches matched at all and how many regions were tried (offsets, positions,
   region sizes, scales, passes).

`thorough` repeats this on the mirrored and shrunk questioned image, and runs a
second search that compares patches as rotation-invariant rings; all of it is
counted in the NFA.

Handles brightness, contrast, color grading (any affine color transform),
JPEG recompression and, with the pyramid, downscaled copies; mirrored, enlarged
and rotated copies with the options above. It does not handle non-uniform
geometric distortion.

A significant NFA means "not chance", which isn't the same as "copied":
content that is genuinely identical by design (the same logo, text glyphs,
identically drawn shapes) matches for real and will be reported. Look at the
region before concluding.

## Benchmark

`python -m spade.bench.runner` scores SPADE (and a RootSIFT + RANSAC baseline)
on one of several datasets:

| `--dataset` | What it is |
|---|---|
| `synthetic` (default) | procedural textures with recolored splices; exact ground truth |
| `photos` | splices between real photos bundled with scikit-image (real noise and JPEG history) |
| `casia2` | CASIA v2 cross-image splices with their real donor images; `python -m spade.bench.download casia2` fetches it from Kaggle (as do `coverage` and `comofod`) (needs a Kaggle API token in `KAGGLE_API_TOKEN`, `KAGGLE_ACCESS_TOKEN` or `~/.kaggle/access_token`, or `KAGGLE_USERNAME`/`KAGGLE_KEY`) |
| `coverage` | COVERAGE copy-moves: 100 forgeries whose originals hold similar-but-genuine objects (the originals are the negatives). Masks are where each forgery differs from its original, which marks the pasted copy but not its source, so only recall of the copy is reported and a region counts as found when either half lands on the copy |
| `comofod` | CoMoFoD copy-moves, the 120 base cases; masks mark both copies; `--category translation\|rotation\|scaling` |
| `manifest` | your own data: a CSV of `donor,tampered,mask,label` |

`--calibrate` checks the NFA's promise directly: for a range of thresholds it
reports false regions per image pair (should stay at or below the threshold)
next to recall. On 40 photo splices + 40 unrelated pairs at JPEG quality 75 it
reports no false regions at any threshold up to 10 and 92.5% recall from
NFA < 10^-4 on.

Synthetic run (96x96 images, 24x24 splice, color grade 0.15, 20 spliced + 20
clean pairs):

| | IoU | Precision | Recall | Detection AUC | s / image |
|---|---|---|---|---|---|
| SPADE, uncompressed | 0.85 | 0.95 | 0.89 | 0.96 | 0.6 |
| SPADE, JPEG q85 | 0.78 | 0.97 | 0.81 | 0.95 | 0.6 |
| RootSIFT + RANSAC, uncompressed | 0.00 | 0.70 | 0.00 | 0.75 | 0.1 |

The clean pairs contain hard-edged disks, and two disks of the same radius are
pixel-identical, so some "false" detections there are real matches (see the
note above). Synthetic data is a sanity check, not a substitute for a real forensic
dataset.

**CASIA v2** (real cross-image splices with their actual donor images; 40
spliced + 40 unrelated pairs, `--dataset casia2`):

| | IoU | Precision | Recall | Detection AUC | Found at NFA < 10^-3 | s / image |
|---|---|---|---|---|---|---|
| SPADE | 0.39 | 0.97 | 0.41 | 0.82 | 47.5% | 18 |
| RootSIFT + RANSAC (60 + 60 pairs) | 0.18 | 0.95 | 0.19 | 0.82 | - | 0.1 |

Real splices are much harder than the generated ones. SPADE finds about half
of them and stays precise: no alarm on any unrelated pair at any threshold,
and one region in the 80 pairs that landed beside a splice instead of on it.
Most of the misses are copies that were enlarged before pasting (about half
of CASIA's splices are scaled up 1.15-2.5x), nearly flat regions with no
texture to match, and very small splices. `query_scales=[1.0, 0.75, 0.5]`
covers enlarged copies; on the same pairs it raises the share found to 50%
(57.5% at NFA < 1) but adds misplaced regions on spliced images and takes
over 4x as long, so it stays opt-in.

**Copy-move** (`--task copymove` on photos, or the standard sets):

| Dataset | Found at NFA < 10^-3 | Found at NFA < 1 | Clean images with an alarm (NFA < 10^-3) |
|---|---|---|---|
| Photo crops, recolored copies (60 + 60) | 72% | 78% | 0% |
| CoMoFoD translation (25 + 25) | 40% | 44% | 0% |
| CoMoFoD rotation (25 + 25) | 20% | 24% | 8% |
| CoMoFoD scaling (25 + 25) | 16% | 16% | 8% |
| COVERAGE (100 + 100) | 31% | 33% | 3% |

Copy-move is the harder task. Rotated and rescaled copies within one image
aren't handled yet (rotation mode works across images only), and the clean
images in CoMoFoD and COVERAGE are full of objects that genuinely repeat
(identical printed logos, rows of windows, similar-but-genuine objects by
design in COVERAGE). Those are real duplicates as far as pixels go, and SPADE
can't yet tell them from a pasted copy; that, more than chance, is where the
alarms on clean images come from. Splitting a match cluster into connected
parts finds many more copies (CoMoFoD translation 40% -> 84%) but also flags
12-28% of the clean images, so it is used for cross-image search only.

## Project structure

```
src/spade/
    extraction/    # patch extraction, entropy, image pyramids, rotation rings
    descriptors/   # luminance-pattern descriptor
    search/        # FAISS index (exact or IVF-PQ, with real deletes)
    verification/  # affine color fit, chi-square test, coherence, NFA, FDR
    aggregation/   # heatmap, localization, match lists
    bench/         # datasets (synthetic, photos, CASIA v2, COVERAGE, CoMoFoD),
                   # calibration, baselines, downloader, hash robustness
    engine.py      # main API
    hashing.py     # perceptual hashes (PDQ, pHash, libpHash) and hash lists
    photodna.py    # PhotoDNA Cloud Service client
    jpeg.py        # noise level from JPEG quantization tables
    scale.py       # native-scale cue for the report
    provenance.py  # which images share copied regions
    visual.py      # HTML report
    report.py      # versioned JSON forensic report
    cli.py         # command line interface
    api.py         # REST API (FastAPI)
```

## Development

```bash
pip install -e ".[dev,api,bench]"
pytest tests/
ruff check src/
```

## License

Apache 2.0
