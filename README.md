# SPADE

Sub-Patch Analysis & Detection Engine - finds where a small region of an image was copied from, across a collection of images, even after recoloring and JPEG recompression. Every reported region comes with a significance score (an a-contrario NFA) you can put in a report.

## Install

```bash
pip install -e .
# or when published:
pip install spade-forensics

# With optional features:
pip install spade-forensics[api]    # REST API
pip install spade-forensics[bench]  # Benchmark harness (OpenCV baseline)
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

# Index a directory (writes myindex.index, myindex.meta.json, myindex.engine.json)
spade index images/ -o myindex

# Search against the saved index; patches are reloaded from the indexed files
spade search query.jpg myindex -k 5

# Scan many questioned images, holding the false discovery rate at 5%
spade scan myindex questioned/*.jpg --fdr 0.05
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
    pyramid_enabled=True,    # also index targets at 1/2, 1/4, 1/8 scale
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
offset its patches support - a region-level ratio test. On real-photo
copy-moves at JPEG 90 that keeps false regions at about 0.02 per image, at the
cost of finding about half of the copies; content that is pixel-identical by
construction (an exact tiling) is still reported, because it is a duplicate.

#### JPEG-Aware Noise

A JPEG file's quantization tables say how much rounding error to expect. The
CLI and API read them automatically; from Python pass them yourself:

```python
from spade.jpeg import quantization_sigma

engine.index_target(load_image(path), path, image_path=path)   # read from the file
result = engine.match(image, jpeg_sigma=quantization_sigma("questioned.jpg") or 0.0)
```

#### Rotated Copies (experimental)

`Config(rotation_invariant=True)` describes every patch in its own orientation
and groups matches by rotation + translation. It finds rotated copies and
recovers the angle, but is much slower and on our photo benchmark catches
about half of the copies rotated by arbitrary angles, so it's off by default.

#### Multi-Scale Matching

Targets can be indexed at several scales so a copy that was shrunk before
pasting is still found. Matches record the level they came from in
`target_scale`.

```python
config = Config(pyramid_enabled=True, pyramid_scales=[1.0, 0.5, 0.25])
engine = ForensicsEngine(config)
```

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
export SPADE_PYRAMID_ENABLED=true
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
export SPADE_PYRAMID_ENABLED=true     # Index targets at several scales
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

## How it works

1. Extract every patch (default 5x5) and drop flat ones.
2. Describe each patch by its normalized luminance pattern: grey levels minus
   their mean, scaled to unit length. Brightness and contrast drop out, and mild
   color grades barely move it. Luminance is used because JPEG keeps it at full
   resolution while it subsamples color.
3. Exact nearest-neighbour search (FAISS), batched.
4. Per-patch pre-filter: fit grey-level gain and offset, chi-square test the
   residual. Pixels clipped at 0 or 255 are ignored, since a recolor that
   saturates breaks the model there.
5. Cluster the surviving matches by translation, per target image and scale.
6. For each cluster, fit one 3x3 color transform over all its pixels, keep the
   patches whose luminance agrees with it, and compute the region's NFA:
   how many regions this consistent you'd expect by chance, given how often
   patches matched at all.

Handles brightness, contrast, color grading (any affine color transform),
JPEG recompression and, with the pyramid, downscaled copies. It does not handle
rotation or non-uniform geometric distortion.

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
| `casia2` | CASIA v2 cross-image splices with their real donor images; `python -m spade.bench.download casia2` fetches it from Kaggle (needs `KAGGLE_USERNAME`/`KAGGLE_KEY`) |
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

## Project structure

```
src/spade/
    extraction/    # patch extraction, entropy, image pyramids
    descriptors/   # luminance-pattern descriptor
    search/        # FAISS index (exact, with real deletes)
    verification/  # affine color fit, chi-square test, coherence, NFA
    aggregation/   # heatmap, localization
    bench/         # synthetic benchmark
    engine.py      # main API
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
