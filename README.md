# SPADE

Sub-Patch Analysis & Detection Engine - finds tiny image fragments (3×3 to 6×6 pixels) in large image collections, even after color shifts, compression, or editing.

## Install

```bash
pip install -e .
# or when published:
pip install spade-forensics

# With optional features:
pip install spade-forensics[api]    # REST API
pip install spade-forensics[lsh]    # LSH pre-filtering
pip install spade-forensics[full]   # Everything
```

Requires: numpy, scipy, pillow, scikit-learn, faiss-cpu, click

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start the API server
docker-compose up

# Index an image
curl -X POST http://localhost:8000/index \
  -F "file=@target.jpg" \
  -F "image_id=target_001"

# Match a source image
curl -X POST http://localhost:8000/match \
  -F "file=@source.jpg" \
  -F "return_heatmap=true"
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

# With options
spade match source.jpg target.jpg -t 0.7 -p 3 -o heatmap.png -v

# Index a directory
spade index images/ -o myindex.spade

# Search against saved index
spade search query.jpg myindex.spade -k 5
```

### Python API

#### Basic Usage

```python
from spade import ForensicsEngine, Config
from spade.engine import load_image

# Create engine with default config
engine = ForensicsEngine()

# Index target images
target1 = load_image("suspect_photo.jpg")
engine.index_target(target1, "suspect_001")

target2 = load_image("crime_scene.jpg")
engine.index_target(target2, "scene_001")

# Match a source image
source = load_image("fragment.jpg")
result = engine.match(source)

if result.best_match:
    match = result.best_match
    print(f"Found match in: {match.image_id}")
    print(f"Probability: {match.probability:.3f}")
    print(f"Source location: {match.source_coord}")
    print(f"Target location: {match.target_coord}")
    print(f"Total matches: {len(result.matches)}")
```

#### Custom Configuration

```python
from spade import ForensicsEngine, Config

config = Config(
    patch_size=4,                    # Larger patches (3-6)
    stride=2,                        # Skip patches for speed
    entropy_threshold=3.0,           # More aggressive filtering
    descriptor_dim=256,              # Feature dimensions
    min_probability=0.7,             # Higher confidence threshold
    pyramid_enabled=True,            # Multi-scale matching
    coherence_enabled=True,          # Spatial coherence verification
    max_targets=100,                 # LRU cache limit
)

engine = ForensicsEngine(config)
```

#### Multi-Scale Matching

```python
from spade import ForensicsEngine, Config

# Enable multi-scale pyramid for scale-invariant matching
config = Config(
    pyramid_enabled=True,
    pyramid_scales=[1.0, 0.5, 0.25, 0.125],  # 1×, 0.5×, 0.25×, 0.125×
    pyramid_min_size=16,                      # Min dimension at smallest scale
)

engine = ForensicsEngine(config)

# Now matches work even if source is scaled differently than target
source_scaled = load_image("fragment_50percent.jpg")
result = engine.match(source_scaled)
```

#### Batch Processing

```python
from spade import ForensicsEngine
from pathlib import Path

engine = ForensicsEngine()

# Index all targets in a directory
target_dir = Path("targets")
for img_path in target_dir.glob("*.jpg"):
    image = load_image(str(img_path))
    engine.index_target(image, img_path.stem)
    print(f"Indexed: {img_path.name}")

# Match multiple sources
source_dir = Path("sources")
results = []

for img_path in source_dir.glob("*.jpg"):
    image = load_image(str(img_path))
    result = engine.match(image)

    if result.best_match:
        results.append({
            "source": img_path.name,
            "target": result.best_match.image_id,
            "probability": result.best_match.probability,
            "coherent_regions": len(result.coherent_regions),
        })

# Print summary
for r in sorted(results, key=lambda x: x["probability"], reverse=True):
    print(f"{r['source']} → {r['target']}: {r['probability']:.3f}")
```

#### Heatmap Visualization

```python
from spade import ForensicsEngine
from spade.engine import load_image
import matplotlib.pyplot as plt

engine = ForensicsEngine()

# Index target
target = load_image("target.jpg")
engine.index_target(target, "target_001")

# Match with heatmap
source = load_image("source.jpg")
result = engine.match(source, return_heatmap=True)

if result.heatmap is not None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(source)
    plt.title("Source Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(result.heatmap, cmap='hot')
    plt.title("Match Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(source)
    plt.imshow(result.heatmap, cmap='hot', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("match_visualization.png", dpi=300, bbox_inches='tight')
```

#### Save and Load Index

```python
from spade import ForensicsEngine

# Index targets
engine = ForensicsEngine()
for i, target in enumerate(target_images):
    engine.index_target(target, f"target_{i:04d}")

# Save index to disk
engine.save_index("myindex")
# Creates: myindex.index (FAISS binary) and myindex.meta.json (metadata)

# Later: load the index
engine2 = ForensicsEngine()
engine2.load_index("myindex")

# Now ready to match
result = engine2.match(source_image)
```

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
    response = requests.post(
        f"{BASE_URL}/index",
        files={"file": f},
        data={"image_id": "target_001"}
    )
    print(response.json())
    # {"success": true, "patches_indexed": 456789, ...}

# Match a source
with open("source.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/match",
        files={"file": f},
        data={"return_heatmap": "true"}
    )
    result = response.json()

    if result["best_match"]:
        match = result["best_match"]
        print(f"Match found: {match['probability']:.3f}")
        print(f"Target: {match['image_id']}")
        print(f"Location: ({match['target_coord']['x']}, {match['target_coord']['y']})")
```

### Environment Variables

Configure SPADE via environment variables:

```bash
# Core settings
export SPADE_PATCH_SIZE=3           # Patch size (3-6)
export SPADE_STRIDE=1               # Patch stride
export SPADE_DESCRIPTOR_DIM=256     # Feature dimension
export SPADE_MIN_PROBABILITY=0.5    # Match threshold

# Filtering
export SPADE_ENTROPY_THRESHOLD=2.5  # Texture filter (higher = more aggressive)

# Multi-scale
export SPADE_PYRAMID_ENABLED=true   # Enable scale invariance
export SPADE_PYRAMID_MIN_SIZE=16    # Min image dimension at smallest scale

# Performance
export SPADE_MAX_TARGETS=1000       # LRU cache limit (prevents memory exhaustion)
export SPADE_K_NEIGHBORS=100        # ANN search results

# Advanced
export SPADE_LSH_ENABLED=true       # Enable LSH pre-filtering
export SPADE_USE_GPU=true           # GPU acceleration (requires CuPy)
export SPADE_DISTRIBUTED=true       # Distributed sharding
export SPADE_NUM_SHARDS=4           # Number of shards
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

### Advanced Features

#### Multi-Scale Pyramid (Scale Invariance)

```python
from spade import ForensicsEngine, Config

config = Config(
    pyramid_enabled=True,
    pyramid_scales=[1.0, 0.5, 0.25, 0.125],
    pyramid_min_size=16,
)

engine = ForensicsEngine(config)
# Now detects fragments even if scaled
```

#### LSH Pre-Filtering (Faster Search)

```python
# Requires: pip install datasketch
config = Config(
    lsh_enabled=True,
    lsh_threshold=0.3,          # Jaccard similarity threshold
    lsh_candidates=1000,        # Pre-filter to top 1000 candidates
)

engine = ForensicsEngine(config)
# 5-10× speedup on large indexes (>1M vectors)
```

#### GPU Acceleration

```python
# Requires: pip install cupy-cuda11x (match your CUDA version)
config = Config(
    use_gpu=True,
)

engine = ForensicsEngine(config)
# 2-5× speedup for patch extraction and search
```

#### Distributed Sharding (Scale-Out)

```python
from spade.search.distributed import ShardedIndex

# Local mode: parallel search across shards
index = ShardedIndex(num_shards=4, mode="local")

# Remote mode: search across multiple servers
index = ShardedIndex(num_shards=4, mode="remote")
index.configure_remote([
    "http://shard0:8000",
    "http://shard1:8000",
    "http://shard2:8000",
    "http://shard3:8000",
])

# Use like normal index
engine.index = index
```

## How it works

1. Extract small patches from images
2. Filter out uniform regions (sky, walls) using entropy
3. Compute photometric-invariant descriptors
4. Search with FAISS
5. Verify matches with affine color model (Huber loss for robustness)
6. Score with chi-square probability

The system handles:
- Brightness/contrast changes
- Color shifts
- Gamma adjustments
- JPEG artifacts

It does NOT handle rotation, blur, or geometric distortion.

## Project structure

```
src/spade/
    extraction/    # patch extraction, entropy filtering, image pyramids
    descriptors/   # difference vectors, chromaticity, LBP, rank-order, gradient
    search/        # FAISS index wrapper, LSH pre-filtering
    verification/  # affine solver, probability model, spatial coherence
    aggregation/   # heatmap generation
    engine.py      # main API
    cli.py         # command line interface
    api.py         # REST API (FastAPI)
```

## Configuration

Key parameters in `Config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| patch_size | 3 | Patch size (3, 4, 5, or 6) |
| entropy_threshold | 2.5 | Filter low-texture patches (None to disable) |
| min_probability | 0.5 | Match probability threshold |
| noise_sigma | 0.02 | Expected image noise level |
| auto_train_pca | True | Auto-train PCA on indexed patches |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

## Roadmap

**Done (Phase 1)**
- 3×3 to 6×6 patches with spatial pooling
- Entropy filtering (adaptive bins)
- Huber loss verification (robust to JPEG)
- Auto PCA training
- CLI + Python API
- Spatial coherence verification

**Done (Phase 2)**
- 256-dim descriptors
- Image pyramids for scale invariance
- LSH pre-filtering (optional, requires datasketch)
- REST API (FastAPI)

**Future (Phase 3)**
- GPU acceleration (CuPy, FAISS-GPU)
- Learned descriptors (PyTorch)
- Distributed search

## License

Apache 2.0
