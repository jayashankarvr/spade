# SPADE REST API Documentation

> FastAPI-based REST interface for the SPADE forensics engine

**Base URL**: `http://localhost:8000`
**API Version**: 0.2.0-alpha

---

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Checks](#health-checks)
  - [Configuration](#configuration)
  - [Index Management](#index-management)
  - [Matching](#matching)
  - [Target Management](#target-management)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

---

## Quick Start

### Start the Server

```bash
# Using python module
python -m spade serve --host 0.0.0.0 --port 8000

# Using Docker
docker-compose up

# Using Makefile
make serve
```

### Basic Usage

```bash
# Check API health
curl http://localhost:8000/health

# Index a target image
curl -X POST "http://localhost:8000/index?image_id=target_001" \
  -F "file=@target.jpg"

# Match a source image
curl -X POST "http://localhost:8000/match?return_heatmap=true" \
  -F "file=@source.jpg"
```

---

## Authentication

**Current Version**: No authentication required

**Production Deployment**: Add authentication via:
- Nginx reverse proxy with Basic Auth
- API Gateway (AWS API Gateway, Kong, etc.)
- Custom FastAPI middleware

---

## Endpoints

### Health Checks

#### `GET /`

Root endpoint - basic health check.

**Response:**
```json
{
  "name": "SPADE Forensics API",
  "version": "0.2.0",
  "status": "healthy",
  "indexed_targets": 42
}
```

**Status Codes:**
- `200 OK`: Service is running

---

#### `GET /health/live`

Kubernetes liveness probe - checks if service is alive.

**Response:**
```json
{
  "status": "alive",
  "timestamp": 1704652800.0
}
```

**Status Codes:**
- `200 OK`: Process is running
- `503 Service Unavailable`: Process is hung/crashed

**Use Case**: Kubernetes will restart pod if this fails

---

#### `GET /health/ready`

Kubernetes readiness probe - checks if service is ready to accept requests.

**Response:**
```json
{
  "status": "ready",
  "index_size": 1500000,
  "target_count": 42,
  "timestamp": 1704652800.0
}
```

**Status Codes:**
- `200 OK`: Service is ready
- `503 Service Unavailable`: Service is initializing or unhealthy

**Use Case**: Kubernetes will not send traffic to pod if this fails

---

#### `GET /health`

Comprehensive health check with detailed metrics.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "timestamp": 1704652800.0,
  "uptime_seconds": 3600.5,
  "index": {
    "size": 1500000,
    "target_count": 42,
    "target_ids": ["img_3f9c2a1b7d04", "..."]
  },
  "config": {
    "patch_size": 5,
    "pyramid_enabled": true,
    "coherence_enabled": true
  },
  "memory": {
    "rss_mb": 2048.5,
    "vms_mb": 4096.2
  }
}
```

**Status Codes:**
- `200 OK`: Service is healthy

**Note**: Memory stats require `psutil` package

---

### Configuration

#### `GET /config`

Get current engine configuration.

**Response:**
```json
{
  "patch_size": 5,
  "stride": 1,
  "entropy_threshold": null,
  "k_neighbors": 8,
  "distance_threshold": 0.5,
  "min_probability": 0.01,
  "max_nfa": 1.0,
  "pyramid_enabled": true,
  "coherence_enabled": true
}
```

**Status Codes:**
- `200 OK`: Configuration returned

---

### Index Management

#### `POST /index`

Index a target image for later matching.

**Request:**
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required, form field): Image file (JPEG, PNG, BMP, TIFF, WEBP)
  - `image_id` (optional, query parameter): Unique identifier (alphanumeric + `._-`).
    If omitted a random `img_<hex>` id is generated.

**Limits:**
- Max file size: 50MB
- Max image pixels: 50 megapixels
- `image_id` max length: 255 characters

**Example:**
```bash
curl -X POST "http://localhost:8000/index?image_id=suspect_photo_001" \
  -F "file=@target.jpg"
```

**Response:**
```json
{
  "success": true,
  "image_id": "suspect_photo_001",
  "patches_indexed": 456789,
  "message": "Indexed 456789 patches from target.jpg"
}
```

**Status Codes:**
- `200 OK`: Image indexed successfully
- `400 Bad Request`: Invalid image or parameters
- `409 Conflict`: Image ID already exists
- `413 Payload Too Large`: File exceeds limits
- `500 Internal Server Error`: Indexing failed

**Error Example:**
```json
{
  "detail": "Image ID 'suspect_photo_001' already exists. Use a different ID or delete the existing one."
}
```

---

#### `GET /targets`

List all indexed target images.

**Response:**
```json
{
  "targets": [
    {
      "image_id": "suspect_photo_001",
      "shape": {"height": 1024, "width": 768},
      "path": "/path/to/file.jpg"
    },
    {
      "image_id": "suspect_photo_002",
      "shape": {"height": 2048, "width": 1536},
      "path": null
    }
  ],
  "count": 2
}
```

**Status Codes:**
- `200 OK`: Targets listed

---

#### `DELETE /targets/{image_id}`

Remove a target and its descriptors from the index.

**Parameters:**
- `image_id` (path): Target identifier

**Example:**
```bash
curl -X DELETE http://localhost:8000/targets/suspect_photo_001
```

**Response:**
```json
{
  "success": true,
  "message": "Removed target suspect_photo_001"
}
```

**Status Codes:**
- `200 OK`: Target removed
- `404 Not Found`: Target does not exist

---

### Matching

#### `POST /match`

Match a source image against indexed targets.

**Request:**
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required, form field): Source image file
  - `target_ids` (optional, query parameter): Comma-separated target IDs to search
  - `return_heatmap` (optional, query parameter): Include base64-encoded heatmap (default: false)

**Example:**
```bash
curl -X POST "http://localhost:8000/match?target_ids=suspect_photo_001,suspect_photo_002&return_heatmap=true" \
  -F "file=@fragment.jpg"
```

**Response:**
```json
{
  "success": true,
  "detected": true,
  "best_match": {
    "source_coord": {"x": 100, "y": 200},
    "target_coord": {"x": 450, "y": 680},
    "patch_size": 5,
    "fit_p_value": 0.83,
    "image_id": "suspect_photo_001"
  },
  "total_matches": 2458,
  "coherent_regions": [
    {
      "offset": {"x": 350, "y": 480},
      "image_id": "suspect_photo_001",
      "num_matches": 156,
      "confidence": 0.92,
      "log10_nfa": -14.2,
      "significant": true,
      "source_bbox": {"x": 0, "y": 0, "width": 200, "height": 300},
      "target_bbox": {"x": 350, "y": 480, "width": 200, "height": 300}
    }
  ],
  "stats": {
    "source_patches": 123456,
    "total_matches": 2458,
    "coherent_regions": 1,
    "significant_regions": 1
  },
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response Fields:**
- `detected`: true if any region is significant (NFA < `max_nfa`)
- `best_match`: Best patch match of the most significant region (null if no matches)
  - `source_coord`: (x, y) in source image
  - `target_coord`: (x, y) in target image
  - `patch_size`: Size of matched patch
  - `fit_p_value`: Per-patch chi-square goodness of fit. Not a match
    probability: small patches fit each other by chance all the time
  - `image_id`: Target image identifier
- `total_matches`: Number of per-patch matches that passed the pre-filter
- `coherent_regions`: Clusters of matches sharing one offset, most significant first
  - `offset`: (dx, dy) from source to target
  - `image_id`: Target image the region was copied from
  - `num_matches`: Patches consistent with the region's color transform
  - `log10_nfa`: log10 of the expected number of regions this good with no
    copy present. Negative = less than one expected by chance
  - `significant`: `log10_nfa` below `log10(max_nfa)`
  - `confidence`: Ranking heuristic (mean fit p-value x log2 size)
  - `source_bbox`/`target_bbox`: Bounding boxes (x, y, width, height)
- `stats`: Search statistics
- `heatmap_base64`: Base64-encoded PNG heatmap (if requested)

**Status Codes:**
- `200 OK`: Matching completed
- `400 Bad Request`: Invalid image or parameters
- `413 Payload Too Large`: File exceeds limits

**Heatmap Decoding:**
```python
import base64
from PIL import Image
import io

heatmap_bytes = base64.b64decode(response['heatmap_base64'])
heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
heatmap_image.save('heatmap.png')
```

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Status Code | Meaning | Example |
|-------------|---------|---------|
| `400 Bad Request` | Invalid input | Unsupported image format, invalid parameters |
| `404 Not Found` | Resource not found | Target ID does not exist |
| `409 Conflict` | Resource conflict | Duplicate image ID |
| `413 Payload Too Large` | File too large | Image exceeds 50MB or 50MP |
| `500 Internal Server Error` | Server error | Indexing or matching failed |
| `503 Service Unavailable` | Service not ready | Initializing or unhealthy |

### Error Examples

**Invalid Image Format:**
```json
{
  "detail": "Unsupported format: GIF. Allowed: {'JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP'}"
}
```

**File Too Large:**
```json
{
  "detail": "File too large. Maximum size is 50MB"
}
```

**Decompression Bomb:**
```json
{
  "detail": "Image too large. Maximum 50000000 pixels"
}
```

---

## Rate Limiting

**Current Version**: No rate limiting by default

**Production Deployment**: Add rate limiting via Nginx:

```nginx
http {
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        location / {
            limit_req zone=api_limit burst=10 nodelay;
            proxy_pass http://spade_backend;
        }
    }
}
```

This limits each IP to 10 requests/second with burst up to 20.

---

## Examples

### Python Client

```python
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"

# Index a target
with open("target.jpg", "rb") as f:
    response = requests.post(
        f"{API_BASE}/index",
        params={"image_id": "target_001"},
        files={"file": f},
    )
    print(response.json())

# Match a source
with open("source.jpg", "rb") as f:
    response = requests.post(
        f"{API_BASE}/match",
        params={"target_ids": "target_001", "return_heatmap": "true"},
        files={"file": f},
    )
    result = response.json()

    for region in result["coherent_regions"]:
        if region["significant"]:
            print(f"Copied from {region['image_id']}")
            print(f"  source bbox: {region['source_bbox']}")
            print(f"  target bbox: {region['target_bbox']}")
            print(f"  log10 NFA: {region['log10_nfa']:.1f}")

    # Save heatmap
    if result["heatmap_base64"]:
        import base64
        from PIL import Image
        import io

        heatmap_bytes = base64.b64decode(result["heatmap_base64"])
        heatmap = Image.open(io.BytesIO(heatmap_bytes))
        heatmap.save("heatmap.png")

# List all targets
response = requests.get(f"{API_BASE}/targets")
targets = response.json()
print(f"Indexed targets: {targets['count']}")

# Delete a target
response = requests.delete(f"{API_BASE}/targets/target_001")
print(response.json())
```

### cURL Examples

**Batch Index Multiple Targets:**
```bash
for img in targets/*.jpg; do
  id=$(basename "$img" .jpg)
  curl -X POST "http://localhost:8000/index?image_id=$id" \
    -F "file=@$img"
done
```

**Match with Specific Targets:**
```bash
curl -X POST "http://localhost:8000/match?target_ids=img_001,img_002,img_003" \
  -F "file=@source.jpg"
```

**Health Check Loop:**
```bash
while true; do
  curl -s http://localhost:8000/health/ready | jq '.status'
  sleep 5
done
```

### JavaScript (Fetch API)

```javascript
// Index a target
async function indexImage(file, imageId) {
  const formData = new FormData();
  formData.append('file', file);

  const url = `http://localhost:8000/index?image_id=${encodeURIComponent(imageId)}`;
  const response = await fetch(url, {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

// Match a source
async function matchImage(file, targetIds = null, returnHeatmap = false) {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams({ return_heatmap: returnHeatmap.toString() });
  if (targetIds) {
    params.set('target_ids', targetIds.join(','));
  }

  const response = await fetch(`http://localhost:8000/match?${params}`, {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

// Usage
const fileInput = document.getElementById('file-input');
const file = fileInput.files[0];

const result = await matchImage(file, ['target_001'], true);
console.log('Detected:', result.detected, result.coherent_regions);

// Display heatmap
if (result.heatmap_base64) {
  const img = document.createElement('img');
  img.src = `data:image/png;base64,${result.heatmap_base64}`;
  document.body.appendChild(img);
}
```

---

## Performance Tips

### Optimize Index Size

1. **Increase stride**: Fewer patches, faster indexing
   ```bash
   export SPADE_STRIDE=2  # 4x fewer patches
   ```

2. **Set an entropy threshold**: drop weakly textured patches (bits; must be
   below log2(8) = 3 for the 8-bin histogram, or nothing passes)
   ```bash
   export SPADE_ENTROPY_THRESHOLD=2.0
   ```

### Concurrent Requests

Index, match and delete run in a worker thread, so the server stays responsive
(health checks keep answering). They take turns on the engine, so throughput
doesn't grow with client concurrency; run more replicas for that. Client-side
concurrency still works:

```python
import concurrent.futures
import requests

files = ["source1.jpg", "source2.jpg", "source3.jpg"]

def match_image(filename):
    with open(filename, "rb") as f:
        response = requests.post(
            "http://localhost:8000/match",
            files={"file": f}
        )
        return response.json()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(match_image, files))
```

### Memory Management

Set maximum target limit to prevent memory exhaustion:

```bash
export SPADE_MAX_TARGETS=100
```

When the limit is reached, the least recently indexed targets are evicted and
their descriptors removed from the index.

---

## Deployment Checklist

- [ ] Set appropriate `SPADE_MAX_TARGETS` for your memory budget
- [ ] Configure health check endpoints in orchestrator
- [ ] Add authentication (Nginx Basic Auth, API Gateway, etc.)
- [ ] Enable rate limiting in reverse proxy
- [ ] Set up monitoring (Prometheus, DataDog, etc.)
- [ ] Configure persistent storage for indexes
- [ ] Test backup/restore procedures
- [ ] Document emergency runbook

---

## OpenAPI Documentation

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

**Version**: 0.2.0-alpha
**Last Updated**: 2026-01-08
**Maintainer**: SPADE Team
