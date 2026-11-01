"""FastAPI REST API for SPADE forensics engine."""

import io
import base64
import threading
import uuid
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Query
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from spade import ForensicsEngine, Config, __version__
from spade.exceptions import DependencyError, TargetExistsError
from spade.jpeg import sigma_from_tables
from spade.report import region_offset

# Security limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels (prevent decompression bombs)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}

# Enforce the pixel cap at decode time too: the header-based size check can be
# spoofed, so make PIL itself raise DecompressionBombError during load/convert.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class MatchCoord(BaseModel):
    """Coordinate pair."""
    x: int
    y: int


class MatchInfo(BaseModel):
    """Information about a single match."""
    source_coord: MatchCoord
    target_coord: MatchCoord
    patch_size: int
    fit_p_value: float   # per-patch goodness of fit, not a match probability
    image_id: str


class CoherentRegionInfo(BaseModel):
    """Information about a coherent region."""
    offset: Optional[MatchCoord] = None   # null when mirrored or rescaled (see the bboxes)
    image_id: str
    num_matches: int
    confidence: float
    log10_nfa: Optional[float] = None   # < 0: expected less than once by chance
    significant: bool
    source_bbox: Dict[str, int]
    target_bbox: Dict[str, int]
    mirrored: bool = False
    source_scale: float = 1.0
    rotation_degrees: float = 0.0


class MatchResponse(BaseModel):
    """Response from match endpoint."""
    success: bool
    detected: bool
    best_match: Optional[MatchInfo] = None
    total_matches: int
    coherent_regions: List[CoherentRegionInfo] = []
    stats: Dict[str, Any]
    heatmap_base64: Optional[str] = None


class IndexResponse(BaseModel):
    """Response from index endpoint."""
    success: bool
    image_id: str
    patches_indexed: int
    message: str


async def validate_uploaded_image(file: UploadFile) -> Tuple[np.ndarray, float]:
    """
    Validate and load uploaded image with security checks.

    Args:
        file: Uploaded file

    Returns:
        (image as numpy array, JPEG rounding-error std from its quantization tables; 0 if not JPEG)

    Raises:
        HTTPException: If validation fails
    """
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Validate image format and load
    try:
        image = Image.open(io.BytesIO(contents))

        # Check format
        if image.format not in ALLOWED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {image.format}. Allowed: {ALLOWED_IMAGE_FORMATS}"
            )

        # Check dimensions (prevent decompression bombs)
        if image.size[0] * image.size[1] > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Maximum {MAX_IMAGE_PIXELS} pixels"
            )

        # Convert to RGB
        jpeg_sigma = (sigma_from_tables(getattr(image, "quantization", None)) or 0.0) if image.format == "JPEG" else 0.0
        image_rgb = image.convert("RGB")
        return np.array(image_rgb), jpeg_sigma

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def create_app(config: Optional[Config] = None) -> "FastAPI":
    """
    Create FastAPI application with SPADE engine.

    Args:
        config: Optional engine configuration

    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise DependencyError("FastAPI required: pip install spade-forensics[api]")

    app = FastAPI(
        title="SPADE Forensics API",
        description="Sub-Patch Analysis & Detection Engine - REST API for image forensics",
        version=__version__,
    )

    # Initialize engine
    engine = ForensicsEngine(config or Config())

    # Index, delete and match all touch the engine's target table, so they take
    # turns. The work itself runs in a worker thread to keep the event loop free.
    engine_lock = threading.Lock()

    def locked(fn, *args, **kwargs):
        with engine_lock:
            return fn(*args, **kwargs)

    # Track service start time for uptime calculation
    @app.on_event("startup")
    async def startup_event():
        """Record service start time."""
        import time
        app.state.start_time = time.time()

    @app.get("/")
    async def root():
        """API health check."""
        return {
            "name": "SPADE Forensics API",
            "version": __version__,
            "status": "healthy",
            "indexed_targets": engine.target_count,
        }

    @app.get("/health/live")
    async def liveness_check():
        """
        Liveness probe for Kubernetes/container orchestration.

        Returns 200 if service is running (even if not ready to serve).
        This should only fail if the process is completely stuck.
        """
        return {"status": "alive", "timestamp": __import__("time").time()}

    @app.get("/health/ready")
    async def readiness_check():
        """
        Readiness probe for Kubernetes/container orchestration.

        Returns 200 if service is ready to accept requests.
        Checks that the engine is initialized and index is available.
        """
        try:
            # Check if index is accessible
            index_size = engine.index.size if hasattr(engine, 'index') else 0
            target_count = engine.target_count

            # Service is ready if engine is initialized
            return {
                "status": "ready",
                "index_size": index_size,
                "target_count": target_count,
                "timestamp": __import__("time").time()
            }
        except Exception as e:
            # If we can't check status, service is not ready
            raise HTTPException(
                status_code=503,
                detail=f"Service not ready: {str(e)}"
            )

    @app.get("/health")
    async def health_check():
        """
        Comprehensive health check endpoint.

        Returns detailed health information including:
        - Service status
        - Index statistics
        - Memory usage (if available)
        - Configuration
        """
        import time

        health_info = {
            "status": "healthy",
            "version": __version__,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, "start_time") else None,
            "index": {
                "size": engine.index.size if hasattr(engine, 'index') else 0,
                "target_count": engine.target_count,
                "target_ids": engine.target_ids[:10] if engine.target_count > 0 else [],
            },
            "config": {
                "patch_size": engine.config.patch_size,
                "pyramid_enabled": engine.config.pyramid_enabled,
                "coherence_enabled": engine.config.coherence_enabled,
            }
        }

        # Add memory info if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            health_info["memory"] = {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
            }
        except ImportError:
            health_info["memory"] = "psutil not available"

        return health_info

    @app.get("/config")
    async def get_config():
        """Get current engine configuration."""
        return {
            "patch_size": engine.config.patch_size,
            "stride": engine.config.stride,
            "entropy_threshold": engine.config.entropy_threshold,
            "k_neighbors": engine.config.k_neighbors,
            "distance_threshold": engine.config.distance_threshold,
            "min_probability": engine.config.min_probability,
            "max_nfa": engine.config.max_nfa,
            "pyramid_enabled": engine.config.pyramid_enabled,
            "coherence_enabled": engine.config.coherence_enabled,
        }

    @app.post("/index", response_model=IndexResponse)
    async def index_image(
        file: UploadFile = File(...),
        image_id: Optional[str] = Query(None, description="Unique image identifier", max_length=255),
    ):
        """
        Index a target image for later matching.

        Upload an image file to add it to the search index.
        Maximum file size: 50MB. Supported formats: JPEG, PNG, BMP, TIFF, WEBP.
        """
        # Validate image
        image_array, jpeg_sigma = await validate_uploaded_image(file)

        # Sanitize and validate image_id
        if image_id is not None:
            # Remove potentially dangerous characters
            image_id = "".join(c for c in image_id if c.isalnum() or c in "._-")
            if not image_id:
                raise HTTPException(status_code=400, detail="Invalid image_id after sanitization")
        else:
            # Random rather than counter-based: a count reuses ids after deletes.
            image_id = f"img_{uuid.uuid4().hex[:12]}"

        try:
            count = await run_in_threadpool(
                locked, engine.index_target, image_array, image_id, jpeg_sigma=jpeg_sigma
            )
        except TargetExistsError:
            raise HTTPException(
                status_code=409,
                detail=f"Image ID '{image_id}' already exists. Use a different ID or delete the existing one."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

        return IndexResponse(
            success=True,
            image_id=image_id,
            patches_indexed=count,
            message=f"Indexed {count} patches from {file.filename}",
        )

    @app.post("/match", response_model=MatchResponse)
    async def match_image(
        file: UploadFile = File(...),
        target_ids: Optional[str] = Query(None, description="Comma-separated target IDs", max_length=1000),
        return_heatmap: bool = Query(False, description="Include base64 heatmap"),
    ):
        """
        Match a source image against indexed targets.

        Upload an image to search for it in the indexed collection.
        Maximum file size: 50MB. Supported formats: JPEG, PNG, BMP, TIFF, WEBP.
        """
        # Validate image
        image_array, jpeg_sigma = await validate_uploaded_image(file)

        try:

            # Parse target IDs if provided
            target_id_list = None
            if target_ids:
                target_id_list = [t.strip() for t in target_ids.split(",")]

            result = await run_in_threadpool(
                locked,
                engine.match,
                image_array,
                jpeg_sigma=jpeg_sigma,
                target_ids=target_id_list,
                return_heatmap=return_heatmap,
            )

            # Build response
            best_match_info = None
            if result.best_match:
                best_match_info = MatchInfo(
                    source_coord=MatchCoord(
                        x=result.best_match.source_coord[0],
                        y=result.best_match.source_coord[1],
                    ),
                    target_coord=MatchCoord(
                        x=result.best_match.target_coord[0],
                        y=result.best_match.target_coord[1],
                    ),
                    patch_size=result.best_match.patch_size,
                    fit_p_value=result.best_match.probability,
                    image_id=result.best_match.image_id,
                )

            # Build coherent regions info
            regions_info = []
            for region in result.coherent_regions:
                offset = region_offset(region)
                regions_info.append(CoherentRegionInfo(
                    offset=None if offset is None else MatchCoord(x=offset[0], y=offset[1]),
                    mirrored=region.mirrored,
                    source_scale=region.source_scale,
                    rotation_degrees=round(float(np.degrees(region.rotation)), 2),
                    image_id=region.image_id,
                    num_matches=len(region.matches),
                    confidence=region.confidence,
                    log10_nfa=region.log10_nfa,
                    significant=region.significant,
                    source_bbox={
                        "x": region.source_bbox[0],
                        "y": region.source_bbox[1],
                        "width": region.source_bbox[2],
                        "height": region.source_bbox[3],
                    },
                    target_bbox={
                        "x": region.target_bbox[0],
                        "y": region.target_bbox[1],
                        "width": region.target_bbox[2],
                        "height": region.target_bbox[3],
                    },
                ))

            # Encode heatmap if requested
            heatmap_b64 = None
            if return_heatmap and result.heatmap is not None:
                heatmap_img = Image.fromarray(
                    (result.heatmap * 255).astype(np.uint8)
                )
                buffer = io.BytesIO()
                heatmap_img.save(buffer, format="PNG")
                heatmap_b64 = base64.b64encode(buffer.getvalue()).decode()

            return MatchResponse(
                success=True,
                detected=bool(result.significant_regions),
                best_match=best_match_info,
                total_matches=len(result.matches),
                coherent_regions=regions_info,
                stats=result.stats,
                heatmap_base64=heatmap_b64,
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Match failed: {str(e)}")

    @app.get("/targets")
    async def list_targets():
        """List all indexed target images."""
        targets = []
        for image_id in engine.target_ids:
            shape = engine.get_target_shape(image_id)
            targets.append({
                "image_id": image_id,
                "shape": {"height": shape[0], "width": shape[1]} if shape else None,
                "path": engine.get_target_path(image_id),
            })
        return {"targets": targets, "count": len(targets)}

    @app.delete("/targets/{image_id}")
    async def delete_target(image_id: str):
        """
        Remove a target and its descriptors from the index.
        """
        if not await run_in_threadpool(locked, engine.remove_target, image_id):
            raise HTTPException(status_code=404, detail=f"Target {image_id} not found")
        return {"success": True, "message": f"Removed target {image_id}"}

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config: Optional[Config] = None,
):
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        config: Engine configuration
    """
    try:
        import uvicorn
    except ImportError:
        raise DependencyError("uvicorn required: pip install spade-forensics[api]")

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


# Create default app instance for uvicorn
app = create_app() if FASTAPI_AVAILABLE else None
