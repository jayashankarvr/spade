"""FastAPI REST API for SPADE forensics engine."""

import io
import base64
import threading
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from spade import ForensicsEngine, Config, __version__
from spade.exceptions import DependencyError

# Security limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels (prevent decompression bombs)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


class MatchCoord(BaseModel):
    """Coordinate pair."""
    x: int
    y: int


class MatchInfo(BaseModel):
    """Information about a single match."""
    source_coord: MatchCoord
    target_coord: MatchCoord
    patch_size: int
    probability: float
    image_id: str


class CoherentRegionInfo(BaseModel):
    """Information about a coherent region."""
    offset: MatchCoord
    num_matches: int
    confidence: float
    source_bbox: Dict[str, int]
    target_bbox: Dict[str, int]


class MatchResponse(BaseModel):
    """Response from match endpoint."""
    success: bool
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


class ConfigModel(BaseModel):
    """API configuration model."""
    patch_size: int = Field(default=3, ge=3, le=6)
    stride: int = Field(default=1, ge=1)
    entropy_threshold: Optional[float] = Field(default=2.5, ge=0)
    descriptor_dim: int = Field(default=256, ge=64, le=512)
    min_probability: float = Field(default=0.5, ge=0, le=1)
    pyramid_enabled: bool = Field(default=False)
    lsh_enabled: bool = Field(default=False)
    coherence_enabled: bool = Field(default=True)


async def validate_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Validate and load uploaded image with security checks.

    Args:
        file: Uploaded file

    Returns:
        Image as numpy array

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
        image_rgb = image.convert("RGB")
        return np.array(image_rgb)

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

    # Lock for thread-safe index operations
    index_lock = threading.Lock()

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
                "descriptor_dim": engine.config.descriptor_dim,
                "pyramid_enabled": engine.config.pyramid_enabled,
                "lsh_enabled": engine.config.lsh_enabled,
                "coherence_enabled": engine.config.coherence_enabled,
                "use_gpu": engine.config.use_gpu,
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
            "descriptor_dim": engine.config.descriptor_dim,
            "min_probability": engine.config.min_probability,
            "pyramid_enabled": engine.config.pyramid_enabled,
            "lsh_enabled": engine.config.lsh_enabled,
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
        image_array = await validate_uploaded_image(file)

        # Sanitize and validate image_id
        if image_id is not None:
            # Remove potentially dangerous characters
            image_id = "".join(c for c in image_id if c.isalnum() or c in "._-")
            if not image_id:
                raise HTTPException(status_code=400, detail="Invalid image_id after sanitization")
        else:
            # Generate ID needs to be inside lock to avoid duplicates
            with index_lock:
                image_id = f"img_{engine.target_count:04d}"

        # Use lock to prevent race condition between check and index
        with index_lock:
            # Check if already exists
            if engine.has_target(image_id):
                raise HTTPException(
                    status_code=409,
                    detail=f"Image ID '{image_id}' already exists. Use a different ID or delete the existing one."
                )

            try:
                # Index the image
                count = engine.index_target(image_array, image_id)

                return IndexResponse(
                    success=True,
                    image_id=image_id,
                    patches_indexed=count,
                    message=f"Indexed {count} patches from {file.filename}",
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

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
        image_array = await validate_uploaded_image(file)

        try:

            # Parse target IDs if provided
            target_id_list = None
            if target_ids:
                target_id_list = [t.strip() for t in target_ids.split(",")]

            # Run matching
            result = engine.match(
                image_array,
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
                    probability=result.best_match.probability,
                    image_id=result.best_match.image_id,
                )

            # Build coherent regions info
            regions_info = []
            for region in result.coherent_regions:
                regions_info.append(CoherentRegionInfo(
                    offset=MatchCoord(x=region.offset[0], y=region.offset[1]),
                    num_matches=len(region.matches),
                    confidence=region.confidence,
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
                best_match=best_match_info,
                total_matches=len(result.matches),
                coherent_regions=regions_info,
                stats=result.stats,
                heatmap_base64=heatmap_b64,
            )

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

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
        Remove a target from the index.

        Note: Removes metadata only. Descriptors remain in FAISS but won't
        match due to metadata filtering during search.
        """
        with index_lock:
            if not engine.remove_target(image_id):
                raise HTTPException(status_code=404, detail=f"Target {image_id} not found")

            return {
                "success": True,
                "message": f"Removed target {image_id}",
                "note": "Descriptors remain in FAISS index but won't match",
            }

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
