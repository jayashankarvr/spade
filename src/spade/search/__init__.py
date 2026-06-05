from spade.search.index import ANNIndex, SearchResult, FAISS_GPU_AVAILABLE
from spade.search.distributed import ShardedIndex

# LSH is optional (requires datasketch)
try:
    from spade.search.lsh import LSHPreFilter, LSHConfig, HybridIndex
    LSH_AVAILABLE = True
except ImportError:
    LSHPreFilter = None
    LSHConfig = None
    HybridIndex = None
    LSH_AVAILABLE = False

__all__ = [
    "ANNIndex",
    "SearchResult",
    "FAISS_GPU_AVAILABLE",
    "ShardedIndex",
    "LSHPreFilter",
    "LSHConfig",
    "HybridIndex",
    "LSH_AVAILABLE",
]
