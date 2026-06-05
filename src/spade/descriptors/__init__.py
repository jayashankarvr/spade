from spade.descriptors.core import DescriptorStrategy, CompositeDescriptor

# Optional: LearnedDescriptor requires PyTorch
try:
    from spade.descriptors.learned import LearnedDescriptor, TORCH_AVAILABLE
except ImportError:
    LearnedDescriptor = None
    TORCH_AVAILABLE = False

__all__ = ["DescriptorStrategy", "CompositeDescriptor", "LearnedDescriptor", "TORCH_AVAILABLE"]
