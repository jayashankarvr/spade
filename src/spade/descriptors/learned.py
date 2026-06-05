"""Learned CNN-based descriptors using pre-trained models."""

import warnings
from typing import Optional
import numpy as np

from spade.descriptors.core import DescriptorStrategy, _normalize
from spade.exceptions import DependencyError, ConfigurationError

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False


class LearnedDescriptor(DescriptorStrategy):
    """
    CNN-based descriptor using pre-trained features.

    Extracts features from a pre-trained ResNet/EfficientNet and
    projects to target dimension. Works on any patch size by resizing.
    """

    SUPPORTED_MODELS = ["resnet18", "resnet50", "efficientnet_b0"]

    def __init__(
        self,
        model_name: str = "resnet18",
        target_dim: int = 256,
        use_gpu: bool = False,
    ):
        """
        Args:
            model_name: Pre-trained model to use (resnet18, resnet50, efficientnet_b0)
            target_dim: Output descriptor dimension
            use_gpu: Use GPU for inference if available
        """
        if not TORCH_AVAILABLE:
            raise DependencyError("PyTorch required: pip install torch torchvision")

        if model_name not in self.SUPPORTED_MODELS:
            raise ConfigurationError(
                f"Unknown model: {model_name}. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_name = model_name
        self.target_dim = target_dim

        # Device selection with fallback
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu and not torch.cuda.is_available():
            warnings.warn("CUDA not available, using CPU for learned descriptors")
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Load pre-trained model
        self.model, self._feature_dim = self._load_model(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Projection to target dimension (deterministic random projection)
        self._projection: Optional[np.ndarray] = None
        if self._feature_dim != target_dim:
            rng = np.random.RandomState(42)
            self._projection = rng.randn(self._feature_dim, target_dim).astype(
                np.float32
            )
            # Normalize columns
            self._projection /= np.linalg.norm(
                self._projection, axis=0, keepdims=True
            )

        # Preprocessing: resize small patches + ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),  # Minimum size for CNN
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Storage for intermediate features (filled by hook)
        self._features: Optional[torch.Tensor] = None

    def _load_model(self, name: str):
        """Load pre-trained model and return (model, feature_dim)."""
        if name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
            # Register hook on avgpool
            model.avgpool.register_forward_hook(self._capture_features)
        elif name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048
            model.avgpool.register_forward_hook(self._capture_features)
        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            feature_dim = 1280
            model.avgpool.register_forward_hook(self._capture_features)
        else:
            raise ConfigurationError(f"Unknown model: {name}")

        # Freeze all parameters (inference only)
        for param in model.parameters():
            param.requires_grad = False

        return model, feature_dim

    def _capture_features(self, module, input, output):
        """Forward hook to capture intermediate features."""
        self._features = output

    def compute(self, patch: np.ndarray) -> np.ndarray:
        """Compute CNN descriptor for single patch."""
        return self.compute_batch(patch[np.newaxis])[0]

    def compute_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Compute CNN descriptors for batch of patches.

        Args:
            patches: (N, H, W, 3) float32 patches in [0, 1]

        Returns:
            (N, target_dim) L2-normalized descriptors
        """
        n = len(patches)
        if n == 0:
            return np.zeros((0, self.target_dim), dtype=np.float32)

        # Convert to torch tensor (N, C, H, W)
        tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float()

        # Apply preprocessing (resize + normalize)
        tensor = self.transform(tensor)
        tensor = tensor.to(self.device)

        # Forward pass (populates self._features via hook)
        with torch.no_grad():
            _ = self.model(tensor)
            features = self._features.squeeze(-1).squeeze(-1)  # (N, feature_dim)

        # Convert to numpy
        features = features.cpu().numpy()

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Project to target dimension
        if self._projection is not None:
            features = features @ self._projection

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms

        return features.astype(np.float32)
