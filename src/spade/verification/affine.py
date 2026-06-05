"""Affine color transform verification using Huber loss."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class VerificationResult:
    """Result of affine color transform verification."""
    M: np.ndarray       # 3x3 color transformation matrix
    b: np.ndarray       # 3x1 bias vector
    rss: float          # Residual Sum of Squares (for chi-square test)
    dof: int            # degrees of freedom
    success: bool       # whether verification succeeded


class AffineVerifier:
    """
    Verifies patch matches by fitting an affine color transform.

    Model: target_color = M @ source_color + b

    Uses Huber loss which is robust to outliers (JPEG artifacts, noise)
    while maintaining efficiency for small residuals.
    """

    def __init__(
        self,
        loss: str = "huber",
        noise_sigma: float = 0.02,
        regularization: float = 1e-4,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6,
    ):
        """
        Args:
            loss: Loss function ("huber" or "l2")
            noise_sigma: Expected noise level for Huber delta scaling
            regularization: L2 regularization strength
            max_iterations: Max IRLS iterations for Huber
            convergence_threshold: Stop IRLS when change < this
        """
        self.loss = loss
        # Huber delta scaled to noise level (1.345σ gives 95% efficiency)
        self.huber_delta = 1.345 * noise_sigma
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def verify(
        self,
        source_patch: np.ndarray,
        target_patch: np.ndarray,
    ) -> VerificationResult:
        """
        Estimate affine color transform between patches.

        Args:
            source_patch: Source RGB patch (H, W, 3)
            target_patch: Target RGB patch (H, W, 3)

        Returns:
            VerificationResult with transform and RSS
        """
        if source_patch.shape != target_patch.shape:
            return VerificationResult(
                M=np.eye(3),
                b=np.zeros(3),
                rss=float('inf'),
                dof=0,
                success=False,
            )

        height, width, _ = source_patch.shape
        n_pixels = height * width
        dof = max(1, n_pixels * 3 - 12)  # 12 affine parameters

        source = source_patch.reshape(-1, 3)
        target = target_patch.reshape(-1, 3)

        if self.loss == "huber":
            M, b, rss = self._solve_huber(source, target)
        else:
            M, b, rss = self._solve_l2(source, target)

        success = rss < float('inf') and not np.isnan(rss)

        return VerificationResult(
            M=M,
            b=b,
            rss=rss,
            dof=dof,
            success=success,
        )

    def _solve_l2(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Standard regularized least squares."""
        n = source.shape[0]
        X = np.hstack([source, np.ones((n, 1))])

        # Normal equations with regularization
        XtX = X.T @ X
        XtX[np.diag_indices(4)] += self.regularization

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return np.eye(3), np.zeros(3), float('inf')

        M = np.zeros((3, 3))
        b = np.zeros(3)

        for c in range(3):
            params = XtX_inv @ (X.T @ target[:, c])
            M[c, :] = params[:3]
            b[c] = params[3]

        predicted = source @ M.T + b
        residuals = target - predicted
        rss = float(np.sum(residuals ** 2))
        return M, b, rss

    def _solve_huber(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Iteratively Reweighted Least Squares for Huber loss."""
        # Initialize with L2 solution. Capture its RSS so that `rss` is always
        # defined even if the first IRLS iteration's matrix inversion fails.
        M, b, rss = self._solve_l2(source, target)

        n = source.shape[0]
        X = np.hstack([source, np.ones((n, 1))])

        prev_rss = float('inf')

        for _ in range(self.max_iterations):
            predicted = source @ M.T + b
            residuals = target - predicted
            residual_norms = np.linalg.norm(residuals, axis=1)

            # Huber weights: 1 for small residuals, decreasing for large
            # This is the key robustness mechanism
            weights = np.where(
                residual_norms <= self.huber_delta,
                1.0,
                self.huber_delta / (residual_norms + 1e-10),
            )

            # Weighted least squares without full diagonal matrix
            # (X.T * weights) @ X is equivalent to X.T @ diag(weights) @ X
            Xw = X * weights[:, np.newaxis]
            XtWX = Xw.T @ X
            XtWX[np.diag_indices(4)] += self.regularization

            try:
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                break

            for c in range(3):
                Xty_weighted = Xw.T @ target[:, c]
                params = XtWX_inv @ Xty_weighted
                M[c, :] = params[:3]
                b[c] = params[3]

            # Check convergence
            predicted = source @ M.T + b
            residuals = target - predicted
            rss = float(np.sum(residuals ** 2))

            if abs(prev_rss - rss) < self.convergence_threshold:
                break
            prev_rss = rss

        return M, b, rss
