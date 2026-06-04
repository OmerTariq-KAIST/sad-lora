"""Covariance estimation with Ledoit-Wolf shrinkage for data-weighted SVD."""

import torch
from torch import Tensor


class CovarianceEstimator:
    """Estimates input covariance matrices from activation samples.

    Supports multiple methods with different accuracy/memory tradeoffs.
    """

    def __init__(self, method: str = "empirical_shrinkage"):
        """
        Args:
            method: One of:
                "empirical" — full empirical covariance
                "empirical_shrinkage" — Ledoit-Wolf shrinkage (recommended)
                "diagonal" — diagonal-only (fast, less accurate)
        """
        if method not in ("empirical", "empirical_shrinkage", "diagonal"):
            raise ValueError(f"Unknown covariance method: {method}")
        self.method = method

    def estimate(self, X: Tensor) -> Tensor:
        """Estimate covariance from (n, d) activation matrix.

        Args:
            X: (n_samples, d_in) input activations, float32.

        Returns:
            (d_in, d_in) covariance matrix (or diagonal if method="diagonal").
        """
        X = X.float()
        n, d = X.shape
        mu = X.mean(dim=0)
        X_c = X - mu  # centered

        if self.method == "diagonal":
            var = (X_c ** 2).mean(dim=0)
            return torch.diag(var.clamp(min=1e-8))

        # Full empirical covariance: (1/(n-1)) X_c^T X_c
        S = (X_c.T @ X_c) / max(n - 1, 1)

        if self.method == "empirical":
            return S

        # Ledoit-Wolf shrinkage
        alpha = self._ledoit_wolf_alpha(X_c, S)
        target = torch.eye(d, device=S.device, dtype=S.dtype) * S.trace() / d
        return (1.0 - alpha) * S + alpha * target

    @staticmethod
    def _ledoit_wolf_alpha(X_c: Tensor, S: Tensor) -> float:
        """Compute optimal Ledoit-Wolf shrinkage intensity.

        Analytical formula from Ledoit & Wolf (2004).
        """
        n, d = X_c.shape
        trace_S2 = (S ** 2).sum()
        trace_S_sq = S.trace() ** 2

        # Estimate sum of squared off-diagonal elements
        # Using the identity: sum_ij (x_i^T x_j)^2 / n^2
        X_sq = X_c ** 2
        phi_sum = (X_sq.T @ X_sq).sum() / n  # sum of E[x_i^2 x_j^2]
        phi = phi_sum / max(n - 1, 1) - trace_S2

        # Optimal shrinkage intensity
        gamma = d / n
        kappa = (phi + trace_S2) / ((n + 1 - 2 / d) * trace_S2 + trace_S_sq)
        alpha = max(0.0, min(1.0, kappa * gamma))
        return alpha

    def compute_sqrt(self, cov: Tensor, eps: float = 1e-8) -> Tensor:
        """Compute matrix square root via eigendecomposition.

        Args:
            cov: (d, d) PSD covariance matrix.
            eps: Eigenvalue floor.

        Returns:
            (d, d) matrix such that result @ result^T ≈ cov.
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(cov.float())
        eigenvalues = eigenvalues.clamp(min=eps)
        return eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
