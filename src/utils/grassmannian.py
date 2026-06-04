"""Grassmannian geometry utilities for subspace comparison.

Implements principal angle computation, alignment scores, and
subspace distance metrics between column spaces.
"""

import torch
from torch import Tensor


def principal_angles(
    A: Tensor, B: Tensor, return_cos: bool = False
) -> Tensor:
    """Compute principal angles between two subspaces.

    Given orthonormal bases or arbitrary matrices whose column spaces
    define the subspaces, compute the r principal angles.

    Args:
        A: (d, r) matrix defining subspace 1.
        B: (d, r) matrix defining subspace 2 (same r).
        return_cos: If True, return cos(theta) instead of theta.

    Returns:
        (r,) principal angles in radians (ascending), or cosines (descending).
    """
    Q_A = _orthonormalize(A)
    Q_B = _orthonormalize(B)

    # Cross-Gram matrix: cos(theta_i) = sigma_i(Q_A^T Q_B)
    G = Q_A.T @ Q_B
    cos_values = torch.linalg.svdvals(G.float()).clamp(0.0, 1.0)

    if return_cos:
        return cos_values  # descending
    return torch.arccos(cos_values)  # ascending


def alignment_score(A: Tensor, B: Tensor) -> float:
    """Compute aggregate alignment score between two subspaces.

    A(A, B) = (1/r) * sum cos^2(theta_i) = (1/r) * ||Q_A^T Q_B||_F^2

    Returns a scalar in [0, 1]. 1 = identical subspaces, 0 = orthogonal.
    """
    Q_A = _orthonormalize(A)
    Q_B = _orthonormalize(B)
    r = Q_A.shape[1]
    G = Q_A.T @ Q_B
    return (G ** 2).sum().item() / r


def subspace_distance(A: Tensor, B: Tensor, metric: str = "projection") -> float:
    """Compute distance between two subspaces.

    Args:
        A, B: (d, r) matrices defining subspaces.
        metric: Distance metric.
            "projection": ||P_A - P_B||_F / sqrt(2r)
            "geodesic": ||theta||_2 (L2 norm of principal angles)
            "chordal": sqrt(sum sin^2(theta_i))

    Returns:
        Non-negative distance scalar.
    """
    thetas = principal_angles(A, B)

    if metric == "projection":
        return (thetas.sin() ** 2).sum().sqrt().item() / (A.shape[1] ** 0.5)
    elif metric == "geodesic":
        return thetas.norm().item()
    elif metric == "chordal":
        return (thetas.sin() ** 2).sum().sqrt().item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def spectral_overlap_matrix(U_adapter: Tensor, U_teacher: Tensor) -> Tensor:
    """Compute pairwise |cos(angle)| between singular vectors.

    Useful for visualization: rows = adapter directions, cols = teacher directions.

    Args:
        U_adapter: (d, r1) adapter singular vectors.
        U_teacher: (d, r2) teacher singular vectors.

    Returns:
        (r1, r2) matrix of absolute cosine similarities.
    """
    Q_A = _orthonormalize(U_adapter)
    Q_T = _orthonormalize(U_teacher)
    return (Q_A.T @ Q_T).abs()


def _orthonormalize(M: Tensor) -> Tensor:
    """QR-based orthonormalization of column space."""
    Q, _ = torch.linalg.qr(M.float(), mode="reduced")
    return Q
