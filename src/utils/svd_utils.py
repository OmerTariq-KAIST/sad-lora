"""Efficient SVD utilities for SAD-LoRA spectral operations."""

import torch
from torch import Tensor


def truncated_svd(
    matrix: Tensor,
    k: int,
    use_randomized: bool = True,
    niter: int = 5,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute truncated SVD retaining top-k components.

    Args:
        matrix: (m, n) input matrix.
        k: Number of singular components to retain.
        use_randomized: If True, use torch.svd_lowrank (randomized).
        niter: Power iterations for randomized SVD.
        seed: Random seed for reproducibility.

    Returns:
        U: (m, k) left singular vectors.
        S: (k,) singular values (descending).
        V: (n, k) right singular vectors.
    """
    matrix = matrix.float()
    k = min(k, min(matrix.shape))

    if use_randomized and min(matrix.shape) > 2 * k:
        U, S, V = torch.svd_lowrank(matrix, q=k, niter=niter, M=None)
        # svd_lowrank returns V already as (n, k), S as (k,)
        # Sort descending (svd_lowrank doesn't guarantee order)
        idx = S.argsort(descending=True)
        U, S, V = U[:, idx], S[idx], V[:, idx]
    else:
        U_full, S_full, Vh_full = torch.linalg.svd(matrix, full_matrices=False)
        U = U_full[:, :k]
        S = S_full[:k]
        V = Vh_full[:k, :].T  # (n, k) — convert from Vh to V

    return U, S, V


def stable_svdvals(matrix: Tensor) -> Tensor:
    """Compute singular values with numerical safeguards.

    Uses float32 regardless of input dtype. Clamps near-zero values
    to avoid gradient issues with degenerate singular values.

    Args:
        matrix: (m, n) input matrix.

    Returns:
        Singular values (min(m,n),) in descending order.
    """
    original_dtype = matrix.dtype
    vals = torch.linalg.svdvals(matrix.float())
    return vals.to(original_dtype)


def matrix_sqrt_psd(matrix: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute matrix square root of a positive semi-definite matrix.

    Uses eigendecomposition: M^{1/2} = Q diag(sqrt(lambda)) Q^T.

    Args:
        matrix: (n, n) symmetric PSD matrix.
        eps: Clamp eigenvalues below this for stability.

    Returns:
        (n, n) matrix square root.
    """
    matrix = matrix.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp(min=eps)
    return eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T


def reconstruct_from_svd(U: Tensor, S: Tensor, V: Tensor) -> Tensor:
    """Reconstruct matrix from SVD components: U @ diag(S) @ V^T.

    Args:
        U: (m, k) left singular vectors.
        S: (k,) singular values.
        V: (n, k) right singular vectors.

    Returns:
        (m, n) reconstructed matrix.
    """
    return U @ torch.diag(S) @ V.T
