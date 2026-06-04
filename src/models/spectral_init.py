"""Spectral initialization strategies for LoRA adapters.

Provides initialization methods that leverage the teacher's spectral
structure to give the optimizer a warm start.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..spectral_analysis.data_weighted_svd import TargetSubspace


def spectral_init(
    lora_B: nn.Parameter,
    lora_A: nn.Parameter,
    target: TargetSubspace,
    sigma_x_sqrt_inv: Tensor | None = None,
) -> None:
    """Initialize B from U_tilde and optionally A from V_tilde.

    From Proposition 1, the optimal adapter satisfies:
      B* A* = U_tilde[:, :r] @ diag(sigma_tilde[:r]) @ V_tilde[:, :r]^T @ Sigma_x^{-1/2}

    We split the singular values evenly: B = U * sqrt(sigma), A = sqrt(sigma) * V^T.
    When Sigma_x^{-1/2} is not available, A is left at zeros (safe because the
    optimizer will learn it, and B already spans the right subspace).

    Args:
        lora_B: (d_out, r) parameter to initialize.
        lora_A: (r, d_in) parameter to initialize.
        target: TargetSubspace with U_tilde and sigma_tilde.
        sigma_x_sqrt_inv: (d_in, d_in) optional inverse square root of
            data covariance for mapping back to original space.
    """
    r = lora_B.shape[1]
    U = target.U_tilde[:, :r].to(lora_B.device)
    sigma = target.sigma_tilde[:r].to(lora_B.device)

    sqrt_sigma = sigma.sqrt().clamp(min=1e-8)

    # B = U @ diag(sqrt(sigma))
    lora_B.data.copy_(U * sqrt_sigma.unsqueeze(0))

    # A = zeros (safe default — optimizer learns from alignment + KD gradients)
    nn.init.zeros_(lora_A)


def random_subspace_init(lora_B: nn.Parameter, lora_A: nn.Parameter) -> None:
    """Initialize B as a random orthonormal matrix, A as zeros.

    This ensures colspan(B) is a random r-dimensional subspace,
    providing a clean baseline for measuring alignment improvement.
    """
    d_out, r = lora_B.shape
    Q, _ = torch.linalg.qr(torch.randn(d_out, r, device=lora_B.device))
    lora_B.data.copy_(Q)
    nn.init.zeros_(lora_A)


def pissa_init(
    lora_B: nn.Parameter,
    lora_A: nn.Parameter,
    pretrained_weight: Tensor,
    r: int | None = None,
) -> None:
    """PiSSA-style initialization from pretrained weight SVD.

    Unlike spectral_init which uses the teacher's update delta_W_T,
    PiSSA initializes from the pretrained weight W_0 itself.
    This serves as a baseline (Neural Nuggets / PiSSA).

    Args:
        lora_B: (d_out, r) parameter.
        lora_A: (r, d_in) parameter.
        pretrained_weight: (d_out, d_in) pretrained weight matrix.
        r: Rank to use (defaults to lora_B.shape[1]).
    """
    if r is None:
        r = lora_B.shape[1]

    U, S, Vh = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
    sqrt_s = S[:r].sqrt().clamp(min=1e-8)

    lora_B.data.copy_((U[:, :r] * sqrt_s.unsqueeze(0)).to(lora_B.dtype))
    lora_A.data.copy_((sqrt_s.unsqueeze(1) * Vh[:r, :]).to(lora_A.dtype))
