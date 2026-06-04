"""Spectral Coefficient Matching Loss (L_coeff) — core SAD-LoRA contribution.

Matches the singular value spectrum of the LoRA adapter B@A to the
target singular values from the data-weighted teacher SVD.

L_coeff = (1/r) sum_i (sigma_i(BA) - sigma_tilde_i)^2

Key optimization: sigma_i(BA) = sigma_i(R_B @ A) when B = Q_B @ R_B,
since Q_B is orthonormal. This reduces SVD from (d_out, d_in) to (r, d_in).

Gradient flows through torch.linalg.svdvals. This is well-defined when
singular values are distinct. Near-degenerate values cause gradient spikes,
mitigated by gradient clipping (max_norm=1.0) in the training loop.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SpectralCoefficientLoss(nn.Module):
    """Computes L_coeff = (1/r) sum_i (sigma_i(BA) - sigma_target_i)^2.

    Targets Term (II) of the distillation error decomposition (Theorem 1):
    ensures the adapter reproduces the teacher's spectral energy within
    the aligned subspace.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        normalize_by_target: bool = True,
    ):
        """
        Args:
            loss_type: How to compute the loss.
                "mse" — mean squared error on raw singular values.
                "log_mse" — MSE on log singular values (scale-invariant).
                "relative_mse" — ((sigma_s - sigma_t) / sigma_t)^2.
            normalize_by_target: Divide by ||sigma_target||^2 to make
                the loss scale-invariant across layers.
        """
        super().__init__()
        if loss_type not in ("mse", "log_mse", "relative_mse"):
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.loss_type = loss_type
        self.normalize_by_target = normalize_by_target

    def forward(
        self,
        R_B: Tensor,
        A: Tensor,
        sigma_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute coefficient matching loss.

        Uses pre-computed R_B from shared QR decomposition:
        sigma_i(BA) = sigma_i(R_B @ A) since Q_B is orthonormal.

        All computation forced to float32 for SVD stability.

        Args:
            R_B: (r, r) upper triangular from QR(B). Float32.
            A: (r, d_in) LoRA A matrix. Must have requires_grad=True.
            sigma_target: (r,) target singular values. Frozen.

        Returns:
            loss: Scalar loss value.
            sigma_student: (r,) current adapter singular values (detached).
        """
        R_B_f32 = R_B.float()
        A_f32 = A.float()
        sigma_t = sigma_target.detach().float()
        r = sigma_t.shape[0]

        # Efficient SVD: work with (r, d_in) matrix instead of (d_out, d_in)
        M = R_B_f32 @ A_f32  # (r, d_in)
        sigma_s = torch.linalg.svdvals(M)[:r]  # (r,), descending

        # Compute loss based on type
        if self.loss_type == "mse":
            diff = sigma_s - sigma_t
            loss = (diff ** 2).mean()
        elif self.loss_type == "log_mse":
            eps = 1e-8
            diff = torch.log(sigma_s + eps) - torch.log(sigma_t + eps)
            loss = (diff ** 2).mean()
        elif self.loss_type == "relative_mse":
            eps = 1e-8
            diff = (sigma_s - sigma_t) / (sigma_t + eps)
            loss = (diff ** 2).mean()

        # Optionally normalize to make loss scale-invariant
        if self.normalize_by_target:
            target_norm_sq = (sigma_t ** 2).sum().clamp(min=1e-10)
            loss = loss * r / target_norm_sq

        return loss, sigma_s.detach()
