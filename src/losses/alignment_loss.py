"""Subspace Alignment Loss (L_align) — core SAD-LoRA contribution.

Measures how well the LoRA adapter's column space colspan(B) aligns
with the teacher's data-weighted spectral subspace colspan(U_tilde).

L_align = 1 - (1/r) ||Q_B^T U_target||_F^2

where Q_B is obtained from QR decomposition of B (orthonormal basis
for colspan(B)), and U_target are the top-r data-weighted left singular
vectors of the teacher's weight update.

The loss equals 0 when the subspaces are identical, and 1 when orthogonal.

Gradient flows through torch.linalg.qr, which PyTorch supports natively.
QR gradients can be unstable when B has near-linearly-dependent columns;
this is mitigated by Kaiming initialization and gradient clipping.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SubspaceAlignmentLoss(nn.Module):
    """Computes L_align = 1 - (1/r) ||Q_B^T U_target||_F^2.

    This loss drives the LoRA adapter's column space toward the teacher's
    spectrally important subspace, directly targeting Term (I) of the
    distillation error decomposition (Theorem 1).
    """

    def __init__(self, normalize_per_layer: bool = True):
        """
        Args:
            normalize_per_layer: Divide by r so loss is in [0, 1].
        """
        super().__init__()
        self.normalize_per_layer = normalize_per_layer

    def forward(
        self, B: Tensor, U_target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute alignment loss between colspan(B) and colspan(U_target).

        All computation forced to float32 for SVD/QR numerical stability.

        Args:
            B: (d_out, r) LoRA B matrix. Must have requires_grad=True.
            U_target: (d_out, r) target left singular vectors. Frozen.

        Returns:
            loss: Scalar in [0, 1].
            alignment_score: 1 - loss (for logging).
            principal_angles: (r,) angles in radians (detached).
        """
        B_f32 = B.float()
        U_target_f32 = U_target.detach().float()
        r = B_f32.shape[1]

        # QR decomposition for orthonormal basis of colspan(B)
        Q_B, R_B = torch.linalg.qr(B_f32, mode="reduced")  # Q: (d_out, r)

        # Cross-Gram matrix: G = Q_B^T @ U_target, shape (r, r)
        G = Q_B.T @ U_target_f32

        # Alignment = (1/r) ||G||_F^2 = (1/r) sum cos^2(theta_i)
        alignment = (G ** 2).sum()
        if self.normalize_per_layer:
            alignment = alignment / r

        loss = 1.0 - alignment

        # Principal angles for diagnostics (detached — no gradient needed)
        with torch.no_grad():
            cos_angles = torch.linalg.svdvals(G).clamp(0.0, 1.0)
            angles = torch.arccos(cos_angles)

        return loss, alignment.detach(), angles

    def forward_from_qr(
        self, Q_B: Tensor, U_target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute alignment loss from a pre-computed Q_B (shared QR path).

        Use this in SADLoRALoss to avoid recomputing QR when R_B is already
        needed for the coefficient loss.

        Args:
            Q_B: (d_out, r) orthonormal basis already computed via compute_qr().
            U_target: (d_out, r) target left singular vectors. Frozen.

        Returns:
            Same as forward(): (loss, alignment_score, principal_angles).
        """
        Q_B_f32 = Q_B.float()
        U_target_f32 = U_target.detach().float()
        r = Q_B_f32.shape[1]

        G = Q_B_f32.T @ U_target_f32
        alignment = (G ** 2).sum()
        if self.normalize_per_layer:
            alignment = alignment / r
        loss = 1.0 - alignment

        with torch.no_grad():
            cos_angles = torch.linalg.svdvals(G).clamp(0.0, 1.0)
            angles = torch.arccos(cos_angles)

        return loss, alignment.detach(), angles

    @staticmethod
    def compute_qr(B: Tensor) -> tuple[Tensor, Tensor]:
        """Shared QR decomposition, reusable by coefficient loss.

        Args:
            B: (d_out, r) LoRA B matrix.

        Returns:
            Q_B: (d_out, r) orthonormal basis.
            R_B: (r, r) upper triangular factor.
        """
        return torch.linalg.qr(B.float(), mode="reduced")
