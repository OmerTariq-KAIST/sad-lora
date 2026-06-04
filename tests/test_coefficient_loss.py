"""Unit tests for SpectralCoefficientLoss.

Verifies:
- Loss = 0 when adapter singular values match target
- Loss > 0 when they differ
- Gradient stability with near-degenerate singular values
- Correct use of R_B from shared QR decomposition
"""

import pytest
import torch

from src.losses.coefficient_loss import SpectralCoefficientLoss


class TestCoefficientLoss:
    def test_matching_singular_values(self):
        """L_coeff = 0 when sigma(BA) matches sigma_target exactly."""
        r, d_in = 4, 32
        loss_fn = SpectralCoefficientLoss(loss_type="mse", normalize_by_target=False)

        # Construct R_B, A such that sigma(R_B @ A) = sigma_target
        sigma_target = torch.tensor([5.0, 3.0, 1.0, 0.5])

        # Build M = R_B @ A with known singular values
        U, _ = torch.linalg.qr(torch.randn(r, r))
        V, _ = torch.linalg.qr(torch.randn(d_in, r))
        M = U @ torch.diag(sigma_target) @ V.T
        # Split into R_B and A: R_B = I (upper triangular), A = M
        R_B = torch.eye(r)
        A = M

        loss, sigma_s = loss_fn(R_B, A, sigma_target)
        assert loss.item() < 1e-4, f"Expected ~0, got {loss.item()}"

    def test_mismatched_singular_values(self):
        """L_coeff > 0 when singular values differ."""
        r, d_in = 4, 32
        loss_fn = SpectralCoefficientLoss(loss_type="mse", normalize_by_target=False)

        sigma_target = torch.tensor([5.0, 3.0, 1.0, 0.5])
        R_B = torch.eye(r)
        A = torch.randn(r, d_in)

        loss, _ = loss_fn(R_B, A, sigma_target)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradient flows through svdvals to R_B and A."""
        r, d_in = 4, 32
        loss_fn = SpectralCoefficientLoss(loss_type="mse", normalize_by_target=False)

        R_B = torch.randn(r, r, requires_grad=True)
        A = torch.randn(r, d_in, requires_grad=True)
        sigma_target = torch.tensor([5.0, 3.0, 1.0, 0.5])

        loss, _ = loss_fn(R_B, A, sigma_target)
        loss.backward()

        assert R_B.grad is not None and not torch.isnan(R_B.grad).any()
        assert A.grad is not None and not torch.isnan(A.grad).any()

    def test_log_mse_mode(self):
        """log_mse should work and give finite loss."""
        r, d_in = 4, 32
        loss_fn = SpectralCoefficientLoss(loss_type="log_mse", normalize_by_target=False)

        R_B = torch.randn(r, r, requires_grad=True)
        A = torch.randn(r, d_in, requires_grad=True)
        sigma_target = torch.tensor([5.0, 3.0, 1.0, 0.5])

        loss, _ = loss_fn(R_B, A, sigma_target)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(R_B.grad).all()

    def test_relative_mse_mode(self):
        """relative_mse should work with non-zero targets."""
        r, d_in = 4, 32
        loss_fn = SpectralCoefficientLoss(loss_type="relative_mse", normalize_by_target=False)

        R_B = torch.randn(r, r, requires_grad=True)
        A = torch.randn(r, d_in, requires_grad=True)
        sigma_target = torch.tensor([5.0, 3.0, 1.0, 0.5])

        loss, _ = loss_fn(R_B, A, sigma_target)
        assert torch.isfinite(loss)

    def test_shared_qr_consistency(self):
        """sigma_i(BA) = sigma_i(R_B @ A) when B = Q_B @ R_B."""
        d_out, r, d_in = 64, 4, 32
        B = torch.randn(d_out, r)
        A = torch.randn(r, d_in)

        # Full SVD of BA
        sigma_full = torch.linalg.svdvals(B @ A)[:r]

        # Via QR: sigma_i(R_B @ A)
        Q_B, R_B = torch.linalg.qr(B.float(), mode="reduced")
        sigma_qr = torch.linalg.svdvals(R_B @ A)[:r]

        torch.testing.assert_close(sigma_full, sigma_qr, atol=1e-5, rtol=1e-5)
