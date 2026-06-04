"""Numerical verification of Theorem 1: Distillation Error Decomposition.

For the linear model f(x) = Wx with x ~ N(0, Sigma_x), verifies:

E_x[||(delta_W - BA)x||^2] = Term_I + Term_II + Term_III

where:
- Term I = sum sigma_tilde_i^2 * sin^2(theta_i)  (subspace misalignment)
- Term II = sum sigma_tilde_i^2 * cos^2(theta_i) * (1 - sigma_hat_i/sigma_tilde_i)^2  (coefficient mismatch)
- Term III = sum_{i>r} sigma_tilde_i^2  (rank residual)

The decomposition should hold EXACTLY for the linear case.
"""

import pytest
import torch

from experiments.run_synthetic import (
    generate_teacher_spectrum,
    generate_data_covariance,
    verify_theorem1,
    compute_data_weighted_svd,
)


class TestTheorem1:
    @pytest.mark.parametrize("spectrum", ["sharp_decay", "gradual_decay", "flat_spectrum"])
    @pytest.mark.parametrize("cov_type", ["identity", "exponential_decay"])
    def test_exact_decomposition(self, spectrum, cov_type):
        """Error decomposition should hold exactly for linear models."""
        d_out, d_in = 64, 32
        r = 4

        U, sigma, V = generate_teacher_spectrum(d_out, d_in, spectrum)
        delta_W = U @ torch.diag(sigma) @ V.T
        sigma_x = generate_data_covariance(d_in, cov_type)

        # Random adapter
        torch.manual_seed(123)
        B = torch.randn(d_out, r) * 0.1
        A = torch.randn(r, d_in) * 0.1

        result = verify_theorem1(delta_W, B, A, sigma_x, r)

        # The key check: sum of terms ≈ total error (only when error is non-trivial)
        if result["decomp_valid"]:
            assert result["relative_residual"] < 0.05, (
                f"Decomposition failed for {spectrum}/{cov_type}: "
                f"relative_residual={result['relative_residual']:.4e}"
            )
        else:
            # Adapter has converged; both total_error and residual are near 0 — OK
            assert result["residual"] < 1e-3, (
                f"Absolute residual too large for converged case: {result['residual']:.4e}"
            )

    def test_perfect_adapter_zero_terms_I_II(self):
        """Optimal adapter should have Term I = 0 and Term II = 0."""
        d_out, d_in = 64, 32
        r = 4
        torch.manual_seed(42)

        U, sigma, V = generate_teacher_spectrum(d_out, d_in, "sharp_decay")
        delta_W = U @ torch.diag(sigma) @ V.T
        sigma_x = torch.eye(d_in)

        # Optimal adapter: truncated SVD of delta_W (since Sigma_x = I)
        U_full, S_full, Vh_full = torch.linalg.svd(delta_W, full_matrices=False)
        # Split optimal BA into B=U, A=diag(S)Vh for correct QR on B
        B_opt = U_full[:, :r]
        A_opt = torch.diag(S_full[:r]) @ Vh_full[:r, :]

        result = verify_theorem1(delta_W, B_opt, A_opt, sigma_x, r)

        assert result["term_I"] < 1e-4, f"Term I should be ~0, got {result['term_I']}"
        assert result["term_II"] < 1e-4, f"Term II should be ~0, got {result['term_II']}"
        assert result["term_III"] > 0  # Rank residual must be positive

    def test_misaligned_adapter_large_term_I(self):
        """Adapter in wrong subspace should have large Term I."""
        d_out, d_in = 64, 32
        r = 4
        torch.manual_seed(42)

        U, sigma, V = generate_teacher_spectrum(d_out, d_in, "sharp_decay")
        delta_W = U @ torch.diag(sigma) @ V.T
        sigma_x = torch.eye(d_in)

        # Deliberately misaligned adapter: random orthogonal B, correct magnitudes
        B_rand, _ = torch.linalg.qr(torch.randn(d_out, r))
        A_rand = torch.diag(sigma[:r]) @ V[:, :r].T

        result = verify_theorem1(delta_W, B_rand, A_rand, sigma_x, r)

        # Term I should be significant because subspaces are misaligned
        assert result["term_I"] > 0.1, f"Term I should be large, got {result['term_I']}"

    def test_data_weighting_matters(self):
        """Data-weighted SVD should differ from unweighted when Sigma_x ≠ I."""
        d_out, d_in = 64, 32
        torch.manual_seed(42)

        U, sigma, V = generate_teacher_spectrum(d_out, d_in, "sharp_decay")
        delta_W = U @ torch.diag(sigma) @ V.T

        sigma_x_iso = torch.eye(d_in)
        sigma_x_aniso = generate_data_covariance(d_in, "exponential_decay", condition_number=100.0)

        U_iso, s_iso = compute_data_weighted_svd(delta_W, sigma_x_iso)
        U_aniso, s_aniso = compute_data_weighted_svd(delta_W, sigma_x_aniso)

        # The left singular vectors should differ
        G = U_iso[:, :4].T @ U_aniso[:, :4]
        cos_vals = torch.linalg.svdvals(G)
        alignment = (cos_vals ** 2).sum().item() / 4

        # Not perfectly aligned (data weighting rotates the subspace)
        assert alignment < 0.99, (
            f"Data-weighted subspace should differ from unweighted, "
            f"got alignment={alignment:.4f}"
        )
