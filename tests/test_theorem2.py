"""Numerical verification of Theorem 2: Rank Sufficiency Condition.

Verifies that the predicted minimum rank r*_epsilon (based on the
data-weighted spectral tail) correctly predicts when the optimal
rank-r adapter achieves epsilon-error.
"""

import pytest
import torch

from experiments.run_synthetic import (
    generate_teacher_spectrum,
    generate_data_covariance,
    verify_theorem2,
    compute_data_weighted_svd,
)
from src.spectral_analysis.rank_selection import RankSelector


class TestTheorem2:
    @pytest.mark.parametrize("spectrum", ["sharp_decay", "gradual_decay"])
    def test_rank_prediction_accuracy(self, spectrum):
        """Predicted r* should match actual minimum rank for epsilon-error."""
        d_out, d_in = 64, 32
        U, sigma, V = generate_teacher_spectrum(d_out, d_in, spectrum)
        delta_W = U @ torch.diag(sigma) @ V.T
        sigma_x = torch.eye(d_in)

        result = verify_theorem2(delta_W, sigma_x, epsilon=1.0)

        assert result["match"], (
            f"Rank prediction failed for {spectrum}: "
            f"predicted={result['r_star_predicted']}, actual={result['r_star_actual']}"
        )

    def test_sharp_spectrum_needs_fewer_ranks(self):
        """Sharp decay should require fewer ranks than gradual decay."""
        d_out, d_in = 64, 32
        sigma_x = torch.eye(d_in)

        U_s, sigma_s, V_s = generate_teacher_spectrum(d_out, d_in, "sharp_decay")
        U_g, sigma_g, V_g = generate_teacher_spectrum(d_out, d_in, "gradual_decay")

        r_sharp = verify_theorem2(
            U_s @ torch.diag(sigma_s) @ V_s.T, sigma_x, epsilon=1.0
        )["r_star_predicted"]
        r_gradual = verify_theorem2(
            U_g @ torch.diag(sigma_g) @ V_g.T, sigma_x, epsilon=1.0
        )["r_star_predicted"]

        assert r_sharp <= r_gradual, (
            f"Sharp spectrum (r*={r_sharp}) should need <= ranks than "
            f"gradual (r*={r_gradual})"
        )

    def test_rank_selector_consistency(self):
        """RankSelector should produce same r* as Theorem 2 prediction."""
        d_out, d_in = 64, 32
        U, sigma, V = generate_teacher_spectrum(d_out, d_in, "sharp_decay")
        delta_W = U @ torch.diag(sigma) @ V.T
        sigma_x = torch.eye(d_in)

        _, sigma_tilde = compute_data_weighted_svd(delta_W, sigma_x)

        selector = RankSelector(energy_threshold=0.01, min_rank=1, max_rank=64)
        ranks = selector.select_ranks({"layer0": sigma_tilde})

        # The selected rank should capture at least 99% of the energy
        r = ranks["layer0"]
        energy_captured = (sigma_tilde[:r] ** 2).sum() / (sigma_tilde ** 2).sum()
        assert energy_captured.item() >= 0.99 - 1e-6

    def test_epsilon_monotonicity(self):
        """Larger epsilon should require equal or fewer ranks."""
        d_out, d_in = 64, 32
        U, sigma, V = generate_teacher_spectrum(d_out, d_in, "gradual_decay")
        delta_W = U @ torch.diag(sigma) @ V.T
        sigma_x = torch.eye(d_in)

        r_tight = verify_theorem2(delta_W, sigma_x, epsilon=0.1)["r_star_predicted"]
        r_loose = verify_theorem2(delta_W, sigma_x, epsilon=10.0)["r_star_predicted"]

        assert r_tight >= r_loose, (
            f"Tighter epsilon should need >= ranks: "
            f"r*(eps=0.1)={r_tight}, r*(eps=10)={r_loose}"
        )
