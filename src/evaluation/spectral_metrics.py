"""Spectral evaluation metrics for trained LoRA adapters.

Implements all spectral analysis needed for Experiment 3 and general diagnostics:
- Principal angles and alignment scores
- Intruder dimension counting (Shuttleworth et al. methodology)
- Error decomposition verification (Theorem 1)
- Effective rank computation
- Spectral overlap matrices for visualization
"""

import logging
from typing import Any

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.sad_lora_model import SADLoRAModel
from ..spectral_analysis.teacher_svd import LayerSpectralInfo
from ..spectral_analysis.data_weighted_svd import TargetSubspace
from ..utils.grassmannian import principal_angles, alignment_score, spectral_overlap_matrix

logger = logging.getLogger("sad_lora.evaluation")


class SpectralMetrics:
    """Computes comprehensive spectral metrics for trained LoRA adapters.

    Provides per-layer and aggregate metrics across four evaluation axes:
    alignment, intruder dimensions, error decomposition, and effective rank.
    """

    def __init__(
        self,
        model: SADLoRAModel,
        teacher_spectral_cache: dict[str, LayerSpectralInfo],
        target_subspaces: dict[str, TargetSubspace],
        intruder_threshold: float = 0.1,
    ):
        self.model = model
        self.teacher_cache = teacher_spectral_cache
        self.targets = target_subspaces
        self.intruder_threshold = intruder_threshold

    @torch.no_grad()
    def compute_all_metrics(self) -> dict[str, Any]:
        """Compute full spectral analysis for all LoRA layers.

        Returns:
            {
                'per_layer': {layer_name: {metric_name: value}},
                'aggregate': {metric_name: value}
            }
        """
        per_layer = {}
        all_alignments = []
        total_intruders = 0
        all_eff_ranks = []

        for name, layer in self.model.lora_layers.items():
            if not layer.target_is_set or name not in self.teacher_cache:
                continue

            info = self.teacher_cache[name]
            target = self.targets.get(name)

            # Alignment
            align = layer.get_alignment_score()
            angles = layer.get_principal_angles()

            # Intruder dimensions
            U_teacher = info.U_T.to(layer.lora_B.device)
            intruders = layer.get_intruder_dimension_count(U_teacher, self.intruder_threshold)

            # Adapter singular values
            sigma_student = layer.get_adapter_singular_values()
            sigma_target = layer.sigma_target

            # Effective rank
            eff_rank = self._effective_rank(sigma_student)

            # Spectral overlap matrix
            U_adapter, _, _ = layer.get_adapter_svd()
            overlap = spectral_overlap_matrix(U_adapter, U_teacher[:, : layer.r])

            # Singular value correlation
            r = min(len(sigma_student), len(sigma_target))
            sigma_corr = self._pearson_correlation(
                sigma_student[:r].cpu().numpy(),
                sigma_target[:r].cpu().numpy(),
            )

            per_layer[name] = {
                "alignment_score": align,
                "principal_angles": angles.tolist(),
                "mean_principal_angle": angles.mean().item(),
                "intruder_count": intruders,
                "effective_rank": eff_rank,
                "sigma_student": sigma_student.tolist(),
                "sigma_target": sigma_target.tolist(),
                "sigma_mse": ((sigma_student[:r] - sigma_target[:r].to(sigma_student.device)) ** 2).mean().item(),
                "sigma_correlation": sigma_corr,
                "spectral_overlap": overlap.cpu(),
            }

            all_alignments.append(align)
            total_intruders += intruders
            all_eff_ranks.append(eff_rank)

        aggregate = {
            "mean_alignment": np.mean(all_alignments) if all_alignments else 0.0,
            "total_intruder_dims": total_intruders,
            "mean_effective_rank": np.mean(all_eff_ranks) if all_eff_ranks else 0.0,
        }

        return {"per_layer": per_layer, "aggregate": aggregate}

    @torch.no_grad()
    def compute_error_decomposition(
        self,
        data_loader: DataLoader,
        device: str = "cuda",
    ) -> dict[str, dict[str, float]]:
        """Empirical verification of Theorem 1 error decomposition.

        For each layer, computes:
        - Term I: subspace misalignment
        - Term II: coefficient mismatch
        - Term III: rank residual (irreducible)
        - Total error and decomposition residual (should be ~0 for linear case)

        Args:
            data_loader: Calibration data for computing expectations.
            device: Computation device.

        Returns:
            {layer_name: {term_I, term_II, term_III, total, residual}}.
        """
        results = {}

        for name, layer in self.model.lora_layers.items():
            if name not in self.teacher_cache or name not in self.targets:
                continue

            info = self.teacher_cache[name]
            target = self.targets[name]
            r = layer.r

            # Get current adapter product BA
            BA = (layer.lora_B @ layer.lora_A).float().to(device)

            # Reconstruct delta_W_T from cache
            delta_W = (
                info.U_T[:, :].to(device)
                @ torch.diag(info.Sigma_T.to(device))
                @ info.V_T[:, :].T.to(device)
            )

            # Data-weighted teacher subspace
            U_tilde = target.U_tilde.to(device)  # (d_out, r_star)
            sigma_tilde = target.sigma_tilde.to(device)  # (r_star,)

            # Adapter column space basis
            Q_B, R_B = torch.linalg.qr(layer.lora_B.float().to(device), mode="reduced")

            # Principal angles between colspan(B) and teacher subspace
            G = Q_B.T @ U_tilde[:, :r]
            cos_vals = torch.linalg.svdvals(G).clamp(0.0, 1.0)
            sin2_vals = 1.0 - cos_vals ** 2

            # Term I: sum sigma_tilde_i^2 * sin^2(theta_i)
            s2 = sigma_tilde[:r] ** 2
            term_I = (s2 * sin2_vals[:r]).sum().item()

            # Term II: coefficient mismatch within aligned subspace
            # sigma_i(BA) projected onto the aligned directions
            M = R_B @ layer.lora_A.float().to(device)
            sigma_student = torch.linalg.svdvals(M)[:r]
            term_II = (s2 * cos_vals[:r] ** 2 * (1.0 - sigma_student / sigma_tilde[:r].clamp(min=1e-10)) ** 2).sum().item()

            # Term III: rank residual (energy beyond rank r)
            total_energy = (info.Sigma_T.to(device) ** 2).sum().item()
            top_r_energy = (sigma_tilde[:r] ** 2).sum().item()
            term_III = max(total_energy - top_r_energy, 0.0)

            # Total actual error: ||delta_W - BA||_F^2 (unweighted for simplicity)
            total_error = ((delta_W - BA) ** 2).sum().item()

            decomposition_sum = term_I + term_II + term_III
            residual = abs(total_error - decomposition_sum) / max(total_error, 1e-10)

            results[name] = {
                "term_I_misalignment": term_I,
                "term_II_coefficient": term_II,
                "term_III_residual": term_III,
                "total_error": total_error,
                "decomposition_sum": decomposition_sum,
                "decomposition_residual": residual,
            }

            del BA, delta_W

        return results

    @staticmethod
    def _effective_rank(sigma: Tensor) -> float:
        """Shannon entropy-based effective rank.

        eff_rank = exp(-sum p_i log p_i) where p_i = sigma_i^2 / sum sigma_j^2.
        """
        sigma = sigma.float()
        energy = sigma ** 2
        total = energy.sum()
        if total < 1e-10:
            return 0.0
        p = energy / total
        p = p[p > 1e-10]  # avoid log(0)
        entropy = -(p * p.log()).sum().item()
        return float(np.exp(entropy))

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation between two arrays."""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum()))
        if denom < 1e-10:
            return 0.0
        return float((x * y).sum() / denom)
