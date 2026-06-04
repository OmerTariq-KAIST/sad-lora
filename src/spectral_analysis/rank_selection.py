"""Theorem 2 implementation: data-dependent per-layer rank selection.

Determines the minimum LoRA rank r* per layer such that the
rank residual (Term III) is below a threshold epsilon.
"""

import logging

import torch
from torch import Tensor

logger = logging.getLogger("sad_lora.spectral")


class RankSelector:
    """Selects per-layer LoRA ranks based on spectral energy threshold.

    From Theorem 2: r >= r*_eps iff sum_{i>r} sigma_tilde_i^2 < eps,
    where sigma_tilde_i are the data-weighted singular values.
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        min_rank: int = 1,
        max_rank: int = 64,
        total_rank_budget: int | None = None,
    ):
        """
        Args:
            energy_threshold: Epsilon — fraction of total energy allowed in residual.
                r* = min{k : cumulative_energy(k) >= 1 - eps}.
            min_rank: Floor for any layer's rank.
            max_rank: Ceiling for any layer's rank.
            total_rank_budget: If set, redistribute ranks to stay within budget
                while respecting min/max per layer.
        """
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.total_rank_budget = total_rank_budget

    def select_ranks(
        self,
        sigma_tilde_per_layer: dict[str, Tensor],
    ) -> dict[str, int]:
        """Determine optimal rank for each layer.

        Args:
            sigma_tilde_per_layer: {layer_name: (r_max,) data-weighted singular values}.

        Returns:
            {layer_name: r_star} rank allocation.
        """
        ranks = {}
        spectral_gaps = {}

        for name, sigma in sigma_tilde_per_layer.items():
            sigma = sigma.float()
            total_energy = (sigma ** 2).sum().item()

            if total_energy < 1e-12:
                ranks[name] = self.min_rank
                spectral_gaps[name] = 1.0
                continue

            cumulative = torch.cumsum(sigma ** 2, dim=0) / total_energy
            # Find smallest k where cumulative[k-1] >= 1 - eps
            threshold = 1.0 - self.energy_threshold
            above = (cumulative >= threshold).nonzero(as_tuple=True)[0]

            if len(above) > 0:
                r_star = above[0].item() + 1  # +1 for 1-indexed rank
            else:
                r_star = len(sigma)

            r_star = max(self.min_rank, min(self.max_rank, r_star))
            ranks[name] = r_star

            # Compute spectral gap gamma = sigma_r / sigma_{r+1}
            if r_star < len(sigma) and sigma[r_star].item() > 1e-10:
                spectral_gaps[name] = sigma[r_star - 1].item() / sigma[r_star].item()
            else:
                spectral_gaps[name] = float("inf")

        # Optionally enforce total budget via greedy reallocation
        if self.total_rank_budget is not None:
            ranks = self._enforce_budget(ranks, sigma_tilde_per_layer, spectral_gaps)

        for name, r in ranks.items():
            logger.info(
                "Layer %s: r*=%d, spectral_gap=%.2f", name, r, spectral_gaps.get(name, 0)
            )

        return ranks

    def _enforce_budget(
        self,
        ranks: dict[str, int],
        sigma_per_layer: dict[str, Tensor],
        spectral_gaps: dict[str, float],
    ) -> dict[str, int]:
        """Greedily reduce ranks to meet total budget.

        Prioritize cutting from layers with smallest spectral gap
        (they benefit least from additional rank).
        """
        total = sum(ranks.values())
        if total <= self.total_rank_budget:
            return ranks

        # Sort layers by spectral gap ascending — cut from flat-spectrum layers first
        layer_order = sorted(spectral_gaps.keys(), key=lambda k: spectral_gaps[k])

        deficit = total - self.total_rank_budget
        for name in layer_order:
            if deficit <= 0:
                break
            reduction = min(ranks[name] - self.min_rank, deficit)
            ranks[name] -= reduction
            deficit -= reduction

        return ranks

    def compute_spectral_gap(self, sigma: Tensor, rank: int) -> float:
        """Compute spectral gap gamma = sigma_r / sigma_{r+1} for a layer."""
        if rank >= len(sigma) or sigma[rank].item() < 1e-10:
            return float("inf")
        return sigma[rank - 1].item() / sigma[rank].item()
