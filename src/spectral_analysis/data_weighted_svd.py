"""Phase 2: Data-Weighted Subspace Estimation (Algorithm 2).

Computes the SVD of the data-weighted teacher update delta_W_T @ Sigma_x^{1/2}
and determines per-layer target subspaces for SAD-LoRA training.
"""

import logging
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .teacher_svd import LayerSpectralInfo
from .covariance import CovarianceEstimator
from .rank_selection import RankSelector
from ..utils.svd_utils import truncated_svd, reconstruct_from_svd

logger = logging.getLogger("sad_lora.spectral")


@dataclass
class TargetSubspace:
    """Target subspace data for one layer, output of Phase 2."""

    layer_name: str
    U_tilde: Tensor        # (d_out, r_star) data-weighted left singular vectors
    sigma_tilde: Tensor    # (r_star,) data-weighted singular values
    r_star: int            # Automatically determined rank
    spectral_gap: float    # sigma_tilde[r-1] / sigma_tilde[r]
    energy_captured: float  # Fraction of total energy in top r_star components


class DataWeightedSubspaceEstimator:
    """Estimates data-weighted teacher subspace via calibration data.

    Implements Algorithm 2: collect activations, estimate covariance,
    compute SVD of delta_W_T @ Sigma_x^{1/2} per layer.
    """

    def __init__(
        self,
        spectral_cache: dict[str, LayerSpectralInfo],
        student_model: nn.Module,
        layer_names: list[str],
        n_calibration: int = 1024,
        energy_threshold: float = 0.01,
        covariance_method: str = "empirical_shrinkage",
        use_implicit_svd: bool = True,
        r_max: int = 64,
        min_rank: int = 1,
        max_rank: int = 64,
        device: str = "cuda",
    ):
        self.spectral_cache = spectral_cache
        self.student_model = student_model
        self.layer_names = layer_names
        self.n_calibration = n_calibration
        self.covariance_method = covariance_method
        self.use_implicit_svd = use_implicit_svd
        self.r_max = r_max
        self.device = device

        self.cov_estimator = CovarianceEstimator(method=covariance_method)
        self.rank_selector = RankSelector(
            energy_threshold=energy_threshold,
            min_rank=min_rank,
            max_rank=max_rank,
        )

    @torch.no_grad()
    def estimate(
        self,
        calibration_loader: DataLoader,
    ) -> dict[str, TargetSubspace]:
        """Run calibration and compute target subspaces.

        Args:
            calibration_loader: DataLoader yielding dicts with 'input_ids'
                and 'attention_mask' keys (standard HuggingFace format).

        Returns:
            Dict mapping layer_name -> TargetSubspace.
        """
        self.student_model = self.student_model.to(self.device)
        self.student_model.eval()

        # Step 1: Collect activations at each adapted layer via forward hooks
        activation_cache: dict[str, list[Tensor]] = {n: [] for n in self.layer_names}
        hooks = []

        for name in self.layer_names:
            module = _get_module(self.student_model, name)
            hook = module.register_forward_hook(
                partial(self._activation_hook, layer_name=name, cache=activation_cache)
            )
            hooks.append(hook)

        # Forward pass over calibration set
        n_collected = 0
        for batch in calibration_loader:
            if n_collected >= self.n_calibration:
                break
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, Tensor)}
            self.student_model(**batch)
            n_collected += batch[next(iter(batch))].shape[0]

        for h in hooks:
            h.remove()

        # Step 2: Compute data-weighted SVD per layer
        target_subspaces = {}
        sigma_tilde_per_layer = {}
        U_tilde_per_layer = {}

        for name in self.layer_names:
            logger.info("Computing data-weighted SVD for layer: %s", name)
            info = self.spectral_cache[name]

            # Stack activations: (n_cal, d_in)
            X = torch.cat(activation_cache[name], dim=0)[:self.n_calibration].float()
            X = X.to(self.device)

            # Reshape if needed (e.g., from (n, seq_len, d) to (n*seq_len, d))
            if X.dim() == 3:
                X = X.reshape(-1, X.shape[-1])

            # Reconstruct delta_W_T from cached SVD
            delta_W = reconstruct_from_svd(
                info.U_T.to(self.device),
                info.Sigma_T.to(self.device),
                info.V_T.to(self.device),
            )

            d_out, d_in = delta_W.shape
            n_cal = X.shape[0]

            # Choose between explicit and implicit SVD
            if self.use_implicit_svd and n_cal < d_in:
                # Option B: Implicit — avoid forming d_in x d_in covariance
                # Z = delta_W @ X_centered^T / sqrt(n-1), shape (d_out, n_cal)
                X_centered = X - X.mean(dim=0)
                Z = delta_W @ X_centered.T / max(n_cal - 1, 1) ** 0.5

                U_tilde, sigma_tilde, _ = truncated_svd(
                    Z, k=self.r_max, use_randomized=True
                )
            else:
                # Option A: Explicit covariance
                cov = self.cov_estimator.estimate(X)
                cov_sqrt = self.cov_estimator.compute_sqrt(cov)

                # Data-weighted teacher update: delta_W @ Sigma_x^{1/2}
                W_tilde = delta_W @ cov_sqrt

                U_tilde, sigma_tilde, _ = truncated_svd(
                    W_tilde, k=self.r_max, use_randomized=True
                )

            # Store both U_tilde and sigma_tilde from the same SVD pass.
            # Slicing U_tilde[:, :r] in step 3 keeps the covariance path
            # consistent — no need to recompute U_tilde separately.
            U_tilde_per_layer[name] = U_tilde.cpu()
            sigma_tilde_per_layer[name] = sigma_tilde.cpu()

            # Clean up
            del X, delta_W
            activation_cache[name].clear()

        # Step 3: Select per-layer ranks
        rank_allocation = self.rank_selector.select_ranks(sigma_tilde_per_layer)

        # Build final target subspaces
        for name in self.layer_names:
            r = rank_allocation[name]
            sigma = sigma_tilde_per_layer[name]
            total_energy = (sigma ** 2).sum().item()
            captured = (sigma[:r] ** 2).sum().item() / max(total_energy, 1e-10)
            gap = self.rank_selector.compute_spectral_gap(sigma, r)

            # Slice the already-computed U_tilde to rank r.
            # This uses the same covariance path as sigma_tilde (no recompute).
            U_tilde_r = U_tilde_per_layer[name][:, :r]

            target_subspaces[name] = TargetSubspace(
                layer_name=name,
                U_tilde=U_tilde_r.cpu(),
                sigma_tilde=sigma[:r].cpu(),
                r_star=r,
                spectral_gap=gap,
                energy_captured=captured,
            )

        logger.info(
            "Data-weighted subspace estimation complete. "
            "Rank allocation: %s",
            {n: ts.r_star for n, ts in target_subspaces.items()},
        )
        return target_subspaces

    def _recompute_subspace(
        self, layer_name: str, calibration_loader: DataLoader, r: int
    ) -> Tensor:
        """Recompute U_tilde[:, :r] for a specific layer and rank.

        This is a lightweight re-run using the cached spectral info.
        """
        info = self.spectral_cache[layer_name]
        delta_W = reconstruct_from_svd(
            info.U_T.to(self.device),
            info.Sigma_T.to(self.device),
            info.V_T.to(self.device),
        )

        # Collect activations again (cheap — single forward pass)
        activations = []
        hooks = []
        module = _get_module(self.student_model, layer_name)
        hook = module.register_forward_hook(
            partial(self._activation_hook, layer_name="tmp", cache={"tmp": activations})
        )

        n_collected = 0
        with torch.no_grad():
            for batch in calibration_loader:
                if n_collected >= self.n_calibration:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, Tensor)}
                self.student_model(**batch)
                n_collected += batch[next(iter(batch))].shape[0]
        hook.remove()

        X = torch.cat(activations, dim=0)[:self.n_calibration].float().to(self.device)
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])

        X_centered = X - X.mean(dim=0)
        n_cal = X_centered.shape[0]
        Z = delta_W @ X_centered.T / max(n_cal - 1, 1) ** 0.5
        U_tilde, _, _ = truncated_svd(Z, k=r, use_randomized=True)

        del X, X_centered, Z, delta_W
        return U_tilde

    @staticmethod
    def _activation_hook(
        module: nn.Module,
        input: tuple,
        output: Tensor,
        layer_name: str,
        cache: dict[str, list[Tensor]],
    ) -> None:
        """Forward hook to capture layer input activations."""
        # input is a tuple; first element is the actual input tensor
        act = input[0].detach().cpu()
        cache[layer_name].append(act)

    def get_rank_allocation(
        self, target_subspaces: dict[str, TargetSubspace]
    ) -> dict[str, int]:
        """Return the rank allocation from estimated subspaces."""
        return {name: ts.r_star for name, ts in target_subspaces.items()}

    def get_spectral_gaps(
        self, target_subspaces: dict[str, TargetSubspace]
    ) -> dict[str, float]:
        """Return per-layer spectral gaps."""
        return {name: ts.spectral_gap for name, ts in target_subspaces.items()}


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Retrieve a submodule by dot-separated name string."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module
