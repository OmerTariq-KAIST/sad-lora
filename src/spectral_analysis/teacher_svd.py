"""Phase 1: Teacher Spectral Analysis (Algorithm 1).

Extracts and caches the truncated SVD of the teacher's weight updates
delta_W_T = W_T - W_0 for each adapted layer.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from torch import Tensor

from ..utils.svd_utils import truncated_svd

logger = logging.getLogger("sad_lora.spectral")


@dataclass
class LayerSpectralInfo:
    """Spectral decomposition data for a single layer."""

    layer_name: str
    U_T: Tensor          # (d_out, r_max) left singular vectors
    Sigma_T: Tensor      # (r_max,) singular values
    V_T: Tensor          # (d_in, r_max) right singular vectors
    cumulative_energy: Tensor  # (r_max,) cumulative energy ratio
    d_out: int
    d_in: int
    frobenius_norm: float


class TeacherSpectralAnalyzer:
    """Computes and caches spectral decomposition of teacher weight updates.

    Implements Algorithm 1: for each adapted layer, compute
    delta_W_T = W_T - W_0 and its truncated SVD.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        pretrained_model: nn.Module,
        layer_names: list[str],
        r_max: int = 64,
        device: str = "cuda",
        use_randomized_svd: bool = True,
        random_seed: int = 42,
    ):
        self.teacher_state = teacher_model.state_dict()
        self.pretrained_state = pretrained_model.state_dict()
        self.layer_names = layer_names
        self.r_max = r_max
        self.device = device
        self.use_randomized_svd = use_randomized_svd
        self.random_seed = random_seed

    def analyze(self) -> dict[str, LayerSpectralInfo]:
        """Compute SVD for all adapted layers.

        Processes one layer at a time to manage memory. All SVD computation
        is done in float32 for numerical stability.

        Returns:
            Dict mapping layer_name -> LayerSpectralInfo.
        """
        cache = {}
        for name in self.layer_names:
            logger.info("Computing SVD for layer: %s", name)

            # Resolve weight key — handle both "weight" suffix and direct match
            weight_key = name if name in self.teacher_state else f"{name}.weight"
            if weight_key not in self.teacher_state:
                raise KeyError(
                    f"Layer '{name}' not found in model state dict. "
                    f"Tried keys: '{name}', '{weight_key}'"
                )

            W_T = self.teacher_state[weight_key].float().to(self.device)
            W_0 = self.pretrained_state[weight_key].float().to(self.device)
            delta_W = W_T - W_0  # (d_out, d_in)

            d_out, d_in = delta_W.shape
            k = min(self.r_max, d_out, d_in)

            U, S, V = truncated_svd(
                delta_W,
                k=k,
                use_randomized=self.use_randomized_svd,
                seed=self.random_seed,
            )

            # Compute full Frobenius norm for energy calculations
            frob_norm_sq = (delta_W ** 2).sum().item()
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / max(frob_norm_sq, 1e-10)

            cache[name] = LayerSpectralInfo(
                layer_name=name,
                U_T=U.cpu(),
                Sigma_T=S.cpu(),
                V_T=V.cpu(),
                cumulative_energy=cumulative_energy.cpu(),
                d_out=d_out,
                d_in=d_in,
                frobenius_norm=frob_norm_sq ** 0.5,
            )

            # Free GPU memory
            del W_T, W_0, delta_W, U, S, V
            torch.cuda.empty_cache() if self.device == "cuda" else None

        logger.info("Spectral analysis complete for %d layers", len(cache))
        return cache

    @staticmethod
    def save_cache(cache: dict[str, LayerSpectralInfo], path: str) -> None:
        """Save spectral cache to disk using safetensors.

        Each layer's tensors are stored with prefixed keys.
        Metadata (dimensions, norms) stored separately.
        """
        tensors = {}
        metadata = {}
        for name, info in cache.items():
            prefix = name.replace(".", "_")
            tensors[f"{prefix}__U_T"] = info.U_T
            tensors[f"{prefix}__Sigma_T"] = info.Sigma_T
            tensors[f"{prefix}__V_T"] = info.V_T
            tensors[f"{prefix}__cumulative_energy"] = info.cumulative_energy
            metadata[f"{prefix}__d_out"] = str(info.d_out)
            metadata[f"{prefix}__d_in"] = str(info.d_in)
            metadata[f"{prefix}__frobenius_norm"] = str(info.frobenius_norm)

        metadata["layer_names"] = ",".join(cache.keys())
        save_file(tensors, path, metadata=metadata)
        logger.info("Saved spectral cache to %s", path)

    @staticmethod
    def load_cache(path: str) -> dict[str, LayerSpectralInfo]:
        """Load spectral cache from disk."""
        from safetensors import safe_open

        cache = {}
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            layer_names = metadata["layer_names"].split(",")

            for name in layer_names:
                prefix = name.replace(".", "_")
                cache[name] = LayerSpectralInfo(
                    layer_name=name,
                    U_T=f.get_tensor(f"{prefix}__U_T"),
                    Sigma_T=f.get_tensor(f"{prefix}__Sigma_T"),
                    V_T=f.get_tensor(f"{prefix}__V_T"),
                    cumulative_energy=f.get_tensor(f"{prefix}__cumulative_energy"),
                    d_out=int(metadata[f"{prefix}__d_out"]),
                    d_in=int(metadata[f"{prefix}__d_in"]),
                    frobenius_norm=float(metadata[f"{prefix}__frobenius_norm"]),
                )

        logger.info("Loaded spectral cache from %s (%d layers)", path, len(cache))
        return cache

    def get_energy_profile(
        self, cache: dict[str, LayerSpectralInfo], layer_name: str
    ) -> Tensor:
        """Return cumulative energy ratio for a given layer."""
        return cache[layer_name].cumulative_energy
