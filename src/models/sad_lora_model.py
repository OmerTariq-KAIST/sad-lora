"""SAD-LoRA Model Wrapper.

Wraps a HuggingFace transformer model, replacing specified linear layers
with SADLoRALinear modules and providing convenience methods for
spectral target registration, LoRA parameter collection, and diagnostics.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .sad_lora_layer import SADLoRALinear
from ..spectral_analysis.data_weighted_svd import TargetSubspace

logger = logging.getLogger("sad_lora.model")


class SADLoRAModel(nn.Module):
    """Wraps a pretrained model with SAD-LoRA adapters on specified layers.

    Usage:
        1. Create: model = SADLoRAModel(base_model, target_modules, rank)
        2. Set targets: model.set_target_subspaces(target_subspaces)
        3. Train: use model.lora_parameters() for the optimizer
        4. Diagnose: model.get_all_alignment_scores()
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: list[str],
        default_rank: int = 8,
        rank_per_layer: dict[str, int] | None = None,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        init_method: str = "kaiming",
    ):
        """
        Args:
            base_model: Pretrained HuggingFace model (will be frozen).
            target_modules: Substrings to match layer names for LoRA injection.
                e.g., ["query", "value"] matches all layers containing those strings.
            default_rank: Default LoRA rank when rank_per_layer doesn't specify.
            rank_per_layer: Optional {full_layer_name: rank} from auto rank selection.
            lora_alpha: LoRA scaling. None = alpha equals rank (no scaling).
            lora_dropout: Dropout on LoRA path.
            init_method: "kaiming", "spectral", or "random_subspace".
        """
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.init_method = init_method

        # Freeze entire base model
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        # Find and replace target linear layers with SADLoRALinear
        self.lora_layers: dict[str, SADLoRALinear] = {}
        self._inject_lora(
            default_rank=default_rank,
            rank_per_layer=rank_per_layer or {},
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_method=init_method,
        )

        logger.info(
            "Injected SAD-LoRA into %d layers. Total trainable: %s",
            len(self.lora_layers),
            _format_param_count(self.count_lora_parameters()),
        )

    def _inject_lora(
        self,
        default_rank: int,
        rank_per_layer: dict[str, int],
        lora_alpha: float | None,
        lora_dropout: float,
        init_method: str,
    ) -> None:
        """Replace matching nn.Linear modules with SADLoRALinear."""
        for full_name, module in list(self.base_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(target in full_name for target in self.target_modules):
                continue

            r = rank_per_layer.get(full_name, default_rank)

            sad_layer = SADLoRALinear(
                base_layer=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_method=init_method,
            )

            # Replace in the model tree
            _set_module(self.base_model, full_name, sad_layer)
            self.lora_layers[full_name] = sad_layer

    def forward(self, **kwargs: Any) -> Any:
        """Forward pass through the base model (with LoRA layers active)."""
        return self.base_model(**kwargs)

    def set_target_subspaces(
        self, target_subspaces: dict[str, TargetSubspace]
    ) -> None:
        """Register spectral targets from Phase 2 into all LoRA layers.

        Args:
            target_subspaces: {layer_name: TargetSubspace} from
                DataWeightedSubspaceEstimator.estimate().
        """
        matched = 0
        for name, layer in self.lora_layers.items():
            if name in target_subspaces:
                ts = target_subspaces[name]
                # Handle rank mismatch: truncate or warn
                if ts.r_star != layer.r:
                    logger.warning(
                        "Layer %s: rank mismatch (layer=%d, target=%d). "
                        "Truncating target to layer rank.",
                        name, layer.r, ts.r_star,
                    )
                r = min(layer.r, ts.r_star)
                U = ts.U_tilde[:, :r]
                sigma = ts.sigma_tilde[:r]

                # Pad if layer rank > target rank.
                # Use random orthonormal complement so spectral init gives L_align = 0.
                # Zero-padding was wrong: B = [U | 0] is rank-deficient and gives
                # L_align = 1 - r*/r even at initialization, overwhelming L_KD.
                if layer.r > ts.r_star:
                    pad_r = layer.r - ts.r_star
                    rand = torch.randn(U.shape[0], pad_r)
                    rand = rand - U @ (U.T @ rand)   # Gram-Schmidt orthogonalize vs U
                    Q_pad, _ = torch.linalg.qr(rand, mode="reduced")
                    U = torch.cat([U, Q_pad], dim=1)
                    sigma = torch.cat([sigma, torch.zeros(pad_r)])

                layer.set_target_subspace(U, sigma)
                matched += 1

        logger.info("Set spectral targets for %d/%d layers", matched, len(self.lora_layers))

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable LoRA parameters (B and A per layer)."""
        params = []
        for layer in self.lora_layers.values():
            params.extend([layer.lora_B, layer.lora_A])
        return params

    def count_lora_parameters(self) -> int:
        """Total number of trainable LoRA parameters."""
        return sum(p.numel() for p in self.lora_parameters())

    def get_lora_state_dict(self) -> dict[str, Tensor]:
        """Extract only LoRA weights for saving."""
        state = {}
        for name, layer in self.lora_layers.items():
            prefix = name.replace(".", "_")
            state[f"{prefix}.lora_B"] = layer.lora_B.data
            state[f"{prefix}.lora_A"] = layer.lora_A.data
        return state

    def get_all_lora_params(self) -> dict[str, dict[str, Tensor]]:
        """Return {layer_name: {'B': tensor, 'A': tensor}} for loss computation."""
        return {name: layer.get_lora_params() for name, layer in self.lora_layers.items()}

    def get_all_target_params(self) -> dict[str, dict[str, Tensor]]:
        """Return {layer_name: {'U_tilde': tensor, 'sigma_tilde': tensor}}."""
        return {
            name: layer.get_target_params()
            for name, layer in self.lora_layers.items()
            if layer.target_is_set
        }

    @torch.no_grad()
    def get_all_alignment_scores(self) -> dict[str, float]:
        """Per-layer alignment scores."""
        return {
            name: layer.get_alignment_score()
            for name, layer in self.lora_layers.items()
            if layer.target_is_set
        }

    @torch.no_grad()
    def get_all_principal_angles(self) -> dict[str, Tensor]:
        """Per-layer principal angles in radians."""
        return {
            name: layer.get_principal_angles()
            for name, layer in self.lora_layers.items()
            if layer.target_is_set
        }


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a submodule in the model tree by dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
