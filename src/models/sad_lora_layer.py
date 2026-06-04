"""Custom LoRA layer with spectral alignment targets.

Extends a frozen linear layer with trainable B, A matrices (LoRA) and
frozen spectral target buffers (U_target, sigma_target) from Phase 1-2.

Design decisions:
- lora_alpha = r (scaling=1.0) is recommended for SAD-LoRA because the
  coefficient loss explicitly controls adapter magnitude. Standard LoRA
  scaling (alpha/r) would interfere with sigma_i(BA) matching.
- U_target and sigma_target are registered as buffers: saved in state_dict,
  moved with .to(device), but NOT trained.
- QR decomposition is used (not SVD) for orthonormalizing colspan(B)
  because it has more stable gradients in PyTorch's autograd.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SADLoRALinear(nn.Module):
    """A frozen linear layer augmented with LoRA adapters and spectral targets.

    Forward: y = base_layer(x) + (B @ A)(x) * scaling

    The spectral targets (U_target, sigma_target) are set after Phase 2
    via set_target_subspace() and remain frozen during training.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        init_method: str = "kaiming",
    ):
        """
        Args:
            base_layer: Frozen pretrained linear layer.
            r: LoRA rank for this layer.
            lora_alpha: Scaling numerator. If None, defaults to r (scaling=1.0).
            lora_dropout: Dropout probability on LoRA path.
            init_method: Weight initialization strategy.
                "kaiming" — Kaiming uniform for B, zeros for A (standard LoRA).
                "random_subspace" — Random orthonormal B, zeros for A.
                "spectral" — Initialize B from U_target (call set_target_subspace first).
        """
        super().__init__()

        # Freeze the base layer
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if base_layer.bias is not None:
            base_layer.bias.requires_grad_(False)

        d_out, d_in = base_layer.weight.shape
        self.r = r
        self.d_out = d_out
        self.d_in = d_in

        # Scaling: default to alpha=r (no scaling) for SAD-LoRA
        if lora_alpha is None:
            lora_alpha = float(r)
        self.scaling = lora_alpha / r

        # Trainable LoRA parameters
        self.lora_B = nn.Parameter(torch.empty(d_out, r))
        self.lora_A = nn.Parameter(torch.empty(r, d_in))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Frozen spectral targets (registered as buffers)
        self.register_buffer("U_target", torch.zeros(d_out, r))
        self.register_buffer("sigma_target", torch.zeros(r))
        self._target_set = False

        self._init_method = init_method
        self._init_weights(init_method)

    def _init_weights(self, method: str) -> None:
        if method == "kaiming":
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A)
        elif method == "random_subspace":
            Q, _ = torch.linalg.qr(torch.randn(self.d_out, self.r))
            self.lora_B.data.copy_(Q)
            nn.init.zeros_(self.lora_A)
        elif method == "spectral":
            # Deferred — requires set_target_subspace() to be called first
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: base output + LoRA output.

        Args:
            x: (..., d_in) input tensor.

        Returns:
            (..., d_out) output tensor.
        """
        base_out = self.base_layer(x)
        # LoRA path: x @ A^T @ B^T = x -> (r,) -> (d_out,)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def set_target_subspace(self, U_tilde: Tensor, sigma_tilde: Tensor) -> None:
        """Register target subspace from Phase 2.

        If init_method="spectral", also reinitializes B from U_tilde.

        Args:
            U_tilde: (d_out, r) data-weighted left singular vectors.
            sigma_tilde: (r,) data-weighted singular values.
        """
        assert U_tilde.shape == (self.d_out, self.r), (
            f"U_tilde shape {U_tilde.shape} doesn't match ({self.d_out}, {self.r})"
        )
        assert sigma_tilde.shape == (self.r,), (
            f"sigma_tilde shape {sigma_tilde.shape} doesn't match ({self.r},)"
        )

        self.U_target.copy_(U_tilde)
        self.sigma_target.copy_(sigma_tilde)
        self._target_set = True

        if self._init_method == "spectral":
            self.lora_B.data.copy_(U_tilde)
            nn.init.zeros_(self.lora_A)

    @torch.no_grad()
    def get_alignment_score(self) -> float:
        """Current alignment score A(B, U_target) in [0, 1]."""
        Q_B, _ = torch.linalg.qr(self.lora_B.float(), mode="reduced")
        G = Q_B.T @ self.U_target.float()
        return (G ** 2).sum().item() / self.r

    @torch.no_grad()
    def get_principal_angles(self) -> Tensor:
        """Principal angles between colspan(B) and colspan(U_target) in radians."""
        Q_B, _ = torch.linalg.qr(self.lora_B.float(), mode="reduced")
        G = Q_B.T @ self.U_target.float()
        cos_angles = torch.linalg.svdvals(G).clamp(0.0, 1.0)
        return torch.arccos(cos_angles)

    @torch.no_grad()
    def get_adapter_singular_values(self) -> Tensor:
        """Singular values of B @ A (the full adapter)."""
        # Efficient: sigma_i(BA) = sigma_i(R_B @ A)
        _, R_B = torch.linalg.qr(self.lora_B.float(), mode="reduced")
        M = R_B @ self.lora_A.float()
        return torch.linalg.svdvals(M)

    @torch.no_grad()
    def get_intruder_dimension_count(
        self, U_teacher_full: Tensor, threshold: float = 0.1
    ) -> int:
        """Count intruder dimensions following Shuttleworth et al. (2024).

        An intruder dimension is a singular vector of BA that has low
        cosine similarity with ALL top singular vectors of the teacher update.

        Args:
            U_teacher_full: (d_out, k) top-k left singular vectors of delta_W_T.
            threshold: Below this max cosine similarity = intruder.

        Returns:
            Count of intruder dimensions in [0, r].
        """
        BA = self.lora_B.float() @ self.lora_A.float()
        U_adapter, _, _ = torch.linalg.svd(BA, full_matrices=False)
        U_adapter = U_adapter[:, : self.r]

        # (r, k) matrix of absolute cosine similarities
        cos_sim = (U_adapter.T @ U_teacher_full.float()).abs()
        max_sim_per_direction = cos_sim.max(dim=1).values
        return (max_sim_per_direction < threshold).sum().item()

    @torch.no_grad()
    def get_adapter_svd(self) -> tuple[Tensor, Tensor, Tensor]:
        """Full SVD of the current B @ A product."""
        BA = self.lora_B.float() @ self.lora_A.float()
        U, S, Vh = torch.linalg.svd(BA, full_matrices=False)
        return U[:, : self.r], S[: self.r], Vh[: self.r, :]

    def get_lora_params(self) -> dict[str, Tensor]:
        """Return B and A tensors for loss computation."""
        return {"B": self.lora_B, "A": self.lora_A}

    def get_target_params(self) -> dict[str, Tensor]:
        """Return frozen target tensors."""
        return {"U_tilde": self.U_target, "sigma_tilde": self.sigma_target}

    @property
    def target_is_set(self) -> bool:
        return self._target_set
