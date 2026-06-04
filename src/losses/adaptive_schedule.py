"""Adaptive scheduling for spectral loss weights alpha and beta.

From Corollary 3: early training should prioritize subspace alignment
(high alpha, low beta), then transition to coefficient matching
(lower alpha, higher beta) as alignment improves.

Schedule: alpha_t = alpha_0 * progress, beta_t = beta_0 * (1 - progress)
where progress = L_align(t) / L_align(0) decreases from 1 toward 0.
"""


class AdaptiveSchedule:
    """Manages adaptive alpha/beta scheduling based on alignment progress."""

    def __init__(
        self,
        alpha_0: float = 1.0,
        beta_0: float = 0.1,
        enabled: bool = True,
        warmup_steps: int = 100,
    ):
        """
        Args:
            alpha_0: Initial/max alignment loss weight.
            beta_0: Initial/max coefficient loss weight.
            enabled: If False, return constant alpha_0, beta_0.
            warmup_steps: Steps before adaptive schedule kicks in.
                During warmup, alpha=alpha_0, beta=0 (pure alignment focus).
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.enabled = enabled
        self.warmup_steps = warmup_steps

        self._initial_align_loss: float | None = None

    def record_initial_alignment(self, align_loss: float) -> None:
        """Record L_align(0) for computing progress ratio."""
        self._initial_align_loss = max(align_loss, 1e-8)

    def get_weights(self, current_align_loss: float, step: int) -> tuple[float, float]:
        """Compute current alpha_t and beta_t.

        Args:
            current_align_loss: Current L_align value.
            step: Current training step.

        Returns:
            (alpha_t, beta_t) weight tuple.
        """
        if not self.enabled:
            return self.alpha_0, self.beta_0

        # Warmup: pure alignment, no coefficient matching
        if step < self.warmup_steps:
            return self.alpha_0, 0.0

        if self._initial_align_loss is None:
            self._initial_align_loss = max(current_align_loss, 1e-8)

        # progress in (0, 1]: how much alignment loss remains relative to initial
        progress = min(1.0, max(0.0, current_align_loss / self._initial_align_loss))

        alpha_t = self.alpha_0 * progress
        beta_t = self.beta_0 * (1.0 - progress)

        return alpha_t, beta_t
