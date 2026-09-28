"""A1.5 native-cache boundary, deliberately not a generic W0 motor adapter."""
import torch
from torch import nn


class A15NativeCacheAdapter(nn.Module):
    """Keep the donor's prefix-cache and native action interface unchanged.

    Callers must obtain caches from a prompt-only native A1.5 prefix pass.
    Token IDs are checked for action-answer leakage; this does not attest that
    an externally supplied cache actually came from those IDs.
    """
    def __init__(self, native_model, *, horizon=50, action_dim=32):
        super().__init__()
        self.native_model = native_model
        self.horizon, self.action_dim = horizon, action_dim

    def forward(self, *, state, prefix_ids, prefix_mask, cache, max_prefix_position_ids, noisy_actions, native_time):
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("A1.5 native action shape mismatch")
        if state.shape != (len(noisy_actions), self.action_dim):
            raise ValueError("A1.5 native state shape mismatch")
        if prefix_ids.shape != prefix_mask.shape or prefix_ids.shape[0] != len(noisy_actions):
            raise ValueError("A1.5 native prefix shape mismatch")
        if ((prefix_ids >= 248077) & (prefix_ids <= 250124) & prefix_mask.bool()).any():
            raise ValueError("action-answer tokens forbidden in native motor prefix")
        if cache is None or native_time.shape != (len(noisy_actions),):
            raise ValueError("native cache and per-sample times required")
        if not all(torch.isfinite(x).all() for x in (state, noisy_actions, native_time)):
            raise ValueError("nonfinite native input")
        if ((native_time < 0) | (native_time > 1)).any():
            raise ValueError("native flow time must lie in [0,1]")
        velocity, _ = self.native_model.denoise_step_full(state, prefix_mask, cache,
            max_prefix_position_ids, noisy_actions, native_time, fast_mask=None)
        if velocity.shape != noisy_actions.shape or not torch.isfinite(velocity).all():
            raise ValueError("invalid native velocity")
        return velocity
