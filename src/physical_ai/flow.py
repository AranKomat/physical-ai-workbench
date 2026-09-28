"""Reference rectified flow. t=0 is noise, t=1 is data, v=data-noise.
Donor schedulers may use the OPPOSITE sign/time or nonlinear time shift. Conversion
must be explicit and qualified. A deterministic ODE sampler has no cheap action
log-probability; this module does not pretend to provide PPO/GRPO likelihoods.
"""
from __future__ import annotations
import torch
from torch import Tensor


def require_action_tensors(action: Tensor, mask: Tensor) -> None:
    if action.ndim != 3 or mask.shape != action.shape or mask.dtype != torch.bool:
        raise ValueError("action/mask must be [B,H,D] and mask bool")
    if not bool(torch.isfinite(action).all()) or not bool(mask.flatten(1).any(1).all()):
        raise ValueError("nonfinite actions or examples with zero active targets")


def flow_sample(actions: Tensor, mask: Tensor, noise: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    require_action_tensors(actions, mask)
    if noise.shape != actions.shape or time.shape != (actions.shape[0],):
        raise ValueError("noise/time shape mismatch")
    if not bool(torch.isfinite(noise).all()) or not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
        raise ValueError("invalid noise/time")
    time = time[:, None, None].to(actions.dtype)
    x = ((1 - time) * noise + time * actions).masked_fill(~mask, 0)
    v = (actions - noise).masked_fill(~mask, 0)
    return x, v


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean active dimensions per example, then mean examples. Padding has no weight."""
    require_action_tensors(target, mask)
    if pred.shape != target.shape or not bool(torch.isfinite(pred).all()):
        raise ValueError("nonfinite predictions or shape mismatch")
    error = (pred.float() - target.float()).square().masked_fill(~mask, 0)
    denom = mask.flatten(1).sum(1)
    return (error.flatten(1).sum(1) / denom).mean()


@torch.no_grad()
def euler_sample(velocity, noise: Tensor, mask: Tensor, steps: int, *, return_trace: bool = False):
    require_action_tensors(noise, mask)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    x = noise.clone().masked_fill(~mask, 0)
    trace = [x.clone()]
    for i in range(steps):
        t = torch.full((len(x),), i / steps, device=x.device, dtype=torch.float32)
        v = velocity(x, t)
        if v.shape != x.shape or not bool(torch.isfinite(v).all()):
            raise ValueError("invalid flow velocity")
        x = (x + v.to(x.dtype) / steps).masked_fill(~mask, 0)
        if return_trace:
            trace.append(x.clone())
    return (x, torch.stack(trace)) if return_trace else x
