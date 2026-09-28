"""Reference training utilities. Production sharding/FP8 is qualified separately."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .checkpoints import atomic_torch_save
from .models import ReferenceConfig, ReferenceVLA, byte_tokens
from .io import write_json, write_jsonl


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def motor_parameter_role(name: str) -> str:
    if name.startswith(("action_in.", "output.", "state_in.", "mask_in.", "context_in.")):
        return "interface"
    if ".cross_attn." in name or ".norm" in name or name.startswith("time_in."):
        return "conditioning"
    return "trunk"


def set_motor_stage(motor: nn.Module, stage: str) -> dict:
    allowed = {"interfaces": {"interface"}, "conditioning": {"interface", "conditioning"},
               "all": {"interface", "conditioning", "trunk"}}
    if stage not in allowed:
        raise ValueError("unknown adaptation stage")
    counts = {"interface": 0, "conditioning": 0, "trunk": 0}
    for name, p in motor.named_parameters():
        role = motor_parameter_role(name)
        p.requires_grad_(role in allowed[stage])
        if not p.requires_grad:
            p.grad = None
        counts[role] += p.numel()
    return {"stage": stage, "parameters_by_role": counts,
            "trainable": sum(p.numel() for p in motor.parameters() if p.requires_grad)}


def optimizer_groups(model: ReferenceVLA, new_lr: float = 1e-3, pretrained_lr: float = 1e-4, backbone_lr: float = 1e-4):
    groups: dict[tuple[str, bool], list] = {}
    for name, p in model.named_parameters():
        if name.startswith("brain."):
            role = "backbone"
        elif name.startswith("motor.") and motor_parameter_role(name[len("motor."):]) != "interface":
            role = "pretrained_motor"
        else:
            role = "new"
        decay = p.ndim >= 2 and not name.endswith("bias")
        groups.setdefault((role, decay), []).append(p)
    rates = {"backbone": backbone_lr, "pretrained_motor": pretrained_lr, "new": new_lr}
    # Include frozen params now: optimizer skips them with grad=None. Later
    # unfreezing won't require resetting moments on already-trained interfaces.
    return [{"params": ps, "lr": rates[role], "weight_decay": 0.01 if decay else 0.0,
             "name": f"{role}.{'decay' if decay else 'no_decay'}"} for (role, decay), ps in groups.items()]


class LoRALinear(nn.Module):
    """Optional parameter-efficient motor adaptation; zero-initialized residual."""
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        if rank < 1 or rank > min(base.in_features, base.out_features):
            raise ValueError("invalid LoRA rank")
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.multiplier = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.multiplier


def synthetic_batch(cfg: ReferenceConfig, seed: int = 0, batch_size: int = 4, device: str = "cpu") -> dict[str, Tensor]:
    """Manufactured learnability fixture. It is NOT a robot benchmark or real data."""
    g = torch.Generator().manual_seed(seed)
    state = torch.rand(batch_size, cfg.action_dim, generator=g) * 0.4 - 0.2
    state_mask = torch.zeros_like(state, dtype=torch.bool)
    active = min(7, cfg.action_dim)
    state_mask[:, :active] = True
    state.masked_fill_(~state_mask, 0)
    actions = torch.zeros(batch_size, cfg.horizon, cfg.action_dim)
    for j in range(cfg.horizon):
        actions[:, j, :active] = 0.3 * state[:, :active] + 0.04 * j
    mask = state_mask[:, None].expand_as(actions).clone()
    images = torch.rand(batch_size, 3, 3, 16, 16, generator=g)
    ids, text_mask = byte_tokens(["move gripper toward target"] * batch_size, min(cfg.max_text_tokens, 32))
    out = {"input_ids": ids, "text_mask": text_mask, "images": images,
           "image_roles": torch.tensor([[0, 1, 2]] * batch_size),
           "image_mask": torch.tensor([[True, True, False]] * batch_size),
           "state": state, "state_mask": state_mask, "actions": actions, "action_mask": mask}
    return {k: v.to(device) for k, v in out.items()}


def run_cpu_smoke(output: str | Path, steps: int = 40, seed: int = 7) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.set_num_threads(1)
    cfg = ReferenceConfig()
    model = ReferenceVLA(cfg)
    opt = torch.optim.AdamW(optimizer_groups(model, 2e-3, 1e-3, 1e-3))
    batch = synthetic_batch(cfg, seed=seed)
    noise = torch.randn_like(batch["actions"])
    t = torch.linspace(0.2, 0.8, len(noise))
    logs = []
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.losses(batch, noise, t)
        loss["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        logs.append({"step": step + 1, "loss": float(loss["total"].detach()), "flow_loss": float(loss["flow"].detach()),
                     "aux_ce": float(loss["aux_ce"].detach()), "grad_norm": float(norm)})
    model.eval()
    actions, trace = model.sample(batch, noise, 4, return_trace=True)
    checkpoint = output / "reference_smoke.pt"
    atomic_torch_save({"format": "pai.reference.v1", "model": model.state_dict(), "optimizer": opt.state_dict(),
                       "config": asdict(cfg), "step": steps, "seed": seed, "torch_rng": torch.get_rng_state(),
                       "model_kind": "random_tiny_cpu_fixture_NOT_pretrained_VLA"}, checkpoint)
    fresh = ReferenceVLA(cfg)
    loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
    fresh.load_state_dict(loaded["model"], strict=True)
    fresh.eval()
    reload_error = float((fresh.sample(batch, noise, 4) - actions).abs().max())
    report = {"model_kind": "random_tiny_cpu_fixture_NOT_pretrained_VLA", "device": "cpu", "dtype": "float32",
              "steps": steps, "initial_flow_loss": logs[0]["flow_loss"], "final_flow_loss": logs[-1]["flow_loss"],
              "initial_total_loss": logs[0]["loss"], "final_total_loss": logs[-1]["loss"],
              "reload_max_abs_error": reload_error, "finite_sample": bool(torch.isfinite(actions).all()),
              "inactive_output_max_abs": float(actions[~batch["action_mask"]].abs().max()),
              "seconds": time.perf_counter() - start, "gpu_validated": False,
              "interpretation": "Fixed-batch learnability and checkpoint round-trip only. No real policy, OOD, AMD, NVIDIA or FP8 result."}
    write_jsonl(output / "learning_curve.jsonl", logs)
    write_json(output / "report.json", report)
    return report
