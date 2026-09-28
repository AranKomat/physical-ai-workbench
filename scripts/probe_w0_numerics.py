#!/usr/bin/env python3
"""Bounded native donor numeric reference, not a physical-policy evaluation."""
import argparse
import json
from pathlib import Path
import sys

import torch

from physical_ai.io import sha256_file, write_json
from physical_ai.native import NativeFlowConvention, load_w0_motor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new numerical output directory required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device unavailable; no fallback")
    spec = json.loads(args.contract.read_text())
    import subprocess
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["source_revision"] or subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("donor source mismatch")
    sys.path.insert(0, str((args.upstream / "src").resolve()))
    torch.manual_seed(7)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    adapter, audit = load_w0_motor(checkpoint=str(args.checkpoint), constructor_config=spec["constructor"],
        state_dict_key=spec["state_dict_key"], source_prefix=spec["source_prefix"],
        context_dim=spec["context_dim"], convention=NativeFlowConvention(1000, "data_to_noise", -1),
        device=args.device, dtype=dtype, reset_action_io=False,
        checkpoint_sha256=spec["checkpoint_sha256"], qualification_mode=True,
        source_context_dim=spec["source_context_dim"])
    args.output.mkdir(parents=True)
    dim, horizon = spec["constructor"]["action_dim"], 4
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, horizon, dim, generator=generator).to(args.device, dtype)
    context = torch.randn(1, 8, spec["context_dim"], generator=generator).to(args.device, dtype)
    state = torch.zeros(1, dim, device=args.device, dtype=dtype)
    am = torch.ones_like(x, dtype=torch.bool)
    sm = torch.ones_like(state, dtype=torch.bool)
    cm = torch.ones(1, 8, device=args.device, dtype=torch.bool)
    time = torch.tensor([.5], device=args.device, dtype=dtype)
    velocity = adapter(x, time, context, cm, state, sm, am)
    if not torch.isfinite(velocity).all():
        raise ValueError("nonfinite native velocity")
    loss = velocity.float().square().mean()
    loss.backward()
    selected = {k: p for k, p in adapter.named_parameters()
                if k.startswith("state_adapter.") or k.startswith("action_dit.action_encoder.") or k.startswith("action_dit.head.")}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in selected.values()):
        raise ValueError("invalid selected gradients")
    capture = {"input": x.detach().cpu(), "context": context.detach().cpu(), "time": time.cpu(),
               "velocity": velocity.detach().cpu(), "loss": loss.detach().cpu()}
    for name, parameter in selected.items():
        capture["gradient." + name] = parameter.grad.detach().cpu().clone()
    # SGD is an explicit numerical probe, not the planned research optimizer.
    optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-5)
    optimizer.step()
    for name, parameter in selected.items():
        capture["updated." + name] = parameter.detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        current = x.clone()
        for step in range(4):
            capture[f"sampler.{step}"] = current.cpu().clone()
            current = current + .25 * adapter(current, torch.tensor([step / 4], device=args.device, dtype=dtype),
                                               context, cm, state, sm, am)
        capture["sampler.4"] = current.cpu().clone()
    if any(not torch.isfinite(t).all() for t in capture.values()):
        raise ValueError("nonfinite native numeric capture")
    torch.save(capture, args.output / "capture.pt")
    write_json(args.output / "report.json", {"load_audit": audit, "device": args.device,
        "precision": str(dtype), "torch": torch.__version__, "cuda": torch.version.cuda,
        "hip": torch.version.hip, "seed": 7, "contract_sha256": sha256_file(args.contract),
        "capture_sha256": sha256_file(args.output / "capture.pt"), "source_revision": revision,
        "purpose": "synthetic-input native motor numerical reference, not policy success",
        "sampler_after_sgd_update": True, "optimizer": "SGD lr=1e-5", "physical_qualification": False})


if __name__ == "__main__":
    main()
