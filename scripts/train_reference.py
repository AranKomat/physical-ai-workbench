#!/usr/bin/env python3
"""Train the reference model on canonical episode windows, optionally with CPU/GPU DDP.
This is not native Qwen/Intern/Tau training and is labeled accordingly in reports.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler
from physical_ai.dataset import EpisodeWindowDataset
from physical_ai.schema import PhysicalScaler
from physical_ai.models import ReferenceConfig, ReferenceVLA
from physical_ai.training import seed_everything, optimizer_groups
from physical_ai.checkpoints import atomic_torch_save
from physical_ai.validation import require_device, hardware_info
from physical_ai.io import write_json, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--scaler", required=True)
    p.add_argument("--output", required=True); p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--device", default="cpu")
    p.add_argument("--goal-fraction", type=float, default=0); p.add_argument("--mode", choices=["ki", "joint", "frozen"], default="ki")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("positive steps/batch required")
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if args.device == "cuda" else args.device
    require_device(device)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    seed_everything(7); torch.set_num_threads(1)
    scale = json.loads(Path(args.scaler).read_text())
    scaler = PhysicalScaler(**scale)
    dataset = EpisodeWindowDataset(args.manifest, split="train", horizon=4, scaler=scaler,
               goal_fraction=args.goal_fraction, goal_mode="train" if args.goal_fraction else "none")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=7, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0, drop_last=False)
    cfg = ReferenceConfig(mode=args.mode)
    model = ReferenceVLA(cfg).to(device)
    # Forward delegates to losses so DDP owns the autograd execution.
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if device.startswith("cuda") else None,
                                                       find_unused_parameters=True) if world > 1 else model
    opt = torch.optim.AdamW(optimizer_groups(model))
    logs, step, epoch = [], 0, 0
    start = time.perf_counter()
    try:
        while step < args.steps:
            dataset.set_epoch(epoch)
            if sampler: sampler.set_epoch(epoch)
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                noise = torch.randn_like(batch["actions"]); t = torch.rand(len(noise), device=device)
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    losses = wrapped(batch, noise, t)
                losses["total"].backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step(); step += 1
                mean_loss = losses["total"].detach().clone()
                if world > 1:
                    torch.distributed.all_reduce(mean_loss); mean_loss /= world
                if rank == 0:
                    logs.append({"step": step, "epoch": epoch, "loss": float(mean_loss), "local_grad_norm": float(grad)})
                if step >= args.steps: break
            epoch += 1
        if rank == 0:
            root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
            atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg), "step": step,
                              "model_kind": "reference_NOT_pretrained_VLA", "torch_rng": torch.get_rng_state()}, root / "checkpoint.pt")
            write_jsonl(root / "metrics.jsonl", logs)
            write_json(root / "run.json", {"model_kind": "reference_NOT_pretrained_VLA", "hardware": hardware_info(),
                         "world_size": world, "steps": step, "seconds": time.perf_counter() - start,
                         "mode": args.mode, "precision": args.precision, "goal_fraction": args.goal_fraction,
                         "native_vla_qualified": False, "exact_distributed_resume_supported": False})
    finally:
        if world > 1: torch.distributed.destroy_process_group()

if __name__ == "__main__": main()
