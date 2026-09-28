"""Portable numerical qualification. CPU self-parity is not CUDA/ROCm evidence."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import time
from typing import Any
import numpy as np
import torch
from torch import Tensor
from .models import ReferenceConfig, ReferenceVLA
from .training import seed_everything, synthetic_batch
from .checkpoints import atomic_torch_save
from .io import write_json, sha256_file, digest_json


def hardware_info() -> dict:
    return {"torch": torch.__version__, "python": platform.python_version(), "cuda": torch.version.cuda,
            "hip": torch.version.hip, "gpu_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}


def make_fixture(directory: str | Path, seed: int = 17) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    cfg = ReferenceConfig()
    model = ReferenceVLA(cfg)
    batch = synthetic_batch(cfg, seed=seed)
    noise = torch.randn_like(batch["actions"])
    times = torch.tensor([0.11, 0.38, 0.62, 0.89])
    atomic_torch_save({"model": model.state_dict(), "batch": batch, "noise": noise, "time": times,
                       "config": asdict(cfg), "kind": "reference_cpu_fixture"}, directory / "fixture.pt")
    meta = {"fixture_sha256": sha256_file(directory / "fixture.pt"), "seed": seed,
            "config": asdict(cfg), "kind": "random_tiny_reference_NOT_native_W0_A15_Tau", "schema": "pai.oracle.v1"}
    write_json(directory / "fixture.json", meta)
    return meta


def require_device(device: str):
    if device != "cpu" and (not device.startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("requested accelerator unavailable; refusing silent CPU fallback")


def run_probe(fixture_dir: str | Path, output_dir: str | Path, *, device: str = "cpu", precision="fp32", learning_steps=12) -> dict:
    require_device(device)
    if precision not in {"fp32", "bf16"}:
        raise ValueError("FP8 needs a qualified model-specific backend; no fake FP8 casting")
    if learning_steps < 0:
        raise ValueError("negative learning steps")
    if precision == "bf16" and device != "cpu" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable on selected GPU")
    fixture_dir, output_dir = Path(fixture_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((fixture_dir / "fixture.json").read_text())
    if sha256_file(fixture_dir / "fixture.pt") != meta["fixture_sha256"]:
        raise ValueError("fixture checksum mismatch")
    saved = torch.load(fixture_dir / "fixture.pt", map_location="cpu", weights_only=True)
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    seed_everything(meta["seed"])
    torch.set_num_threads(1)
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model = ReferenceVLA(ReferenceConfig(**saved["config"])).to(device=device, dtype=dtype)
    model.load_state_dict(saved["model"], strict=True)
    batch = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in saved["batch"].items()}
    noise = saved["noise"].to(device=device, dtype=dtype)
    t = saved["time"].to(device=device)  # scalar flow interpolation kept FP32
    model.eval()
    arrays = {}
    def record(key, value):
        arrays[key] = value.detach().float().cpu().numpy()
    with torch.no_grad():
        layers, cmask = model.encode(batch)
        record("brain.first", layers[0])
        record("brain.last", layers[-1])
        action, trace = model.sample(batch, noise, 4, return_trace=True)
        record("sample.actions", action)
        record("sample.trace", trace)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    loss = model.losses(batch, noise, t)
    record("forward.velocity", loss["velocity"])
    for key in ("total", "flow", "aux_ce"):
        record(f"loss.{key}", loss[key])
    loss["total"].backward()
    params = {n: p for n, p in model.named_parameters() if p.grad is not None}
    before = {n: p.detach().float().clone() for n, p in params.items()}
    for n, p in params.items():
        record("gradient." + n, p.grad)
    opt.step()
    for n, p in params.items():
        record("update." + n, p.detach().float() - before[n])
    curve = []
    for _ in range(learning_steps):
        opt.zero_grad(set_to_none=True)
        l = model.losses(batch, noise, t)
        l["total"].backward()
        opt.step()
        curve.append(float(l["total"].detach()))
    arrays["learning.loss"] = np.asarray(curve, np.float32)
    np.savez_compressed(output_dir / "tensors.npz", **arrays)
    hw = hardware_info()
    report = {**meta, "device": device, "precision": precision, "hardware": hw, "learning_steps": learning_steps,
              "scope": "reference_model_only", "native_checkpoint_validated": False,
              "framework": "portable_pytorch_NOT_Primus_native", "tensor_count": len(arrays),
              "all_finite": all(np.isfinite(x).all() for x in arrays.values()),
              "tensor_file_sha256": sha256_file(output_dir / "tensors.npz")}
    write_json(output_dir / "probe.json", report)
    return report


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float) -> dict:
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        return {"passed": False, "reason": "shape_mismatch", "reference": list(a.shape), "candidate": list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "reason": "nonfinite"}
    if not a.size:
        return {"passed": True, "count": 0}
    delta = np.abs(a - b)
    norm = float(np.linalg.norm(a.ravel()))
    bn = float(np.linalg.norm(b.ravel()))
    cosine = float(np.dot(a.ravel(), b.ravel()) / (norm * bn)) if norm > 1e-15 and bn > 1e-15 else None
    return {"passed": bool(np.all(delta <= atol + rtol * np.abs(a))), "count": int(a.size),
            "max_abs": float(delta.max()), "p99_abs": float(np.quantile(delta, 0.99)),
            "relative_l2": float(np.linalg.norm((a - b).ravel()) / max(norm, 1e-12)), "cosine": cosine,
            "failing_fraction": float(np.mean(delta > atol + rtol * np.abs(a)))}


def compare_probes(reference_dir: str | Path, candidate_dir: str | Path, output: str | Path, *, atol=1e-5, rtol=1e-4) -> dict:
    if atol < 0 or rtol < 0:
        raise ValueError("negative tolerance")
    paths = [Path(reference_dir), Path(candidate_dir)]
    metas = [json.loads((p / "probe.json").read_text()) for p in paths]
    for p, meta in zip(paths, metas):
        if sha256_file(p / "tensors.npz") != meta["tensor_file_sha256"]:
            raise ValueError("probe tensor checksum mismatch")
    for key in ("fixture_sha256", "scope", "learning_steps"):
        if metas[0][key] != metas[1][key]:
            raise ValueError(f"not a matched comparison: {key}")
    with np.load(paths[0] / "tensors.npz", allow_pickle=False) as a, np.load(paths[1] / "tensors.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("different tensor keys; cannot silently skip missing probes")
        results = {k: compare_arrays(a[k], b[k], atol=atol, rtol=rtol) for k in a.files}
    cuda = any(m["hardware"]["cuda"] and m["device"] != "cpu" for m in metas)
    hip = any(m["hardware"]["hip"] and m["device"] != "cpu" for m in metas)
    report = {"numeric_pass": all(x["passed"] for x in results.values()), "atol": atol, "rtol": rtol,
              "cuda_rocm_pair": bool(cuda and hip), "native_vla_qualified": False,
              "scope": "reference_model_only", "results": results,
              "note": "Tolerance is an explicit diagnostic setting, not a universal success criterion. Native-module and closed-loop checks still required."}
    write_json(output, report)
    return report
