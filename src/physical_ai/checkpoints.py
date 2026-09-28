"""Fail-closed checkpoint handling. Never call a partially loaded model 'pretrained'
without recording exactly what was loaded, reset, missing, or incompatible.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Mapping
import os
import tempfile
import zipfile
import torch
from torch import Tensor, nn
from .io import sha256_file


def read_tensor_state(path: str | Path, key_path: str = "") -> dict[str, Tensor]:
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(p), device="cpu")
    else:
        # No unrestricted pickle fallback. Convert trusted legacy upstream files
        # in an isolated environment if weights_only rejects their container.
        state = torch.load(p, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(p))
    for part in filter(None, key_path.split(".")):
        if not isinstance(state, dict) or part not in state:
            raise ValueError(f"checkpoint key path missing: {key_path}")
        state = state[part]
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not isinstance(v, Tensor) for k, v in state.items()):
        raise ValueError("expected a nonempty tensor-only state dict; specify --key-path")
    return state


def select_prefix(state: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    if not prefix:
        return dict(state)
    selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not selected:
        raise ValueError(f"no checkpoint keys under explicit prefix {prefix!r}")
    return selected


def audited_load(module: nn.Module, state: Mapping[str, Tensor], *, reset_patterns: tuple[str, ...] = (),
                 ignored_source_patterns: tuple[str, ...] = (), minimum_coverage: float = 1.0,
                 apply: bool = True) -> dict:
    if not 0 < minimum_coverage <= 1:
        raise ValueError("coverage must be in (0,1]")
    expected = module.state_dict()
    reset = {k for k in expected if any(fnmatchcase(k, pat) for pat in reset_patterns)}
    for pat in reset_patterns:
        if not any(fnmatchcase(k, pat) for k in expected):
            raise ValueError(f"reset pattern matches nothing: {pat}")
    compatible, mismatch, missing, ignored = {}, {}, [], []
    for key, target in expected.items():
        if key in reset:
            continue
        value = state.get(key)
        if value is None:
            missing.append(key)
        elif value.shape != target.shape:
            mismatch[key] = {"checkpoint": list(value.shape), "model": list(target.shape)}
        elif not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint has nonfinite tensor: {key}")
        else:
            compatible[key] = value
    unexpected = []
    for key in state:
        if key not in expected:
            if any(fnmatchcase(key, p) for p in ignored_source_patterns):
                ignored.append(key)
            else:
                unexpected.append(key)
    required_numel = sum(v.numel() for k, v in expected.items() if k not in reset)
    loaded_numel = sum(expected[k].numel() for k in compatible)
    coverage = loaded_numel / max(1, required_numel)
    report = {"coverage": coverage, "loaded_numel": loaded_numel, "required_numel": required_numel,
              "loaded": sorted(compatible), "reset": sorted(reset), "missing": sorted(missing),
              "shape_mismatch": mismatch, "unexpected": sorted(unexpected), "ignored_source": sorted(ignored),
              "applied": False}
    # Unexplained missing/shape mismatch always fails. A lower threshold is not
    # permission to silently accept them; resets must be explicitly allowlisted.
    if missing or mismatch or unexpected or coverage < minimum_coverage or required_numel == 0:
        raise ValueError(f"checkpoint audit failed: {report}")
    if apply:
        full = dict(expected)
        full.update(compatible)
        module.load_state_dict(full, strict=True)
        report["applied"] = True
    return report


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parameter_summary(module: nn.Module) -> dict:
    return {"total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
