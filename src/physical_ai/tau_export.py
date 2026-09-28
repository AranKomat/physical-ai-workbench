"""Audit the pinned native Tau LIBERO export before running upstream code."""
import json
from pathlib import Path

from safetensors import safe_open

from .io import sha256_file


TAU_LIBERO_REVISION = "ce1dd8f0d917011ce45671fa40edaae1fd6a6ec1"
TAU_LIBERO_WEIGHT_SHA256 = "e03870720cbddbd0f3be44ee929a5d23efb9bf9532f0ac1c1bf676224aacc8ec"
TAU_LIBERO_MANIFEST_SHA256 = "de235821a91a4a8972ec045321bb0f6108d065d4d97e00d9fb6c7498826166b8"
ROUTE = "libero-eef-robot-prompt-ft"


def verify_checksums(root: Path, manifest: str = "SHA256SUMS") -> dict[str, str]:
    root = root.resolve()
    checked = {}
    for line in (root / manifest).read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in checked:
            raise ValueError("checksum manifest has an escaping or repeated path")
        if sha256_file(path) != digest:
            raise ValueError(f"checksum mismatch: {name}")
        checked[name] = digest
    if not checked:
        raise ValueError("empty checksum manifest")
    return checked


def audit_tau_libero(root: Path) -> dict:
    if sha256_file(root / "SHA256SUMS") != TAU_LIBERO_MANIFEST_SHA256:
        raise ValueError("checksum manifest differs from the pinned export")
    checked = verify_checksums(root)
    required = {
        "model.safetensors", "config.json", "run_spec.json", "policy_manifest.json",
        "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        *(f"finch_data_spec/{ROUTE}/{name}.json" for name in
          ("spec", "norm_stats", "components", "field_descriptions")),
    }
    if required - checked.keys():
        raise ValueError(f"export missing authenticated files: {sorted(required - checked.keys())}")
    if checked["model.safetensors"] != TAU_LIBERO_WEIGHT_SHA256:
        raise ValueError("weights differ from the selected native LIBERO baseline")
    config = json.loads((root / "config.json").read_text())
    spec = json.loads((root / "finch_data_spec" / ROUTE / "spec.json").read_text())
    expected = {"robot_name": "libero", "action_dim": 40, "state_dim": 40,
                "action_chunk_size": 10, "cam_keys": ["image", "wrist_image"],
                "action_active_indices": [*range(9), 18],
                "state_active_indices": [*range(9), 18], "is_eef": True}
    for name, value in expected.items():
        if spec.get(name) != value:
            raise ValueError(f"unexpected native LIBERO field: {name}")
    if config.get("model_type") != "tau_vla" or config.get("n_action_steps") != 10:
        raise ValueError("unexpected Tau model config")
    tensors, dtype_counts, total = {}, {}, 0
    # Header-only inspection avoids loading the 6 GB model into host memory.
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as state:
        for name in state.keys():
            tensor = state.get_slice(name)
            shape, dtype = tensor.get_shape(), tensor.get_dtype()
            count = 1
            for dim in shape:
                count *= dim
            total += count
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
            tensors[name] = {"shape": shape, "dtype": dtype}
    return {
        "repo_id": "sii-research/tau-0-vla-libero", "revision": TAU_LIBERO_REVISION,
        "verified_files": checked, "tensor_count": len(tensors),
        "stored_elements": total, "elements_by_dtype": dtype_counts,
        "tensors": tensors, "native_contract": expected,
        "pretrained_inference_executed": False,
    }
