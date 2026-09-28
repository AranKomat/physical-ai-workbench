#!/usr/bin/env python3
"""Compare actual Base tensor inventory with native meta-device architecture."""
import argparse
import json
from pathlib import Path
import sys

import torch

from physical_ai.io import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec, audit = json.loads(args.contract.read_text()), json.loads(args.audit.read_text())
    if audit["sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("audit/contract mismatch")
    import subprocess
    if subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() != spec["source_revision"]:
        raise ValueError("source revision mismatch")
    sys.path.insert(0, str((args.upstream / "src").resolve()))
    from wam.model.modules.experts.action_dit import ActionDiT
    with torch.device("meta"):
        native = ActionDiT(**spec["constructor"])
        native.configure_vlm_conditioning(spec["source_context_dim"])
    prefix = spec["state_dict_key"] + "." + spec["source_prefix"]
    actual = {k[len(prefix):]: v for k, v in audit["tensors"].items() if k.startswith(prefix)}
    expected = native.state_dict()
    if actual.keys() != expected.keys():
        raise ValueError(f"key mismatch: missing={expected.keys()-actual.keys()}, extra={actual.keys()-expected.keys()}")
    mismatch = {k: [list(v.shape), actual[k]["shape"]] for k, v in expected.items() if list(v.shape) != actual[k]["shape"]}
    if mismatch:
        raise ValueError(f"shape mismatch: {mismatch}")
    write_json(args.output, {"key_and_shape_match": True, "tensor_count": len(actual),
        "parameter_elements": sum(p.numel() for p in native.parameters()),
        "source_context_dim": spec["source_context_dim"], "checkpoint_sha256": audit["sha256"],
        "tensor_values_loaded_or_checked": False, "pretrained_forward": False})
    print({"match": True, "tensors": len(actual), "parameters": sum(p.numel() for p in native.parameters())})


if __name__ == "__main__":
    main()
