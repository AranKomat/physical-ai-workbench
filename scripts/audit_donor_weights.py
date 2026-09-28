#!/usr/bin/env python3
"""Hash and inspect real donor tensors without materializing model weights."""
import argparse
from collections import Counter
from pathlib import Path

from safetensors import safe_open
import torch

from physical_ai.io import sha256_file, write_json

PINS = {
    "a15": ("InternRobotics/InternVLA-A1.5-base", "325331b52fc788a7a84419dfcb4930a43b14df1e",
            "9fe03a224b88f041c33dd1aae6bd24364bbae8d85158cf0a32ae9d237bd98851"),
    "w0": ("InternRobotics/InternW0-Delta-Base", "ca534d4f3c9c7131205b4ee37e965b3ef21ffb5f",
           "a6bcf7a1eb2cf3d303e02faca05579cf77f4ac90863a15ff896b7e85ef884a08"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=PINS)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    repo, revision, digest = PINS[args.kind]
    actual = sha256_file(args.checkpoint)
    if actual != digest:
        raise ValueError("donor hash differs from pinned HF LFS hash")
    tensors, containers = {}, []
    if args.kind == "a15":
        with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
            for key in source.keys():
                value = source.get_slice(key)
                tensors[key] = {"shape": value.get_shape(), "dtype": value.get_dtype()}
    else:
        # Safe restricted unpickling and meta tensors avoid a 12 GB RAM copy.
        payload = torch.load(args.checkpoint, map_location="meta", weights_only=True, mmap=True)
        def walk(value, path=""):
            if isinstance(value, torch.Tensor):
                tensors[path] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            elif isinstance(value, dict):
                containers.append(path)
                for key, item in value.items():
                    walk(item, f"{path}.{key}" if path else str(key))
            elif value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"unsupported checkpoint metadata at {path}: {type(value)}")
        walk(payload)
    if not tensors:
        raise ValueError("empty tensor inventory")
    groups = Counter(".".join(key.split(".")[:3]) for key in tensors)
    write_json(args.output, {"repo": repo, "revision": revision, "sha256": actual,
        "bytes": args.checkpoint.stat().st_size, "tensor_count": len(tensors), "containers": containers,
        "groups": dict(groups), "tensors": tensors, "tensor_values_checked": False,
        "pretrained_inference_executed": False})
    print({"kind": args.kind, "tensors": len(tensors), "groups": dict(groups)})


if __name__ == "__main__":
    main()
