#!/usr/bin/env python3
"""Build a hashed transfer manifest and gated native CUDA plan, without execution."""
import argparse
import json
from pathlib import Path
import subprocess

from physical_ai.io import sha256_file, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--batches", type=Path, required=True)
    p.add_argument("--w0-contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new bundle output required")
    files = []
    for directory in (Path("weights/tau0-libero"), Path("weights/w0-base"), Path("weights/a15-base"),
                      args.captures, args.batches):
        if not directory.is_dir():
            raise ValueError(f"missing bundle input: {directory}")
        files.extend(f for f in directory.rglob("*") if f.is_file() and ".cache" not in f.parts
                     and "libero_config" not in f.parts and f.suffix != ".lock")
    files.append(args.w0_contract)
    inventory = [{"path": str(f), "sha256": sha256_file(f), "bytes": f.stat().st_size}
                 for f in sorted(set(files))]
    jobs = []
    for observation in sorted(args.captures.glob("task-*/observation.npz")):
        record = json.loads((observation.parent / "provenance.json").read_text())
        if record["input_kind"] != "simulator" or sha256_file(observation) != record["observation_sha256"]:
            raise ValueError("real, hash-matched simulator fixture required")
        jobs.append({"name": "tau_" + observation.parent.name.replace("-", "_"), "timeout_seconds": 600,
            "argv": [".venv-tau/bin/python", "scripts/probe_tau_native.py", "--upstream", "third_party/tau0",
                     "--checkpoint", "weights/tau0-libero", "--input", str(observation), "--instruction", record["instruction"],
                     "--input-kind", "simulator", "--device", "cuda", "--seed", "7", "--output",
                     "runs/cuda-native/" + observation.parent.name]})
    if not jobs:
        raise ValueError("no real simulator observations to bundle")
    jobs.append({"name": "w0_numerical", "timeout_seconds": 900, "argv": [".venv-tau/bin/python",
        "scripts/probe_w0_numerics.py", "--upstream", "third_party/internw0_delta", "--checkpoint",
        "weights/w0-base/pretrain.pt", "--contract", str(args.w0_contract), "--device", "cuda",
        "--output", "runs/cuda-native/w0-numerical"]})
    # Readiness is promoted only after host-specific runtime import/VRAM preflight.
    plan = {"ready_for_cuda": False, "jobs": jobs, "required_files": inventory,
            "readiness_blockers": ["NVIDIA runtime import/driver/VRAM preflight not performed",
                                   "Native pretrained probes have not executed"],
            "research_training_ready": False,
            "note": "Motor probe has synthetic inputs; Tau jobs use real recorded observations. No closed-loop success implied."}
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "transfer.json", {"files": inventory, "total_bytes": sum(f["bytes"] for f in inventory),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_lock_sha256": sha256_file("upstreams.lock.json"),
        "install_note": "Clone pinned sources separately; do not copy Mac virtualenvs or caches to Linux."})
    print({"jobs": len(jobs), "files": len(files), "ready_for_cuda": False})


if __name__ == "__main__":
    main()
