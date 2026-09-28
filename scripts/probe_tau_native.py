#!/usr/bin/env python3
"""One native Tau eager inference, using its full export and an explicit input NPZ.

Run in the isolated upstream model environment. The NPZ must contain `image`,
`wrist_image` (uint8 HWC RGB, already rotated/resized as the native LIBERO client
does) and `state` (8D xyz/axis-angle/two-finger positions). This script neither
creates observations nor claims a simulator success from an action array.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from physical_ai.io import sha256_file, write_json
from physical_ai.tau_export import audit_tau_libero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input-kind", choices=["simulator", "recorded", "synthetic"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    lock = json.loads((Path(__file__).resolve().parents[1] / "upstreams.lock.json").read_text())
    expected = next(e["revision"] for e in lock["repositories"] if e["name"] == "tau0")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True)
    if revision != expected or dirty:
        raise ValueError("native probe requires the clean pinned Tau source")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU unavailable; no implicit CPU fallback")
    archive = np.load(args.input, allow_pickle=False)
    with archive:
        images = {key: archive[key].copy() for key in ("image", "wrist_image")}
        state = archive["state"].copy()
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("finite native 8D state required")
    for key, image in images.items():
        if image.dtype != np.uint8 or image.shape != (224, 224, 3):
            raise ValueError(f"{key}: expected processed uint8 224x224 RGB")
    audit = audit_tau_libero(args.checkpoint)
    args.output.mkdir(parents=True)
    write_json(args.output / "export_audit.json", audit)
    sys.path[:0] = [str(args.upstream.resolve()), str((args.upstream / "src").resolve())]
    from deploy._bootstrap import ensure_configs_registered, discover_checkpoint_config_modules
    from deploy.policy import Tau0VLAPolicy
    from deploy.warmup import _configure_inference_mode

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_configs_registered()
    discover_checkpoint_config_modules(args.checkpoint)
    start = time.perf_counter()
    policy = Tau0VLAPolicy.from_checkpoint(args.checkpoint, device=args.device)
    _configure_inference_mode(policy, "eager", 0)
    loaded = time.perf_counter()
    actions = policy.infer({"images": images, "state": state,
                            "prompt": args.instruction})["actions"]
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise RuntimeError("invalid native action chunk")
    np.save(args.output / "actions.npy", actions, allow_pickle=False)
    write_json(args.output / "report.json", {
        "upstream_revision": revision, "checkpoint_revision": audit["revision"],
        "input_sha256": sha256_file(args.input), "input_kind": args.input_kind,
        "instruction": args.instruction, "seed": args.seed, "device": args.device,
        "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
        "load_seconds": loaded - start, "inference_seconds": time.perf_counter() - loaded,
        "action_shape": list(actions.shape), "finite": True,
        "closed_loop_success": None,
    })


if __name__ == "__main__":
    main()
