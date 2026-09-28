#!/usr/bin/env python3
"""Convert an explicit raw LIBERO observation using the pinned native client.

This does not launch a simulator or create observations. Synthetic inputs must
be labeled synthetic; their successful conversion is not policy qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from physical_ai.io import sha256_file, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--input-kind", choices=["simulator", "synthetic"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output directory already exists")
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "upstreams.lock.json").read_text())
    expected = next(r["revision"] for r in lock["repositories"] if r["name"] == "tau0")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected or subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("clean pinned Tau source required")
    sys.path.insert(0, str(args.upstream.resolve()))
    from deploy.libero.main import _prepare_image, _quat2axisangle

    with np.load(args.raw, allow_pickle=False) as raw:
        pos = np.asarray(raw["robot0_eef_pos"])
        quat = np.asarray(raw["robot0_eef_quat"])
        gripper = np.asarray(raw["robot0_gripper_qpos"])
        for array, shape in ((pos, (3,)), (quat, (4,)), (gripper, (2,))):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("invalid native proprioception")
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-4):
            raise ValueError("expected normalized xyzw quaternion")
        images = {out: _prepare_image(raw[key], 224) for out, key in
                  (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image"))}
    state = np.concatenate((pos, _quat2axisangle(quat), gripper))
    args.output.mkdir(parents=True)
    np.savez(args.output / "observation.npz", state=state, **images)
    write_json(args.output / "provenance.json", {
        "upstream_revision": revision, "raw_sha256": sha256_file(args.raw),
        "input_kind": args.input_kind, "task": args.task, "suite": args.suite,
        "episode": args.episode, "seed": args.seed,
        "output_sha256": sha256_file(args.output / "observation.npz"),
        "state_contract": "xyz3 + axis-angle3 + gripper2; raw quaternion xyzw",
        "image_contract": "native 180-degree rotation, PIL bilinear resize to 224",
        "pretrained_inference_executed": False,
    })


if __name__ == "__main__":
    main()
