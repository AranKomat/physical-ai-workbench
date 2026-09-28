#!/usr/bin/env python3
"""Capture native LIBERO observations without a policy or rented GPU.

macOS uses its local OpenGL renderer; this is not a software-rendering claim.
Run in the separate Python 3.10 simulator environment, not the model environment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--libero", type=Path, required=True)
    p.add_argument("--tau", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new capture output directory required")
    for repo, expected in ((args.libero, "8f1084e3132a39270c3a13ebe37270a43ece2a01"),
                           (args.tau, "f1665fbaf624d1468b168e3a54cccc9e326212d9")):
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != expected:
            raise ValueError("incorrect simulator/client revision")
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]):
            raise ValueError("dirty simulator/client source")
    args.output.mkdir(parents=True)
    config = args.output / "libero_config"
    config.mkdir()
    base = args.libero.resolve() / "libero/libero"
    (config / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"), "assets": str(base / "assets"),
        "datasets": str(args.libero.resolve() / "libero/datasets"),
    }))
    os.environ["LIBERO_CONFIG_PATH"] = str(config.resolve())
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "osmesa")
    sys.path[:0] = [str(args.libero.resolve()), str(args.tau.resolve())]
    import torch
    from PIL import Image
    from libero.libero import benchmark
    from deploy.libero.main import _get_libero_env, _prepare_image, _quat2axisangle, LIBERO_DUMMY_ACTION

    suite = benchmark.get_benchmark_dict()[args.suite]()
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        initial = base / "init_files" / task.problem_folder / task.init_states_file
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                                type(np.dtype(np.float64)), type(np.dtype(np.float32))]):
            starts = torch.load(initial, map_location="cpu", weights_only=True)
        env = None
        try:
            env, instruction = _get_libero_env(task, 256, args.seed)
            env.reset()
            obs = env.set_init_state(starts[0])
            for _ in range(10):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            dest = args.output / f"task-{task_id}"
            dest.mkdir()
            keys = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                    "robot0_eef_quat", "robot0_gripper_qpos")
            np.savez(dest / "raw.npz", **{k: obs[k] for k in keys})
            images = {"image": _prepare_image(obs[keys[0]], 224),
                      "wrist_image": _prepare_image(obs[keys[1]], 224)}
            state = np.concatenate((obs[keys[2]], _quat2axisangle(obs[keys[3]]), obs[keys[4]]))
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("invalid native simulator state")
            np.savez(dest / "observation.npz", state=state, **images)
            for camera, pixels in images.items():
                Image.fromarray(pixels).save(dest / f"{camera}.png")
            record = {"input_kind": "simulator", "suite": args.suite, "task_id": task_id,
                      "instruction": instruction, "episode": 0, "seed": args.seed,
                      "settle_actions": 10, "initial_states_sha256": digest(initial),
                      "selected_initial_state_sha256": hashlib.sha256(np.asarray(starts[0]).tobytes()).hexdigest(),
                      "bddl_sha256": digest(base / "bddl_files" / task.problem_folder / task.bddl_file),
                      "raw_sha256": digest(dest / "raw.npz"),
                      "observation_sha256": digest(dest / "observation.npz"),
                      "sim_time": float(env.sim.data.time), "renderer": os.environ["MUJOCO_GL"],
                      "versions": {k: importlib.metadata.version(k) for k in ("torch", "mujoco", "robosuite", "numpy")},
                      "pretrained_policy_executed": False}
            (dest / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
