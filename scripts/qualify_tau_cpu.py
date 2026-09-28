#!/usr/bin/env python3
"""Exercise native Tau data conversion on real bundled data, without model weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from physical_ai.io import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "upstreams.lock.json").read_text())
    expected = next(r["revision"] for r in lock["repositories"] if r["name"] == "tau0")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True)
    if revision != expected or dirty:
        raise ValueError("requires clean pinned Tau source")
    sys.path[:0] = [str(upstream), str(upstream / "src")]
    from configs.example_agibot_world_gong.data import agibot_world_gong_ft
    from deploy.policy import Tau0VLAPolicy  # noqa: F401 Native import qualification.
    from tau0_vla.models.model_builder import ModelBuilder  # noqa: F401
    from tau0_vla.models.vision_language_action_models.tau0_vla.modeling_tau_vla import Tau0VLAModel  # noqa: F401
    from tau0_vla.data import FinchDataLoader

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    loader = FinchDataLoader(agibot_world_gong_ft(), batch_size=1, num_workers=0,
                             video_backend="pyav")
    reports = []
    for index in (0, len(loader.dataset) // 2, len(loader.dataset) - 1):
        sample = loader.dataset[index]
        state, action = np.asarray(sample["state"]), np.asarray(sample["action"])
        state_mask, action_mask = np.asarray(sample["state_mask"]), np.asarray(sample["action_mask"])
        if state.shape != (40,) or action.shape != (30, 40):
            raise ValueError("unexpected native window shape")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("nonfinite native window")
        if np.any(state[state_mask == 0]) or np.any(action[:, action_mask == 0]):
            raise ValueError("inactive native slots are nonzero")
        images = {}
        for camera, value in sample["images"].items():
            array = np.asarray(value)
            if array.shape != (224, 224, 3) or array.dtype != np.uint8:
                raise ValueError(f"unexpected native image contract: {camera}: {array.shape}, {array.dtype}")
            images[camera] = {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        if set(images) != {"head", "wrist_left", "wrist_right"}:
            raise ValueError("unexpected native camera keys")
        reports.append({"filtered_index": index, "prompt": sample["prompt"],
                        "state_shape": list(state.shape), "action_shape": list(action.shape),
                        "active_state_slots": np.flatnonzero(state_mask).tolist(),
                        "active_action_slots": np.flatnonzero(action_mask).tolist(),
                        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                        "images": images})
    report = {
        "upstream_revision": revision, "native_policy_import_passed": True,
        "native_model_import_passed": True,
        "filtered_anchors": len(loader.dataset), "seed": 7, "samples": reports,
        "video_backend": "pyav", "native_training_augmentation_enabled": True,
        "source_info_sha256": sha256_file(upstream / "example_data/meta/info.json"),
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "torchvision", "transformers", "lerobot", "datasets",
                      "huggingface-hub", "pyarrow", "av")},
        "installed_packages": dict(sorted((d.metadata["Name"], d.version)
                                           for d in importlib.metadata.distributions())),
        "dependency_overrides": ["LeRobot installed without dependency resolution; Hub 1.x retained",
                                 "PyArrow 25.0.1 instead of upstream 20.0.0 for datasets 4.1.1"],
        "pretrained_inference_executed": False, "libero_data": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
