"""Clearly synthetic fixtures for exercising the offline pipeline."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .schema import ActionContract
from .data import Episode, Segment, save_episode
from .io import write_jsonl


def synthetic_episode(index: int = 0) -> Episode:
    rng = np.random.default_rng(index)
    n, hz = 40, 10.0
    timestamps = np.arange(n) / hz
    state = np.zeros((n, 80), np.float32)
    actions = np.zeros_like(state)
    mask = np.zeros_like(state, dtype=bool)
    mask[:, :7] = True
    state[:, :3] = np.cumsum(rng.normal(0, 0.002, (n, 3)), 0)
    state[:, 6] = np.linspace(1, 0, n)
    actions[:, :3] = rng.normal(0, 0.001, (n, 3))
    actions[:, 6] = state[:, 6]
    cameras = {}
    for view in ["head", "left", "right"]:
        video = np.zeros((n, 32, 48, 3), np.uint8)
        for t in range(n):
            video[t] = np.array([20 + index, 30, 40], dtype=np.uint8)
            x = 5 + t // 2
            video[t, 10:20, x:x + 8] = [180, 80, 100]
        cameras[view] = video
    return Episode(episode_id=f"synthetic-{index}", source="SYNTHETIC_NOT_ROBOT_DATA",
                   source_episode_id=str(index), split_group=f"synthetic-physical-{index}",
                   family="robot", embodiment="synthetic_fixture", task="move colored square (not physics)",
                   contract=ActionContract(frame="base", control_hz=hz), timestamps=timestamps,
                   actions=actions, action_mask=mask, states=state, state_mask=mask.copy(), cameras=cameras,
                   segments=[Segment(0, 2, "approach the synthetic square"), Segment(2, 4, "move the synthetic square")],
                   provenance={"synthetic": True, "purpose": "schema/unit tests only", "physics_simulated": False})


def write_demo_dataset(directory: str | Path, count: int = 6) -> Path:
    if count < 3:
        raise ValueError("need at least three episodes to exercise disjoint splits")
    directory = Path(directory)
    rows = []
    for i in range(count):
        ep = synthetic_episode(i)
        p = save_episode(ep, directory / ep.episode_id)
        split = "test" if i == count - 1 else "validation" if i == count - 2 else "train"
        rows.append({"episode_id": ep.episode_id, "episode": str(p.relative_to(directory)),
                     "split_group": ep.split_group, "split": split, "family": ep.family,
                     "source": ep.source, "synthetic": True})
    manifest = directory / "manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest
