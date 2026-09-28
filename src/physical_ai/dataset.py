"""Reference episode-window Dataset; deterministic, goal-aware, causal history.

Intended for local integration tests and modest converted subsets. NPZ episodes
are cached in RAM; use native LeRobot/video streaming for the large foundation pool.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .io import read_jsonl, resolve_local
from .data import load_episode, validate_disjoint_splits, anchor_indices, action_window, history_indices, select_future_goal
from .schema import PhysicalScaler
from .models import byte_tokens


class EpisodeWindowDataset(Dataset):
    def __init__(self, manifest: str | Path, *, split: str, horizon: int, scaler: PhysicalScaler,
                 anchor_hz: float = 1.0, image_size: int = 32, goal_fraction: float = 0.0,
                 goal_mode: str = "none", use_subtasks: bool = False, seed: int = 0,
                 cache_episodes: int = 2, complete_horizons: bool = False):
        if split not in {"train", "validation", "test"} or goal_mode not in {"none", "train", "oracle_eval"}:
            raise ValueError("invalid split/goal mode")
        if not 0 <= goal_fraction <= 1 or horizon < 1 or image_size < 1 or cache_episodes < 1:
            raise ValueError("invalid dataset parameters")
        if goal_fraction > 0 and goal_mode == "none":
            raise ValueError("goal conditioning needs an explicit privileged mode")
        if goal_mode == "train" and split != "train":
            raise ValueError("real-future conditioning forbidden on deployment validation/test")
        if use_subtasks and split != "train" and goal_mode != "oracle_eval":
            raise ValueError("ground-truth subtask labels in held-out eval are oracle inputs; label explicitly")
        self.root = Path(manifest).parent
        all_rows = list(read_jsonl(manifest)); validate_disjoint_splits(all_rows)
        self.rows = [row for row in all_rows if row["split"] == split]
        self.scaler, self.horizon = scaler, horizon
        self.image_size, self.goal_fraction, self.goal_mode = image_size, goal_fraction, goal_mode
        self.use_subtasks, self.seed, self.epoch = use_subtasks, seed, 0
        self.cache_episodes = cache_episodes
        self.cache = OrderedDict()
        self.index = []
        for i, row in enumerate(self.rows):
            ep = self._episode(i)
            if row["split_group"] != ep.split_group or row["episode_id"] != ep.episode_id:
                raise ValueError("manifest/episode identity mismatch")
            for anchor in anchor_indices(ep.timestamps, anchor_hz):
                if complete_horizons and anchor + horizon > len(ep.timestamps):
                    continue
                self.index.append((i, int(anchor)))
        if not self.index:
            raise ValueError("no valid windows for requested split")

    def _episode(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        ep = load_episode(resolve_local(self.root, self.rows[i]["episode"]))
        self.cache[i] = ep
        while len(self.cache) > self.cache_episodes:
            self.cache.popitem(last=False)
        return ep

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row_id, anchor = self.index[index]; ep = self._episode(row_id)
        entropy = int(hashlib.sha256(f"{self.seed}:{self.epoch}:{ep.episode_id}:{anchor}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(entropy)
        a, am = action_window(ep, anchor, self.horizon)
        a = self.scaler.encode(a, am)
        # State normalization must be specified separately in production; reference
        # states use physical values and masks, not action displacement scales.
        state, sm = ep.states[anchor].copy(), ep.state_mask[anchor].copy()
        views = sorted(ep.cameras)[:3]
        hist = history_indices(ep.timestamps, anchor, (.5, 0))
        target = select_future_goal(ep, anchor, rng) if self.goal_fraction and rng.random() < self.goal_fraction else None
        images, roles, masks = [], [], []
        for kind, timestamp in ((1, hist[0]), (0, anchor), (2, target)):
            for camera_idx in range(3):
                available = timestamp is not None and camera_idx < len(views)
                if available:
                    raw = ep.cameras[views[camera_idx]][timestamp]
                    im = Image.fromarray(raw).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    arr = np.asarray(im, dtype=np.float32) / 255
                else:
                    arr = np.zeros((self.image_size, self.image_size, 3), np.float32)
                images.append(arr.transpose(2, 0, 1)); roles.append(kind); masks.append(available)
        seg = ep.segment_at(float(ep.timestamps[anchor]))
        prompt = f"Task: {ep.task}. Control: {ep.contract.frame}; Hz: {ep.contract.control_hz:g}."
        if self.use_subtasks and seg:
            prompt += f" Subtask: {seg.subtask}."
        ids, text_mask = byte_tokens([prompt], 64)
        return {"input_ids": ids[0], "text_mask": text_mask[0], "images": torch.tensor(np.stack(images)),
                "image_roles": torch.tensor(roles), "image_mask": torch.tensor(masks, dtype=torch.bool),
                "state": torch.tensor(state), "state_mask": torch.tensor(sm),
                "actions": torch.tensor(a), "action_mask": torch.tensor(am)}
