"""Strict, complete native windows for the research collation boundary."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch

from .collation import ObservationExample, QwenObservationCollator


@dataclass
class NativeWindow:
    source: str
    episode: str
    schema: str
    normalization: str
    anchor: int
    episode_length: int
    timestamps: np.ndarray
    camera_names: tuple[str, ...]
    camera_times: tuple[float, ...]
    images: tuple[Image.Image, ...]
    instruction: str
    auxiliary_answer: str
    auxiliary_source: str
    state: np.ndarray
    state_mask: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray

    def validate(self):
        if not all((self.source, self.episode, self.schema, self.normalization)):
            raise ValueError("native window provenance required")
        a, s = np.asarray(self.actions), np.asarray(self.state)
        am, sm, ts = np.asarray(self.action_mask), np.asarray(self.state_mask), np.asarray(self.timestamps)
        if a.ndim != 2 or not a.size or s.shape != (a.shape[1],):
            raise ValueError("native state/action shapes disagree")
        if am.shape != a.shape or sm.shape != s.shape or am.dtype != bool or sm.dtype != bool:
            raise ValueError("explicit boolean state/action masks required")
        if not np.isfinite(a).all() or not np.isfinite(s).all():
            raise ValueError("nonfinite native window")
        if np.any(a[~am]) or np.any(s[~sm]) or not am.any(axis=1).all():
            raise ValueError("inactive slots must be zero and all timesteps valid")
        if self.anchor < 0 or self.anchor + len(a) > self.episode_length:
            raise ValueError("window crosses episode boundary")
        if ts.shape != (len(a),) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError("strictly increasing action timestamps required")
        if not self.images or not (len(self.camera_names) == len(self.images) == len(self.camera_times)):
            raise ValueError("camera identities, images and timestamps must align")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("duplicate camera identity")
        if any(not np.isfinite(t) or t > ts[0] + 1e-6 for t in self.camera_times):
            raise ValueError("future or invalid camera timestamp")


class NativeWindowCollator:
    def __init__(self, processor):
        self.observations = QwenObservationCollator(processor)

    def __call__(self, windows: list[NativeWindow]):
        if not windows:
            raise ValueError("nonempty native batch required")
        for w in windows:
            w.validate()
        contracts = {(w.schema, w.normalization, w.actions.shape, w.camera_names) for w in windows}
        if len(contracts) != 1:
            raise ValueError("do not mix native schemas, normalizations, horizons or camera order")
        batch = self.observations([ObservationExample(w.instruction, w.images, w.auxiliary_answer,
                                                      w.auxiliary_source) for w in windows])
        for output, field, dtype in (("actions", "actions", torch.float32),
                                     ("state", "state", torch.float32),
                                     ("action_mask", "action_mask", torch.bool),
                                     ("state_mask", "state_mask", torch.bool)):
            batch[output] = torch.as_tensor(np.stack([getattr(w, field) for w in windows]), dtype=dtype)
        batch["provenance"] = [{"source": w.source, "episode": w.episode, "anchor": w.anchor,
                                "schema": w.schema, "normalization": w.normalization,
                                "timestamps": w.timestamps.tolist(), "cameras": list(w.camera_names),
                                "camera_times": list(w.camera_times)} for w in windows]
        return batch
