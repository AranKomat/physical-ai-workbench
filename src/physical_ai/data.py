"""Local, explicit episode interchange and sampling.

NPZ is an offline interchange format, not a replacement for upstream LeRobot
video storage. Source-specific adapters should export this contract or implement
an equivalent streaming interface. No generic adapter guesses FK/calibration.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import numpy as np
from PIL import Image
from .schema import ACTION_DIM, ActionContract
from .geometry import check_transform
from .io import read_jsonl, resolve_local, write_json, write_jsonl


@dataclass
class Segment:
    start: float
    end: float
    subtask: str
    quality: int | None = None
    mistake: bool | None = None

    def validate(self) -> None:
        if not np.isfinite([self.start, self.end]).all() or self.start < 0 or self.end <= self.start or not self.subtask.strip():
            raise ValueError("invalid subtask segment")
        if self.quality is not None and (isinstance(self.quality, bool) or self.quality not in range(1, 6)):
            raise ValueError("quality must be 1..5 or unknown")
        if self.mistake is not None and not isinstance(self.mistake, bool):
            raise ValueError("mistake must be boolean or unknown")

@dataclass
class Episode:
    episode_id: str
    source: str
    source_episode_id: str
    split_group: str  # shared by duplicates/annotation overlays of the same physical episode
    family: str
    embodiment: str
    task: str
    contract: ActionContract
    timestamps: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    states: np.ndarray
    state_mask: np.ndarray
    cameras: dict[str, np.ndarray]  # synchronized uint8 [T,H,W,3]
    segments: list[Segment] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.contract.validate()
        for name in ("episode_id", "source", "source_episode_id", "split_group", "family", "embodiment", "task"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        t = np.asarray(self.timestamps)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or t[0] < 0 or np.any(np.diff(t) <= 0):
            raise ValueError("timestamps must be finite, nonnegative and strictly increasing")
        for kind in ("action", "state"):
            x = np.asarray(self.actions if kind == "action" else self.states)
            mask = np.asarray(self.action_mask if kind == "action" else self.state_mask)
            if x.shape != (len(t), ACTION_DIM) or mask.shape != x.shape or mask.dtype != np.bool_:
                raise ValueError(f"{kind} tensors must be [T,80] with boolean mask")
            if not np.isfinite(x).all() or np.any(x[~mask] != 0):
                raise ValueError(f"{kind} has NaN/Inf or nonzero padding")
            if kind == "action" and not np.all(mask.any(axis=1)):
                raise ValueError("every action timestep needs an active channel")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        for name, video in self.cameras.items():
            if not name or video.dtype != np.uint8 or video.ndim != 4 or len(video) != len(t) or video.shape[-1] != 3:
                raise ValueError(f"camera {name}: expected synchronized uint8 [T,H,W,3]")
        if self.contract.frame == "camera":
            name = self.contract.reference_camera
            if name not in self.cameras or name not in self.calibration:
                raise ValueError("reference camera calibration missing")
            c = self.calibration[name]
            k = np.asarray(c.get("intrinsics"))
            if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError("invalid intrinsics")
            transform = np.asarray(c.get("camera_from_base"))
            if transform.shape == (4, 4):
                check_transform(transform)
            elif transform.shape == (len(t), 4, 4):
                for mat in transform:
                    check_transform(mat)
            else:
                raise ValueError("camera_from_base must be [4,4] or [T,4,4]")
        last_end = -1.0
        for seg in self.segments:
            seg.validate()
            if seg.start < last_end or seg.end > t[-1] + 1 / self.contract.control_hz + 1e-6:
                raise ValueError("segments must not overlap or extend past the episode")
            last_end = seg.end

    def segment_at(self, timestamp: float) -> Segment | None:
        return next((s for s in self.segments if s.start <= timestamp < s.end), None)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])


def save_episode(ep: Episode, directory: str | Path) -> Path:
    ep.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = {"timestamps": ep.timestamps, "actions": ep.actions, "action_mask": ep.action_mask,
               "states": ep.states, "state_mask": ep.state_mask}
    for i, name in enumerate(sorted(ep.cameras)):
        payload[f"camera_{i}"] = ep.cameras[name]
    np.savez_compressed(root / "arrays.npz", **payload)
    meta = {k: getattr(ep, k) for k in ("episode_id", "source", "source_episode_id", "split_group", "family", "embodiment", "task", "provenance", "calibration")}
    meta.update(contract=asdict(ep.contract), segments=[asdict(s) for s in ep.segments],
                cameras=sorted(ep.cameras), arrays="arrays.npz", format="pai.episode.v1")
    write_json(root / "episode.json", meta)
    return root / "episode.json"


def load_episode(path: str | Path) -> Episode:
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get("format") != "pai.episode.v1":
        raise ValueError("unknown episode format")
    with np.load(resolve_local(path.parent, meta["arrays"]), allow_pickle=False) as arrays:
        values = {k: arrays[k].copy() for k in ("timestamps", "actions", "action_mask", "states", "state_mask")}
        cameras = {name: arrays[f"camera_{i}"].copy() for i, name in enumerate(meta["cameras"])}
    ep = Episode(**{k: meta[k] for k in ("episode_id", "source", "source_episode_id", "split_group", "family", "embodiment", "task")},
                 contract=ActionContract(**meta["contract"]), **values, cameras=cameras,
                 segments=[Segment(**s) for s in meta.get("segments", [])],
                 calibration=meta.get("calibration", {}), provenance=meta.get("provenance", {}))
    ep.validate()
    return ep


def split_for_group(group: str, seed: int = 0, validation_fraction: float = 0.1, test_fraction: float = 0.1) -> str:
    if not group or not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid split parameters")
    x = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if x < test_fraction else "validation" if x < test_fraction + validation_fraction else "train"


def validate_disjoint_splits(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    ids = set()
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        if row["episode_id"] in ids:
            raise ValueError("duplicate episode id in manifest")
        ids.add(row["episode_id"])
        group, split = row["split_group"], row["split"]
        if group in seen and seen[group] != split:
            raise ValueError(f"physical episode leaked across splits: {group}")
        seen[group] = split


def anchor_indices(timestamps: np.ndarray, anchor_hz: float) -> np.ndarray:
    t = np.asarray(timestamps)
    if anchor_hz <= 0 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("invalid times or anchor rate")
    targets = np.arange(t[0], t[-1] + 1e-9, 1 / anchor_hz)
    # Most recent available observation, never one from the future.
    indices = np.searchsorted(t, targets, side="right") - 1
    return np.unique(indices.clip(0, len(t) - 1))


def history_indices(timestamps: np.ndarray, anchor: int, offsets_seconds=(1.0, 0.25, 0.0)) -> list[int | None]:
    if not 0 <= anchor < len(timestamps) or any(o < 0 for o in offsets_seconds):
        raise ValueError("invalid anchor/history offsets")
    result = []
    for offset in offsets_seconds:
        target = timestamps[anchor] - offset
        i = int(np.searchsorted(timestamps, target + 1e-9, side="right") - 1)
        result.append(None if i < 0 else min(i, anchor))
    return result


def action_window(ep: Episode, anchor: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0 or not 0 <= anchor < len(ep.timestamps):
        raise ValueError("invalid horizon/anchor")
    n = min(horizon, len(ep.timestamps) - anchor)
    a = np.zeros((horizon, ACTION_DIM), np.float32)
    m = np.zeros_like(a, dtype=bool)
    a[:n], m[:n] = ep.actions[anchor:anchor + n], ep.action_mask[anchor:anchor + n]
    return a, m


def select_future_goal(ep: Episode, anchor: int, rng: np.random.Generator, max_seconds: float = 4.0, endpoint_probability: float = 0.25) -> int | None:
    """Strictly future and inside the same subtask. Endpoints may exceed max_seconds.
    The branch probability is a configurable planning prior, not a reproduced paper result.
    """
    if max_seconds <= 0 or not 0 <= endpoint_probability <= 1 or not 0 <= anchor < len(ep.timestamps):
        raise ValueError("invalid future-goal sampling parameters")
    now = float(ep.timestamps[anchor])
    seg = ep.segment_at(now)
    limit = seg.end if seg else float(ep.timestamps[-1]) + 1e-9
    ids = np.where((ep.timestamps > now) & (ep.timestamps < limit))[0]
    if not len(ids):
        return None
    if seg is not None and rng.random() < endpoint_probability:
        return int(ids[-1])
    ids = ids[ep.timestamps[ids] <= now + max_seconds]
    return int(rng.choice(ids)) if len(ids) else None


def goal_pair_manifest(episodes: list[tuple[Episode, str]], output: Path, pairs_per_episode: int = 8, seed: int = 0) -> list[dict]:
    """Export real start/target pairs for editor adaptation, NOT generated labels.
    Only segment text is used. No future frames may cross episode/split boundaries.
    """
    if pairs_per_episode <= 0:
        raise ValueError("pairs_per_episode must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    for ep, split in episodes:
        ep.validate()
        anchors = anchor_indices(ep.timestamps, 1.0)
        rng.shuffle(anchors)
        count = 0
        for anchor in anchors:
            seg = ep.segment_at(float(ep.timestamps[anchor]))
            if seg is None:
                continue  # Whole-task labels cannot silently become subtask labels.
            goal = select_future_goal(ep, int(anchor), rng)
            if goal is None:
                continue
            for view, video in ep.cameras.items():
                identity = f"{ep.source}:{ep.source_episode_id}:{anchor}:{goal}:{view}"
                key = hashlib.sha256(identity.encode()).hexdigest()[:24]
                image = f"images/{key}_start.png"
                target = f"images/{key}_goal.png"
                (output / "images").mkdir(exist_ok=True)
                Image.fromarray(video[anchor]).save(output / image)
                Image.fromarray(video[goal]).save(output / target)
                rows.append({"id": key, "image": image, "target": target, "instruction": seg.subtask,
                             "episode_id": ep.episode_id, "source_episode_id": ep.source_episode_id,
                             "split_group": ep.split_group, "split": split, "camera": view,
                             "start_time": float(ep.timestamps[anchor]), "target_time": float(ep.timestamps[goal]),
                             "goal_kind": "real_future_privileged", "label_origin": "dataset_segment"})
            count += 1
            if count >= pairs_per_episode:
                break
    write_jsonl(output / "pairs.jsonl", rows)
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", [r for r in rows if r["split"] == split])
    return rows


def quality_report(ep: Episode, max_abs_action: float | None = None, max_gap_seconds: float | None = None) -> dict:
    ep.validate()
    gaps = np.diff(ep.timestamps)
    report = {"episode_id": ep.episode_id, "samples": len(ep.timestamps), "duration_seconds": ep.duration,
              "median_dt": float(np.median(gaps)), "max_dt": float(gaps.max()),
              "action_abs_max": float(np.abs(ep.actions[ep.action_mask]).max()),
              "frame_tier": {"camera": "A", "base": "B", "native_joint": "C"}[ep.contract.frame],
              "flags": [], "quality": "unknown"}
    if max_abs_action is not None and report["action_abs_max"] > max_abs_action:
        report["flags"].append("action_range_exceeds_configured_threshold")
    if max_gap_seconds is not None and report["max_dt"] > max_gap_seconds:
        report["flags"].append("timestamp_gap")
    # Warn, do not reject intentional pauses or derive success from smoothness.
    report["static_action_fraction"] = float(np.mean(np.linalg.norm(ep.actions, axis=1) < 1e-8))
    report["camera_frozen_fraction"] = {name: float(np.mean(np.all(np.diff(v.astype(np.int16), axis=0) == 0, axis=(1, 2, 3)))) for name, v in ep.cameras.items()}
    return report


class FamilySampler:
    """Explicit family weights + uniform episodes within family. Not window-count weighting.
    Missing families are recorded; raw ego with no valid actions belongs to VL, not VLA loss.
    """
    def __init__(self, rows: list[dict], weights: dict[str, float], seed: int = 0):
        if not rows or not weights or any(not np.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("nonempty rows and nonnegative finite weights required")
        self.rows, self.rng = rows, np.random.default_rng(seed)
        self.groups = {f: [i for i, r in enumerate(rows) if r["family"] == f] for f in weights}
        self.missing = [f for f, ids in self.groups.items() if not ids]
        self.families = [f for f in weights if self.groups[f] and weights[f] > 0]
        if not self.families:
            raise ValueError("no available family with positive weight")
        p = np.array([weights[f] for f in self.families])
        self.p = p / p.sum()

    def sample(self, n: int) -> list[dict]:
        if n < 0:
            raise ValueError("n must be nonnegative")
        families = self.rng.choice(self.families, size=n, p=self.p)
        return [self.rows[int(self.rng.choice(self.groups[f]))] for f in families]
