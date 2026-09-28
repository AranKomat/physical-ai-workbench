"""Compact semantic planning, explicit execution memory, no physical actuation."""
from __future__ import annotations
from dataclasses import dataclass, field
import base64
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import math
import numpy as np
from PIL import Image

@dataclass
class ExecutionMemory:
    task: str = ""
    current_subtask: str = ""
    completed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    limit: int = 16

    def begin(self, task: str) -> None:
        if task != self.task:
            self.task, self.current_subtask = task, ""
            self.completed.clear()
            self.failures.clear()

    def propose(self, subtask: str) -> None:
        if not subtask.strip():
            raise ValueError("empty subtask")
        self.current_subtask = subtask.strip()  # NOT yet completed

    def observe_result(self, *, completed: bool, failure: str | None = None) -> None:
        if completed and failure:
            raise ValueError("cannot mark both complete and failed")
        if completed and self.current_subtask:
            self.completed.append(self.current_subtask)
            self.completed[:] = self.completed[-self.limit:]
            self.current_subtask = ""
        if failure:
            self.failures.append(failure)
            self.failures[:] = self.failures[-self.limit:]

    def render(self) -> str:
        return json.dumps({"verified_completed": self.completed, "current": self.current_subtask,
                           "observed_failures": self.failures}, ensure_ascii=False)


@dataclass
class ReplanScheduler:
    max_age_seconds: float = 10.0  # fallback timeout, NOT reported mean subtask duration
    last_plan_time: float | None = None

    def should_replan(self, now: float, *, new_task=False, complete=False, failure=False, scene_change=False) -> bool:
        if not math.isfinite(now) or self.max_age_seconds <= 0:
            raise ValueError("invalid scheduler time")
        if self.last_plan_time is not None and now < self.last_plan_time:
            raise ValueError("time moved backwards")
        return self.last_plan_time is None or new_task or complete or failure or scene_change or now - self.last_plan_time >= self.max_age_seconds

    def mark_planned(self, now: float):
        if not math.isfinite(now):
            raise ValueError("nonfinite time")
        self.last_plan_time = now


def compact_subtask(text: str, *, max_characters: int = 180) -> str:
    result = " ".join(text.strip().split())
    if not result or len(result) > max_characters:
        # Reject rather than truncate a safety-critical qualifier/object name.
        raise ValueError("subtask empty or too long; regenerate concisely")
    return result


def tau_proposal_payload(task: str, memory: str, images: dict[str, str | Path], mode: str = "subtask_only") -> dict:
    if mode not in {"subtask_only", "full_qa"} or set(images) != {"head", "left", "right"}:
        raise ValueError("Tau proposal requires head/left/right cameras and a supported output mode")
    encoded = {}
    for name, p in images.items():
        with Image.open(p) as im:
            im.verify()
        encoded[name] = base64.b64encode(Path(p).read_bytes()).decode("ascii")
    return {"instruction": task, "memory": memory or "(empty)", "task_type": mode, "images": encoded}


class TauProposalClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:10089/predict", timeout: float = 60):
        if not endpoint.startswith(("http://", "https://")) or timeout <= 0:
            raise ValueError("invalid endpoint/timeout")
        self.endpoint, self.timeout = endpoint, timeout

    def predict(self, payload: dict) -> dict:
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not result.get("subtask"):
            raise ValueError("Tau response has no subtask; inspect service response contract")
        if result.get("format_valid") is False:
            raise ValueError("Tau response parse invalid; do not act or mark memory complete")
        result["subtask"] = compact_subtask(result["subtask"])
        return result


def corrupt_memory(completed: list[str], future_subtasks: list[str], mode: str) -> tuple[list[str], str]:
    """TRAINING ONLY: create lagging/optimistic memory inputs; targets stay causal truth.
    Future text is supplied only to build corrupted inputs on train data, not runtime.
    """
    if mode == "lagging":
        return completed[:-1], "omitted_last_completed"
    if mode == "optimistic":
        return completed + future_subtasks[:1], "false_completion"
    if mode == "clean":
        return list(completed), "unchanged"
    raise ValueError("unknown memory corruption")


def subtask_duration_summary(segments: list) -> dict:
    durations = np.array([s.end - s.start for s in segments], dtype=float)
    if not len(durations):
        return {"count": 0, "median_seconds": None}
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("invalid segment durations")
    q = np.quantile(durations, [0.25, 0.5, 0.75, 0.9])
    return {"count": len(durations), "p25_seconds": float(q[0]), "median_seconds": float(q[1]), "p75_seconds": float(q[2]), "p90_seconds": float(q[3])}


def crop_normalized(image: Image.Image, box: list[float], padding: float = 0.05) -> Image.Image:
    if len(box) != 4 or not np.isfinite(box).all() or padding < 0:
        raise ValueError("box must contain four finite normalized coordinates")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("invalid normalized box")
    w, h = x1 - x0, y1 - y0
    left, top = math.floor(max(0, x0 - padding * w) * image.width), math.floor(max(0, y0 - padding * h) * image.height)
    right, bottom = math.ceil(min(1, x1 + padding * w) * image.width), math.ceil(min(1, y1 + padding * h) * image.height)
    if right <= left or bottom <= top:
        raise ValueError("box collapses after rasterization")
    return image.crop((left, top, right, bottom))


def bbox_iou(a, b) -> float:
    for box in (a, b):
        if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("invalid box")
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area - inter))
