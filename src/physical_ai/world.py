"""Optional visual goals: provenance, caching, batch generation and Tau CLI bridges.

Tau's world model is used for high-level proposals/search in its released design.
This does NOT establish that its stock low-level policy accepts visual goals.
Our goal-conditioned motor path is a separate research extension.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol
import json
import math
import subprocess
import time
from PIL import Image
from .io import digest_json, sha256_file, write_json, read_jsonl, resolve_local

@dataclass(frozen=True)
class GoalRecord:
    path: str
    kind: str  # generated | user_supplied | real_future_privileged
    observation_hash: str
    camera_id: str
    subtask: str
    created_at: float
    model_id: str | None = None
    model_revision: str | None = None
    adapter_hash: str | None = None

    def validate_for(self, mode: str, *, camera_id: str, now: float | None = None, max_age: float | None = None):
        if mode not in {"train", "oracle_eval", "deployment_eval", "deployment"}:
            raise ValueError("unknown goal mode")
        if self.kind not in {"generated", "user_supplied", "real_future_privileged"}:
            raise ValueError("unknown goal provenance")
        if self.kind == "real_future_privileged" and mode not in {"train", "oracle_eval"}:
            raise ValueError("future demonstration frame leaked into deployment evaluation")
        if camera_id != self.camera_id or not self.observation_hash or not self.subtask:
            raise ValueError("goal camera/provenance mismatch")
        if self.kind == "generated" and (not self.model_id or not self.model_revision):
            raise ValueError("generated goal requires generator revision")
        if not math.isfinite(self.created_at):
            raise ValueError("invalid goal timestamp")
        if now is not None and max_age is not None and (now < self.created_at or now - self.created_at > max_age):
            raise ValueError("stale or future-timestamped goal")
        if not Path(self.path).is_file():
            raise ValueError("goal image file missing")
        with Image.open(self.path) as image:
            image.verify()


class ImageEditor(Protocol):
    model_id: str
    revision: str
    adapter_hash: str | None
    def generate(self, image: Image.Image, instruction: str, *, seed: int, steps: int) -> Image.Image: ...


class GoalCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def generate(self, editor: ImageEditor, image_path: str | Path, instruction: str, camera_id: str,
                 *, seed: int, steps: int, extra_config: dict | None = None) -> tuple[GoalRecord, bool]:
        if not editor.revision or not editor.model_id or steps <= 0:
            raise ValueError("pinned model and positive inference steps required")
        image_hash = sha256_file(image_path)
        spec = {"model": editor.model_id, "revision": editor.revision, "adapter": editor.adapter_hash,
                "image_sha256": image_hash, "instruction": instruction, "camera": camera_id,
                "seed": seed, "steps": steps, "extra": extra_config or {}}
        key = digest_json(spec)
        meta, png = self.root / f"{key}.json", self.root / f"{key}.png"
        if meta.exists() and png.exists():
            record = json.loads(meta.read_text())
            if record["spec"] != spec or record["output_sha256"] != sha256_file(png):
                raise ValueError("goal cache corrupted")
            g = GoalRecord(**{**record["goal"], "path": str(png.resolve())})
            return g, True
        with Image.open(image_path) as src:
            start = time.perf_counter()
            goal = editor.generate(src.convert("RGB"), instruction, seed=seed, steps=steps)
            seconds = time.perf_counter() - start
        if not isinstance(goal, Image.Image):
            raise ValueError("editor must return a PIL image")
        tmp = self.root / f"{key}.tmp.png"
        goal.convert("RGB").save(tmp)
        tmp.replace(png)
        g = GoalRecord(str(png.resolve()), "generated", image_hash, camera_id, instruction, time.time(),
                       editor.model_id, editor.revision, editor.adapter_hash)
        write_json(meta, {"spec": spec, "goal": asdict(g), "output_sha256": sha256_file(png),
                          "inference_seconds": seconds, "quality_verified": False})
        return g, False


class DiffusersKleinEditor:
    """Optional adapter. Requires compatible installed Diffusers and local weights.
    No claim of robotics quality or AMD kernel qualification. Actual generator
    dependency/version and model revision are recorded by the caller.
    """
    def __init__(self, model_path: str, revision: str, *, device: str, dtype, adapter_path: str | None = None,
                 local_files_only: bool = True):
        if not revision:
            raise ValueError("model revision required")
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError("Install a Diffusers build exposing Flux2KleinPipeline; no fallback to another model") from exc
        self.model_id, self.revision = model_path, revision
        self.adapter_hash = sha256_file(adapter_path) if adapter_path else None
        self.device = device
        self.pipe = Flux2KleinPipeline.from_pretrained(model_path, revision=revision, torch_dtype=dtype, local_files_only=local_files_only).to(device)
        if adapter_path:
            self.pipe.load_lora_weights(str(Path(adapter_path).parent), weight_name=Path(adapter_path).name)

    def generate(self, image, instruction, *, seed, steps):
        import torch
        result = self.pipe(prompt=instruction, image=image, num_inference_steps=steps,
                           guidance_scale=1.0, generator=torch.Generator(device=self.device).manual_seed(seed))
        return result.images[0]


def tau_world_command(python: str, *, model: str, inputs: str, output: str, lora: str | None = None,
                      mode: str = "infer", steps: int | None = None) -> list[str]:
    """Build commands from the inspected Tau README; never invokes shell=True.
    For infer, --steps isn't in that README's public CLI; configure native
    inference-step overrides only after checking the pinned implementation.
    """
    if mode not in {"infer", "finetune"}:
        raise ValueError("unsupported Tau world mode")
    if mode == "finetune" and (steps is None or steps < 1):
        raise ValueError("positive train steps required")
    cmd = [python, "-m", f"tau0_world_model.{mode}", "--model", model, "--input", inputs, "--output", output]
    if lora:
        cmd += ["--lora", lora]
    if mode == "finetune":
        cmd += ["--steps", str(steps)]
    elif steps is not None:
        raise ValueError("do not invent Tau infer flags; inspect its --help")
    return cmd


def run_tau_world(command: list[str], *, repo: str | Path, timeout_seconds: float, execute: bool = False) -> dict:
    if not Path(repo).is_dir() or timeout_seconds <= 0:
        raise ValueError("valid repo and timeout required")
    if not execute:
        return {"command": command, "cwd": str(repo), "executed": False}
    run = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    if run.returncode:
        raise RuntimeError(f"Tau world failed ({run.returncode}): {run.stderr[-4000:]}")
    return {"command": command, "executed": True, "stdout": run.stdout[-4000:], "returncode": run.returncode}


def generator_eval_jobs(pair_manifest: str | Path) -> list[dict]:
    """No target image path is passed to the generator. Pair target stays evaluator-only."""
    root = Path(pair_manifest).parent
    jobs = []
    seen = set()
    for row in read_jsonl(pair_manifest):
        if row["id"] in seen:
            raise ValueError("duplicate goal-pair id")
        seen.add(row["id"])
        path = resolve_local(root, row["image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        jobs.append({"id": row["id"], "image": str(path), "instruction": row["instruction"],
                     "camera": row["camera"], "split_group": row["split_group"], "split": row["split"]})
    return jobs


def unchanged_region_mae(current: Image.Image, generated: Image.Image, change_mask) -> float:
    """Diagnostic only: penalizes spurious edits, not a measure of physical success."""
    import numpy as np
    a, b, mask = np.asarray(current.convert("RGB")), np.asarray(generated.convert("RGB")), np.asarray(change_mask, dtype=bool)
    if a.shape != b.shape or mask.shape != a.shape[:2] or mask.all():
        raise ValueError("aligned images and some unchanged pixels required")
    return float(np.abs(a.astype(float) - b.astype(float))[~mask].mean() / 255)
