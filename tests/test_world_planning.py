from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from physical_ai.world import (GoalRecord, GoalCache, tau_world_command, generator_eval_jobs, unchanged_region_mae)
from physical_ai.planning import (ExecutionMemory, ReplanScheduler, compact_subtask, crop_normalized, bbox_iou,
                                  corrupt_memory, subtask_duration_summary, tau_proposal_payload)
from physical_ai.data import Segment
from physical_ai.io import write_jsonl

@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "current.png"; Image.new("RGB", (20, 20), (50, 60, 70)).save(p); return p


def test_future_goal_never_deployment(image_path):
    goal = GoalRecord(str(image_path), "real_future_privileged", "obs", "head", "move cup", 1)
    goal.validate_for("train", camera_id="head")
    goal.validate_for("oracle_eval", camera_id="head")
    for mode in ("deployment", "deployment_eval"):
        with pytest.raises(ValueError): goal.validate_for(mode, camera_id="head")


def test_generated_goal_identity_and_age(image_path):
    g = GoalRecord(str(image_path), "generated", "obs", "head", "move cup", 1, "model", "revision")
    g.validate_for("deployment", camera_id="head", now=2, max_age=4)
    with pytest.raises(ValueError): g.validate_for("deployment", camera_id="left")
    with pytest.raises(ValueError): g.validate_for("deployment", camera_id="head", now=10, max_age=4)


class StubEditor:
    model_id, revision, adapter_hash = "synthetic-test-editor", "fixed-test", None
    def __init__(self): self.calls = 0
    def generate(self, image, instruction, *, seed, steps):
        self.calls += 1; return image.copy()


def test_goal_cache_content_addressed(tmp_path, image_path):
    cache = GoalCache(tmp_path / "cache"); editor = StubEditor()
    first, hit = cache.generate(editor, image_path, "grasp", "head", seed=4, steps=4)
    second, hit2 = cache.generate(editor, image_path, "grasp", "head", seed=4, steps=4)
    assert not hit and hit2 and first.path == second.path and editor.calls == 1
    cache.generate(editor, image_path, "release", "head", seed=4, steps=4)
    assert editor.calls == 2
    editor.adapter_hash = "new-adapter"
    cache.generate(editor, image_path, "grasp", "head", seed=4, steps=4)
    assert editor.calls == 3


def test_generator_never_gets_target_file(tmp_path, image_path):
    manifest = tmp_path / "pairs.jsonl"
    write_jsonl(manifest, [{"id": "x", "image": image_path.name, "target": "future.png", "instruction": "grasp",
                            "camera": "head", "split_group": "e", "split": "test"}])
    jobs = generator_eval_jobs(manifest)
    assert "target" not in jobs[0] and "future.png" not in json.dumps(jobs)


def test_tau_commands_not_shell_strings():
    cmd = tau_world_command("python", model="model", inputs="pairs.jsonl", output="out", lora="lora", mode="finetune", steps=20)
    assert "tau0_world_model.finetune" in cmd and cmd[-2:] == ["--steps", "20"]
    with pytest.raises(ValueError): tau_world_command("python", model="m", inputs="p", output="o", steps=4)


def test_unchanged_region_metric():
    im = Image.new("RGB", (10, 10), (50, 50, 50)); mask = np.zeros((10, 10), bool)
    assert unchanged_region_mae(im, im, mask) == 0
    with pytest.raises(ValueError): unchanged_region_mae(im, im, np.ones((10, 10), bool))


def test_memory_never_assumes_proposed_action_succeeded():
    m = ExecutionMemory(); m.begin("clean"); m.propose("grasp cup")
    assert not m.completed
    m.observe_result(completed=False, failure="slipped")
    assert not m.completed and len(m.failures) == 1
    m.observe_result(completed=True)
    assert m.completed == ["grasp cup"]
    m.begin("different task")
    assert not m.completed and not m.failures


def test_replan_event_driven_with_timeout():
    s = ReplanScheduler(10)
    assert s.should_replan(0)
    s.mark_planned(0)
    assert not s.should_replan(1) and s.should_replan(1, failure=True) and s.should_replan(10)
    with pytest.raises(ValueError): s.should_replan(-1)


def test_memory_corruption():
    assert corrupt_memory(["grasp", "lift"], ["place"], "lagging")[0] == ["grasp"]
    assert corrupt_memory(["grasp"], ["place"], "optimistic")[0] == ["grasp", "place"]


def test_compact_subtask_no_silent_truncation():
    assert compact_subtask(" grasp   cup ") == "grasp cup"
    with pytest.raises(ValueError): compact_subtask("x" * 181)


def test_grounding_crop_validation(image_path):
    with Image.open(image_path) as im:
        crop = crop_normalized(im, [.25, .25, .75, .75], padding=0)
        assert crop.size == (10, 10)
        with pytest.raises(ValueError): crop_normalized(im, [0, 0, 2, 1])
    assert bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1


def test_tau_payload_camera_contract(image_path):
    payload = tau_proposal_payload("grasp", "", {k: image_path for k in ("head", "left", "right")})
    assert payload["memory"] == "(empty)" and payload["task_type"] == "subtask_only"
    with pytest.raises(ValueError): tau_proposal_payload("t", "", {"head": image_path})


def test_average_subtask_duration_measured_not_assumed():
    report = subtask_duration_summary([Segment(0, 2, "a"), Segment(2, 8, "b")])
    assert report["median_seconds"] == 4
