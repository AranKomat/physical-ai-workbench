import json
import numpy as np
import pytest
import torch
from physical_ai.examples import write_demo_dataset
from physical_ai.schema import PhysicalScaler
from physical_ai.dataset import EpisodeWindowDataset
from physical_ai.budget import TokenBudget, token_fraction_from_example_mix, generation_cost
from physical_ai.validation import make_fixture, run_probe, compare_probes, compare_arrays


def test_dataset_pipeline(tmp_path):
    manifest = write_demo_dataset(tmp_path / "data")
    scale = PhysicalScaler(np.zeros(80), np.ones(80), "test-only")
    ds = EpisodeWindowDataset(manifest, split="train", horizon=4, scaler=scale, goal_fraction=1, goal_mode="train")
    b = ds[0]
    assert b["images"].shape == (9, 3, 32, 32) and b["actions"].shape == (4, 80)
    assert b["image_mask"][b["image_roles"] == 2].any()
    assert torch.equal(ds[0]["images"], b["images"])
    with pytest.raises(ValueError): EpisodeWindowDataset(manifest, split="test", horizon=4, scaler=scale, goal_fraction=1, goal_mode="train")
    with pytest.raises(ValueError): EpisodeWindowDataset(manifest, split="test", horizon=4, scaler=scale, use_subtasks=True)


def test_budget_includes_vision_and_explicit_fraction():
    result = TokenBudget().estimate()
    assert result["context_tokens_per_window_estimate"] == 726
    assert result["robot_context_tokens_per_source_hour_one_pass"] == 2613600
    assert token_fraction_from_example_mix(.9, 1000, 2000) == pytest.approx(.81818181818)
    assert generation_cost(1000, 1.25, 4)["gpu_hours"] == pytest.approx(1.3888888889)


def test_probe_cpu_self_parity_is_not_gpu_qualification(tmp_path):
    f, a, b = [tmp_path / name for name in ("fixture", "a", "b")]
    make_fixture(f)
    run_probe(f, a, learning_steps=2); run_probe(f, b, learning_steps=2)
    report = compare_probes(a, b, tmp_path / "result.json", atol=0, rtol=0)
    assert report["numeric_pass"] and not report["cuda_rocm_pair"] and not report["native_vla_qualified"]


def test_fp8_not_silently_accepted(tmp_path):
    make_fixture(tmp_path / "f")
    with pytest.raises(ValueError): run_probe(tmp_path / "f", tmp_path / "p", precision="fp8")

@pytest.mark.parametrize("candidate", [np.array([np.nan]), np.array([2]), np.ones(2)])
def test_compare_fails_corruption(candidate):
    assert not compare_arrays(np.array([1]), candidate, atol=0, rtol=0)["passed"]


def test_different_protocol_probe_not_compared(tmp_path):
    f, a, b = [tmp_path / name for name in ("fixture", "a", "b")]
    make_fixture(f); run_probe(f, a, learning_steps=1); run_probe(f, b, learning_steps=2)
    with pytest.raises(ValueError): compare_probes(a, b, tmp_path / "result.json")
