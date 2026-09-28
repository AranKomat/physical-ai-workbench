"""Session preflight tests. They do not execute any GPU job or download."""
import copy
import importlib.util
from pathlib import Path
import pytest
from physical_ai.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan():
    return {"ready_for_cuda": True, "required_files": [],
            "jobs": [{"name": "native_probe", "argv": ["python", "-V"], "timeout_seconds": 30}]}


def test_session_all_jobs_preflight_before_execution():
    m = script("run_nvidia_window")
    plan = valid_plan()
    m.validate_plan(plan, 120, execute=True)
    plan["jobs"].append({"name": "later", "argv": "not-an-argument-array"})
    with pytest.raises(ValueError):
        m.validate_plan(plan, 120, execute=True)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_session_rejects_bad_total_budget(budget):
    with pytest.raises(ValueError):
        script("run_nvidia_window").validate_plan(valid_plan(), budget, execute=False)


@pytest.mark.parametrize("name", ["../overwrite", "", "bad/name"])
def test_session_rejects_unsafe_log_names(name):
    plan = valid_plan()
    plan["jobs"][0]["name"] = name
    with pytest.raises(ValueError):
        script("run_nvidia_window").validate_plan(plan, 120, execute=False)


def test_session_ready_gate_and_nonempty_jobs():
    m = script("run_nvidia_window")
    plan = valid_plan()
    plan["ready_for_cuda"] = False
    m.validate_plan(plan, 120, execute=False)
    with pytest.raises(RuntimeError):
        m.validate_plan(plan, 120, execute=True)
    plan["jobs"] = []
    with pytest.raises(ValueError):
        m.validate_plan(plan, 120, execute=False)


def test_session_artifact_hashes_checked(tmp_path):
    m = script("run_nvidia_window")
    asset = tmp_path / "small_fixture"
    asset.write_text("deterministic input")
    plan = valid_plan()
    plan["required_files"] = [{"path": str(asset), "sha256": sha256_file(asset)}]
    m.validate_plan(plan, 120, execute=False)
    asset.write_text("changed")
    with pytest.raises(RuntimeError):
        m.validate_plan(plan, 120, execute=False)


def test_session_duplicate_job_and_invalid_timeout():
    m = script("run_nvidia_window")
    plan = valid_plan()
    plan["jobs"].append(copy.deepcopy(plan["jobs"][0]))
    with pytest.raises(ValueError):
        m.validate_plan(plan, 120, execute=False)
    plan = valid_plan()
    plan["jobs"][0]["timeout_seconds"] = float("inf")
    with pytest.raises(ValueError):
        m.validate_plan(plan, 120, execute=False)


def test_bootstrap_requires_immutable_commit(tmp_path):
    m = script("bootstrap_upstreams")
    with pytest.raises(ValueError):
        m.commands({"name": "example", "revision": "main", "url": "https://github.com/example/repo.git"}, tmp_path)
    dest, commands = m.commands({"name": "example", "revision": "a" * 40,
                                "url": "https://github.com/example/repo.git"}, tmp_path)
    assert dest == tmp_path / "example"
    assert commands[2][-1] == "a" * 40
    assert "--detach" in commands[3]
