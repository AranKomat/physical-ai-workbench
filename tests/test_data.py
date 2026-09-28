import numpy as np
import pytest
from physical_ai.schema import ActionContract, ChannelMap, map_channels, PhysicalScaler
from physical_ai.data import (load_episode, save_episode, anchor_indices, history_indices, action_window,
                              select_future_goal, validate_disjoint_splits, split_for_group, FamilySampler,
                              goal_pair_manifest, quality_report, Segment)
from physical_ai.examples import synthetic_episode, write_demo_dataset
from physical_ai.io import read_jsonl, resolve_local


def test_episode_roundtrip(tmp_path):
    ep = synthetic_episode()
    p = save_episode(ep, tmp_path)
    other = load_episode(p)
    assert other.task == ep.task
    assert np.array_equal(ep.actions, other.actions)
    assert np.array_equal(ep.cameras["head"], other.cameras["head"])
    assert other.contract == ep.contract

@pytest.mark.parametrize("issue", ["nan", "padding", "clock", "mask", "camera"])
def test_episode_rejects_invalid(issue):
    ep = synthetic_episode()
    if issue == "nan": ep.actions[0, 0] = np.nan
    if issue == "padding": ep.actions[0, 79] = 1
    if issue == "clock": ep.timestamps[1] = ep.timestamps[0]
    if issue == "mask": ep.action_mask[0] = False
    if issue == "camera": ep.cameras["head"] = ep.cameras["head"][:-1]
    with pytest.raises(ValueError): ep.validate()


def test_camera_contract_is_not_assumed():
    with pytest.raises(ValueError): ActionContract(frame="camera", control_hz=10).validate()
    ep = synthetic_episode()
    ep.contract = ActionContract(frame="camera", control_hz=10, reference_camera="head", calibration_valid=True)
    with pytest.raises(ValueError): ep.validate()
    ep.calibration["head"] = {"intrinsics": np.eye(3).tolist(), "camera_from_base": np.eye(4).tolist()}
    ep.validate()


def test_channel_mapping_explicit_and_reversible_gripper():
    a, m = map_channels(np.array([[1., .25]]), [ChannelMap(1, 6, -1, 1), ChannelMap(0, 0, .01)])
    assert a[0, 6] == .75 and np.isclose(a[0, 0], .01)
    assert m.sum() == 2 and np.count_nonzero(a[~m]) == 0
    with pytest.raises(ValueError): map_channels(np.ones((3, 2)), [ChannelMap(0, 0), ChannelMap(1, 0)])


def test_scaler_preserves_shared_semantics():
    scaler = PhysicalScaler(np.zeros(80), np.ones(80) * .1, "global-v1")
    a, m = map_channels(np.array([[.03]]), [ChannelMap(0, 0)])
    z = scaler.encode(a, m)
    assert np.isclose(z[0, 0], .3)
    assert np.allclose(scaler.decode(z, m), a)
    a[0, 0] = .2
    assert scaler.encode(a, m)[0, 0] == 2  # no silent clipping


def test_history_never_uses_future():
    t = np.arange(40) / 10
    ids = history_indices(t, 3, (1, .2, 0))
    assert ids == [None, 1, 3]
    assert anchor_indices(t, 1).tolist() == [0, 10, 20, 30]


def test_horizon_padding_masked():
    ep = synthetic_episode()
    a, mask = action_window(ep, 38, 4)
    assert a.shape == (4, 80) and mask[2:].sum() == 0 and np.count_nonzero(a[2:]) == 0


def test_future_stays_inside_segment():
    ep = synthetic_episode(); rng = np.random.default_rng(10)
    for _ in range(30):
        i = select_future_goal(ep, 15, rng, endpoint_probability=.5)
        assert 15 < i < 20
    assert select_future_goal(ep, 19, rng) is None


def test_goal_manifest_has_no_cross_split_pairs(tmp_path):
    rows = goal_pair_manifest([(synthetic_episode(0), "train"), (synthetic_episode(1), "test")], tmp_path, 2)
    assert len(rows) == 12
    assert all(r["target_time"] > r["start_time"] for r in rows)
    assert all(r["goal_kind"] == "real_future_privileged" for r in rows)
    assert len(list(read_jsonl(tmp_path / "train.jsonl"))) == 6


def test_split_alias_leak_rejected():
    with pytest.raises(ValueError):
        validate_disjoint_splits([{"episode_id": "raw", "split_group": "physical-episode", "split": "train"},
                                  {"episode_id": "annotation", "split_group": "physical-episode", "split": "test"}])
    assert split_for_group("episode") == split_for_group("episode")


def test_family_sampler_renormalizes_available_families():
    rows = [{"family": "robot", "id": 1}, {"family": "umi", "id": 2}]
    sampler = FamilySampler(rows, {"robot": .8, "umi": .08, "h2r": .1, "ego": .02}, seed=4)
    assert set(sampler.missing) == {"h2r", "ego"}
    samples = sampler.sample(1000)
    assert sum(x["id"] == 1 for x in samples) > 850


def test_quality_does_not_invent_success():
    report = quality_report(synthetic_episode())
    assert report["quality"] == "unknown" and "success" not in report


def test_unknown_quality_not_false_failure():
    Segment(0, 1, "grasp", None, None).validate()
    with pytest.raises(ValueError): Segment(0, 1, "grasp", True).validate()


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError): resolve_local(tmp_path, "../outside")
