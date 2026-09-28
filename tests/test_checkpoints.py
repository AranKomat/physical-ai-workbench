import torch
import pytest
from physical_ai.checkpoints import read_tensor_state, select_prefix, audited_load, atomic_torch_save


def test_exact_checkpoint_load(tmp_path):
    module = torch.nn.Linear(3, 2)
    state = module.state_dict()
    path = tmp_path / "test.pt"
    atomic_torch_save({"model": state}, path)
    new = torch.nn.Linear(3, 2)
    report = audited_load(new, read_tensor_state(path, "model"))
    assert report["coverage"] == 1 and torch.equal(module.weight, new.weight)

@pytest.mark.parametrize("failure", ["missing", "shape", "extra", "nan"])
def test_bad_checkpoint_rejected(failure):
    module = torch.nn.Linear(3, 2)
    state = {k: v.clone() for k, v in module.state_dict().items()}
    if failure == "missing": del state["weight"]
    if failure == "shape": state["weight"] = torch.ones(9, 9)
    if failure == "extra": state["mystery"] = torch.zeros(1)
    if failure == "nan": state["weight"][0, 0] = torch.nan
    before = module.weight.detach().clone()
    with pytest.raises(ValueError): audited_load(module, state)
    assert torch.equal(before, module.weight)  # atomic failure


def test_explicit_reset_audited():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    state = {k: v.clone() for k, v in m.state_dict().items()}; state["1.weight"] = torch.ones(9, 9)
    before = m[1].weight.detach().clone()
    report = audited_load(m, state, reset_patterns=("1.*",))
    assert report["reset"] == ["1.bias", "1.weight"] and torch.equal(before, m[1].weight)
    with pytest.raises(ValueError): audited_load(m, state, reset_patterns=("typo.*",))


def test_prefix_not_guessed():
    with pytest.raises(ValueError): select_prefix({"brain.a": torch.ones(1)}, "motor.")
    assert list(select_prefix({"motor.a": torch.ones(1)}, "motor.")) == ["a"]
