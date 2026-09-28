from types import SimpleNamespace
import torch
import pytest
from physical_ai.native import NativeFlowConvention, W0MotorAdapter, HuggingFaceQwen, NativeQwenFlowPolicy


def test_donor_time_direction_explicit():
    t = torch.tensor([0., 1.])
    c = NativeFlowConvention(1000, "data_to_noise", -1)
    assert torch.equal(c.time(t), torch.tensor([1000., 0.]))
    with pytest.raises(ValueError): NativeFlowConvention(1, "unknown", 1).validate()


class FakeNativeAction(torch.nn.Module):
    def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.ones(())); self.last = None
    def forward(self, action_tokens, timestep, context, context_mask):
        self.last = (timestep, context, context_mask)
        return action_tokens * self.weight


def test_w0_adapter_mask_state_and_velocity_contract():
    native = FakeNativeAction(); m = W0MotorAdapter(native, 8, 3, NativeFlowConvention(1000, "data_to_noise", -1))
    x = torch.ones(2, 4, 3); mask = torch.ones_like(x, dtype=torch.bool); mask[:, :, -1] = False
    out = m(x, torch.zeros(2), torch.zeros(2, 6, 8), torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool), mask)
    assert (out[:, :, :2] == -1).all() and (out[:, :, 2] == 0).all()
    assert native.last[1].shape == (2, 7, 8)
    assert (native.last[0] == 1000).all()
    mask[:, -1] = False
    with pytest.raises(ValueError): m(x, torch.zeros(2), torch.zeros(2, 6, 8), torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool), mask)


class FakeHF(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8)); self.embedding = torch.nn.Embedding(10, 8)
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        h = self.embedding(input_ids)
        return SimpleNamespace(hidden_states=(h, 2 * h), loss=h.square().mean() if labels is not None else None)


def test_hf_prefix_cannot_receive_labels():
    h = HuggingFaceQwen(FakeHF())
    ids = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(ValueError): h.prefix({"input_ids": ids, "labels": ids})
    layers, mask = h.prefix({"input_ids": ids})
    assert layers[0].shape == (2, 3, 8) and mask.all()


def test_native_auxiliary_requires_supervision():
    h = HuggingFaceQwen(FakeHF())
    ids = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(ValueError): h.auxiliary_ce({"input_ids": ids, "labels": torch.full_like(ids, -100)})
    assert h.auxiliary_ce({"input_ids": ids, "labels": ids}).item() > 0
