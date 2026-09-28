from dataclasses import replace
import torch
import pytest
from physical_ai.models import ReferenceConfig, ReferenceVLA, ScalarAuxiliaryCodec
from physical_ai.flow import flow_sample, masked_mse, euler_sample
from physical_ai.training import synthetic_batch, set_motor_stage, optimizer_groups, LoRALinear


def grad_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

@pytest.mark.parametrize("mode,expected", [("ki", False), ("joint", True), ("frozen", False)])
def test_motor_gradient_boundary(mode, expected):
    cfg = ReferenceConfig(mode=mode)
    model, batch = ReferenceVLA(cfg), synthetic_batch(cfg)
    loss = model.losses(batch, torch.randn_like(batch["actions"]), torch.ones(4) * .5)
    loss["flow"].backward()
    assert (grad_sum(model.brain) > 0) == expected
    assert grad_sum(model.motor) > 0
    assert grad_sum(model.mixer) > 0  # Detach must not also freeze the new bridge.


def test_aux_ce_adapts_brain_when_insulated():
    cfg = ReferenceConfig(mode="ki")
    model, batch = ReferenceVLA(cfg), synthetic_batch(cfg)
    loss = model.losses(batch, torch.randn_like(batch["actions"]), torch.ones(4) * .5)
    loss["aux_ce"].backward()
    assert grad_sum(model.brain) > 0 and grad_sum(model.motor) == 0


def test_detach_without_objective_is_not_ki():
    with pytest.raises(ValueError): ReferenceConfig(mode="ki", aux_weight=0).validate()


def test_goal_dropout_removes_all_goal_information():
    cfg = ReferenceConfig(); model = ReferenceVLA(cfg).eval(); batch = synthetic_batch(cfg)
    a, mask_a = model.encode(batch, allow_goals=False)
    changed = {k: v.clone() for k, v in batch.items()}
    changed["images"][:, 2] = 200.0
    changed["image_mask"][:, 2] = True
    b, mask_b = model.encode(changed, allow_goals=False)
    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(a[-1], b[-1], atol=1e-6)


def test_aux_ce_never_uses_privileged_goal():
    cfg = ReferenceConfig(); model = ReferenceVLA(cfg).eval(); batch = synthetic_batch(cfg)
    batch["image_mask"][:, 2] = True
    noise = torch.randn_like(batch["actions"]); t = torch.full((4,), .5)
    a = model.losses(batch, noise, t)["aux_ce"]
    batch["images"][:, 2] = 100
    b = model.losses(batch, noise, t)["aux_ce"]
    assert torch.allclose(a, b, atol=1e-6)


def test_masked_loss_independent_of_padding_dimension():
    pred = torch.ones(2, 4, 7); target = torch.zeros_like(pred); mask = torch.ones_like(pred, dtype=torch.bool)
    small = masked_mse(pred, target, mask)
    p2 = torch.cat([pred, torch.zeros(2, 4, 73)], -1); t2 = torch.zeros_like(p2)
    m2 = torch.cat([mask, torch.zeros(2, 4, 73, dtype=torch.bool)], -1)
    assert masked_mse(p2, t2, m2) == small


def test_per_example_loss_not_weighted_by_dof_count():
    target = torch.zeros(2, 1, 80); pred = torch.ones_like(target)
    pred[0] = 2; mask = torch.zeros_like(target, dtype=torch.bool); mask[0, :, :1] = True; mask[1] = True
    assert masked_mse(pred, target, mask) == 2.5


def test_flow_sign_and_endpoints():
    a, n = torch.ones(2, 3, 5), torch.zeros(2, 3, 5); m = torch.ones_like(a, dtype=torch.bool)
    x, v = flow_sample(a, m, n, torch.tensor([0., 1.]))
    assert torch.equal(x[0], n[0]) and torch.equal(x[1], a[1]) and (v == 1).all()
    sample = euler_sample(lambda x, t: torch.ones_like(x), n, m, 4)
    assert torch.equal(sample, a)


def test_sampler_keeps_inactive_zero():
    cfg = ReferenceConfig(); model = ReferenceVLA(cfg).eval(); batch = synthetic_batch(cfg)
    actions = model.sample(batch, torch.randn_like(batch["actions"]), 2)
    assert torch.count_nonzero(actions[~batch["action_mask"]]) == 0


def test_zero_target_mask_rejected():
    a = torch.zeros(2, 3, 5); m = torch.zeros_like(a, dtype=torch.bool)
    with pytest.raises(ValueError): masked_mse(a, a, m)


def test_aux_teacher_forcing_shift():
    codec = ScalarAuxiliaryCodec(17)
    a = torch.tensor([[[0., .5, -1.]]]); mask = torch.tensor([[[True, False, True]]])
    shifted, labels, valid = codec.encode(a, mask)
    assert labels.tolist() == [[8, 0, codec.eos]]
    assert shifted.tolist() == [[codec.bos, 8, 0]]
    assert valid.all()


def test_aux_range_not_silently_clipped():
    with pytest.raises(ValueError): ScalarAuxiliaryCodec().encode(torch.tensor([[[2.]]]), torch.ones(1, 1, 1, dtype=torch.bool))


def test_staged_unfreeze_preserves_trunk():
    cfg = ReferenceConfig(); model = ReferenceVLA(cfg); batch = synthetic_batch(cfg)
    opt = torch.optim.AdamW(optimizer_groups(model))
    before = {n: p.detach().clone() for n, p in model.motor.named_parameters()}
    set_motor_stage(model.motor, "interfaces")
    model.losses(batch, torch.randn_like(batch["actions"]), torch.ones(4) * .5)["total"].backward(); opt.step()
    assert torch.equal(model.motor.blocks[0].self_attn.in_proj_weight, before["blocks.0.self_attn.in_proj_weight"])
    assert not torch.equal(model.motor.output.weight, before["output.weight"])
    set_motor_stage(model.motor, "all")
    assert all(p.requires_grad for p in model.motor.parameters())
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))


def test_lora_zero_init_preserves_exact_base():
    base = torch.nn.Linear(8, 8); x = torch.randn(4, 8); y = base(x).detach()
    adapted = LoRALinear(base, 2)
    assert torch.equal(y, adapted(x))
    assert all(not p.requires_grad for p in adapted.base.parameters())
