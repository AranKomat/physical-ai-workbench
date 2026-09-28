"""GPU-dependent native adapters. Imports are lazy; no silent random fallback.

The W0 API was inspected at commit 90801baa3bdc7829c1e4989edfe3d0fb2410ea6a.
These adapters are NOT checkpoint-/GPU-qualified in this deliverable. Full W0
MoT behavior is not preserved merely by extracting its ActionDiT.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import torch
from torch import Tensor, nn
from .checkpoints import read_tensor_state, select_prefix, audited_load
from .flow import flow_sample, masked_mse, euler_sample
from .io import sha256_file


@dataclass(frozen=True)
class NativeFlowConvention:
    time_scale: float
    time_direction: str  # noise_to_data or data_to_noise
    velocity_sign: int
    verified_against_native: bool = False

    def validate(self):
        if self.time_scale <= 0 or self.time_direction not in {"noise_to_data", "data_to_noise"} or self.velocity_sign not in {-1, 1}:
            raise ValueError("explicit donor time scale, direction and velocity sign required")

    def time(self, t: Tensor) -> Tensor:
        self.validate()
        return (1 - t if self.time_direction == "data_to_noise" else t) * self.time_scale


class W0MotorAdapter(nn.Module):
    """Wraps the native ActionDiT; state is a new context token, not native W0 semantics."""
    def __init__(self, action_dit: nn.Module, context_dim: int, action_dim: int, convention: NativeFlowConvention):
        super().__init__()
        convention.validate()
        self.action_dit, self.convention = action_dit, convention
        self.state_adapter = nn.Linear(2 * action_dim, context_dim)

    def forward(self, x, t, context, context_mask, state, state_mask, action_mask):
        # Native standalone ActionDiT has no action-key-padding API. Drop incomplete
        # horizons rather than let padded steps affect valid action attention.
        if not bool(action_mask.any(-1).all()):
            raise ValueError("W0 native adapter requires complete horizons; filter tail windows")
        ref = next(self.action_dit.parameters())
        context = context.to(dtype=ref.dtype)
        state = state.to(dtype=self.state_adapter.weight.dtype).masked_fill(~state_mask, 0)
        st = self.state_adapter(torch.cat((state, state_mask.to(state.dtype)), -1))[:, None].to(context.dtype)
        context = torch.cat((context, st), 1)
        cmask = torch.cat((context_mask, torch.ones(len(context), 1, device=context.device, dtype=torch.bool)), 1)
        result = self.action_dit(action_tokens=x.to(dtype=ref.dtype).masked_fill(~action_mask, 0),
                                 timestep=self.convention.time(t).to(dtype=ref.dtype),
                                 context=context, context_mask=cmask)
        return (result * self.convention.velocity_sign).masked_fill(~action_mask, 0)


def load_w0_motor(*, checkpoint: str, constructor_config: dict, state_dict_key: str,
                  source_prefix: str, context_dim: int, convention: NativeFlowConvention,
                  device: str, dtype: torch.dtype, reset_action_io: bool,
                  checkpoint_sha256: str, qualification_mode: bool = False) -> tuple[W0MotorAdapter, dict]:
    """Requires the ACTUAL full Base action-state keys, not a Wan-only init payload.

    The caller inspects keys and sets state_dict_key/source_prefix explicitly.
    No interpolation or layer replication. Full source keys load before Qwen
    reconditioning; resets are recorded. Unverified convention only allowed for
    qualification work, not training.
    """
    if not Path(checkpoint).is_file() or not checkpoint_sha256:
        raise ValueError("local checkpoint and recorded SHA-256 required")
    actual_hash = sha256_file(checkpoint)
    if actual_hash != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    if not convention.verified_against_native and not qualification_mode:
        raise ValueError("verify donor flow convention before native training")
    try:
        from wam.model.modules.experts.action_dit import ActionDiT
    except ImportError as exc:
        raise RuntimeError("Install pinned InternW0-Delta in the isolated ROCm environment first") from exc
    action = ActionDiT(**constructor_config)
    state = select_prefix(read_tensor_state(checkpoint, state_dict_key), source_prefix)
    resets = ("action_encoder.*", "head.*") if reset_action_io else ()
    report = audited_load(action, state, reset_patterns=resets)
    replaced = action.configure_vlm_conditioning(context_dim)
    report.update(checkpoint_sha256=actual_hash, source_prefix=source_prefix, state_dict_key=state_dict_key,
                  new_cross_attention_blocks=replaced, action_semantics_reset=reset_action_io,
                  warning="Extracted standalone expert is not the intact W0 MoT policy. Validate transfer.")
    adapter = W0MotorAdapter(action, context_dim, constructor_config["action_dim"], convention)
    return adapter.to(device=device, dtype=dtype), report


class HuggingFaceQwen(nn.Module):
    """Generic Qwen HF wrapper; configuration support != proven Primus/Megatron parity."""
    def __init__(self, model: nn.Module, selected_layers: tuple[int, ...] = (-1,)):
        super().__init__()
        self.model, self.selected_layers = model, selected_layers
        config = getattr(model.config, "text_config", model.config)
        self.hidden_size = config.hidden_size

    @classmethod
    def from_pretrained(cls, path: str, *, revision: str | None, dtype=torch.bfloat16,
                        attention_backend: str = "eager", selected_layers=(-1,), local_files_only=True):
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError("Install the HF extra; transformers is intentionally optional for CPU tests") from exc
        if not local_files_only and not revision:
            raise ValueError("remote model loading requires an explicit revision")
        model = AutoModelForImageTextToText.from_pretrained(path, revision=revision, torch_dtype=dtype,
                  trust_remote_code=False, attn_implementation=attention_backend, local_files_only=local_files_only)
        return cls(model, tuple(selected_layers))

    def prefix(self, inputs: dict[str, Tensor]) -> tuple[list[Tensor], Tensor]:
        if "labels" in inputs:
            raise ValueError("motor prefix must not contain teacher-forced action/reasoning labels")
        kwargs = dict(inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            kwargs["logits_to_keep"] = 1
        output = self.model(**kwargs)
        hidden = getattr(output, "hidden_states", None)
        if hidden is None:
            raise RuntimeError("HF model did not expose hidden_states; adapt explicitly, do not substitute logits")
        layers = [hidden[i] for i in self.selected_layers]
        if any(x.shape[-1] != self.hidden_size for x in layers):
            raise RuntimeError("unexpected Qwen hidden state dimensions")
        mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
        if layers[0].shape[:2] != mask.shape:
            raise RuntimeError("expanded image-token mask mismatch; use the native Qwen processor")
        return layers, mask

    def auxiliary_ce(self, inputs: dict[str, Tensor]) -> Tensor:
        labels = inputs.get("labels")
        if labels is None or not bool((labels != -100).any()):
            raise ValueError("KI requires actual assistant-only token targets")
        result = self.model(**inputs, use_cache=False, return_dict=True)
        if result.loss is None or not bool(torch.isfinite(result.loss)):
            raise RuntimeError("invalid HF CE loss")
        return result.loss


class NativeQwenFlowPolicy(nn.Module):
    """Executable composition once local Qwen and qualified native motor are supplied.

    Batches provide SEPARATE prefix_inputs and auxiliary_inputs. The former must
    not contain ground-truth action tokens; otherwise the motor has target leakage.
    auxiliary_inputs can include versioned FAST/other action token targets and VL
    replay. Tokenizer preparation itself is not claimed reproduced here.
    """
    def __init__(self, brain: HuggingFaceQwen, motor: nn.Module, *, insulated: bool = True, aux_weight: float = 1.0):
        super().__init__()
        if insulated and aux_weight <= 0:
            raise ValueError("insulated trainable Qwen needs an auxiliary objective")
        self.brain, self.motor = brain, motor
        self.insulated, self.aux_weight = insulated, aux_weight
        self.layer_logits = nn.Parameter(torch.zeros(len(brain.selected_layers)))
        self.context_norm = nn.LayerNorm(brain.hidden_size)

    def context(self, batch):
        layers, mask = self.brain.prefix(batch["prefix_inputs"])
        if self.insulated:
            layers = [x.detach() for x in layers]
        mixed = (torch.stack(layers) * self.layer_logits.softmax(0)[:, None, None, None]).sum(0)
        return self.context_norm(mixed), mask

    def forward(self, batch: dict, noise: Tensor, time: Tensor):
        context, mask = self.context(batch)
        x, velocity = flow_sample(batch["actions"], batch["action_mask"], noise, time)
        pred = self.motor(x, time, context, mask, batch["state"], batch["state_mask"], batch["action_mask"])
        flow = masked_mse(pred, velocity, batch["action_mask"])
        aux = self.brain.auxiliary_ce(batch["auxiliary_inputs"]) if self.aux_weight else flow.new_zeros(())
        return {"total": flow + self.aux_weight * aux, "flow": flow, "aux_ce": aux}

    @torch.no_grad()
    def sample(self, batch, noise, steps=4, return_trace=False):
        context, mask = self.context(batch)
        fn = lambda x, t: self.motor(x, t, context, mask, batch["state"], batch["state_mask"], batch["action_mask"])
        return euler_sample(fn, noise, batch["action_mask"], steps, return_trace=return_trace)
