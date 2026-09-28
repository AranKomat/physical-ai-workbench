"""Small *reference* implementations, not pretrained Intern/Tau/Qwen replicas.

They make mask, gradient, optional-goal, sampling, staged-training and checkpoint
contracts executable on CPU. The same motor interface can wrap a native donor.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .flow import flow_sample, masked_mse, euler_sample

@dataclass(frozen=True)
class ReferenceConfig:
    action_dim: int = 80
    horizon: int = 4
    width: int = 32
    heads: int = 4
    brain_layers: int = 2
    motor_layers: int = 2
    vocab_size: int = 260
    aux_bins: int = 17
    max_text_tokens: int = 64
    mode: str = "ki"
    multilayer: bool = True
    aux_weight: float = 1.0

    def validate(self):
        if min(self.action_dim, self.horizon, self.width, self.heads, self.brain_layers, self.motor_layers) <= 0:
            raise ValueError("all model dimensions must be positive")
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be divisible by heads and 2")
        if self.mode not in {"ki", "joint", "frozen"}:
            raise ValueError("mode must be ki, joint or frozen")
        if self.aux_bins < 3 or self.aux_weight < 0 or self.vocab_size < 260 or self.max_text_tokens < 1:
            raise ValueError("invalid auxiliary/token configuration")
        if self.mode == "ki" and self.aux_weight <= 0:
            raise ValueError("KI requires a separate robot-relevant backbone objective, not just detach()")


def byte_tokens(texts: list[str], limit: int = 64) -> tuple[Tensor, Tensor]:
    """CPU reference only. Production must use the selected Qwen processor."""
    if not texts or limit < 1:
        raise ValueError("nonempty texts and positive limit required")
    ids = torch.zeros((len(texts), limit), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, text in enumerate(texts):
        seq = [257] + [b + 1 for b in text.encode("utf-8")][:limit - 1]
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = True
    return ids, mask


def sinusoidal(t: Tensor, width: int) -> Tensor:
    scale = torch.exp(-math.log(10000) * torch.arange(width // 2, device=t.device).float() / max(1, width // 2 - 1))
    args = t.float()[:, None] * scale[None, :]
    return torch.cat((args.sin(), args.cos()), -1)


class ReferenceBrain(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        w = cfg.width
        self.text = nn.Embedding(cfg.vocab_size, w)
        self.position = nn.Embedding(cfg.max_text_tokens, w)
        self.vision = nn.Sequential(nn.Conv2d(3, w, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 2)))
        self.role = nn.Embedding(4, w)  # current, past, goal, crop
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(w, cfg.heads, 2 * w, dropout=0, batch_first=True, norm_first=True) for _ in range(cfg.brain_layers)])

    def forward(self, input_ids: Tensor, text_mask: Tensor, images: Tensor, image_mask: Tensor, image_roles: Tensor) -> tuple[list[Tensor], Tensor]:
        if input_ids.ndim != 2 or text_mask.shape != input_ids.shape or text_mask.dtype != torch.bool:
            raise ValueError("bad text mask")
        if images.ndim != 5 or images.shape[2] != 3 or image_mask.shape != images.shape[:2] or image_roles.shape != image_mask.shape:
            raise ValueError("images must be [B,N,3,H,W] with masks/roles [B,N]")
        if image_mask.dtype != torch.bool or bool(((image_roles < 0) | (image_roles > 3)).any()):
            raise ValueError("bad image mask/role")
        if not bool(text_mask.any(1).all()) or not bool(torch.isfinite(images).all()):
            raise ValueError("empty text context or nonfinite images")
        b, n = images.shape[:2]
        vision = self.vision(images.reshape(b * n, *images.shape[2:])).flatten(2).transpose(1, 2)
        vision = vision.reshape(b, n, 4, -1) + self.role(image_roles)[:, :, None]
        text = self.text(input_ids) + self.position(torch.arange(input_ids.shape[1], device=input_ids.device))[None]
        x = torch.cat((text, vision.flatten(1, 2)), 1)
        mask = torch.cat((text_mask, image_mask.repeat_interleave(4, 1)), 1)
        layers = []
        for block in self.blocks:
            x = block(x, src_key_padding_mask=~mask).masked_fill(~mask[..., None], 0)
            layers.append(x)
        return layers, mask


class LayerMixer(nn.Module):
    """Simple learned scalar mixing, NOT a reproduction of LayerRoute or M²-VLA."""
    def __init__(self, layer_count: int, width: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(layer_count))
        self.norm = nn.LayerNorm(width)

    def forward(self, layers: list[Tensor]) -> Tensor:
        if len(layers) != len(self.logits):
            raise ValueError("wrong number of layer states")
        return self.norm((torch.stack(layers) * self.logits.softmax(0)[:, None, None, None]).sum(0))


class MotorBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(width) for _ in range(3)]
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: Tensor, context: Tensor, action_valid: Tensor, context_mask: Tensor) -> Tensor:
        z = self.norm1(x)
        x = x + self.self_attn(z, z, z, key_padding_mask=~action_valid, need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), context, context, key_padding_mask=~context_mask, need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class FlowMotor(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.action_in = nn.Linear(cfg.action_dim, cfg.width)
        self.state_in = nn.Linear(cfg.action_dim * 2, cfg.width)
        self.mask_in = nn.Linear(cfg.action_dim, cfg.width)
        self.context_in = nn.Linear(cfg.width, cfg.width)
        self.time_in = nn.Sequential(nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.position = nn.Parameter(torch.randn(1, cfg.horizon, cfg.width) * 0.01)
        self.blocks = nn.ModuleList([MotorBlock(cfg.width, cfg.heads) for _ in range(cfg.motor_layers)])
        self.output = nn.Linear(cfg.width, cfg.action_dim)

    def forward(self, x: Tensor, t: Tensor, context: Tensor, context_mask: Tensor, state: Tensor, state_mask: Tensor, action_mask: Tensor) -> Tensor:
        b, h, d = x.shape
        if h > self.cfg.horizon or d != self.cfg.action_dim or action_mask.shape != x.shape or t.shape != (b,):
            raise ValueError("bad motor action/time shape")
        if state.shape != (b, d) or state_mask.shape != state.shape or not bool(context_mask.any(1).all()):
            raise ValueError("bad motor state/context mask")
        valid = action_mask.any(-1)
        if not bool(valid.any(-1).all()):
            raise ValueError("empty action sample")
        state_value = state.masked_fill(~state_mask, 0)
        context = self.context_in(context)
        z = self.action_in(x.masked_fill(~action_mask, 0)) + self.mask_in(action_mask.to(x.dtype))
        z = z + self.state_in(torch.cat((state_value, state_mask.to(state.dtype)), -1))[:, None]
        z = z + self.time_in(sinusoidal(t, self.cfg.width).to(x.dtype))[:, None] + self.position[:, :h]
        for block in self.blocks:
            z = block(z, context, valid, context_mask)
        return self.output(z).masked_fill(~action_mask, 0)


class ScalarAuxiliaryCodec:
    """Diagnostic scalar bins; explicitly NOT FAST/ActionCodec and no compression claim.
    Packs active channels only. Production KI should use a frozen, versioned
    tokenizer and actual Qwen next-token CE on physically aligned actions.
    """
    def __init__(self, bins: int = 17):
        self.bins, self.pad, self.bos, self.eos = bins, bins, bins + 1, bins + 2

    @property
    def vocab_size(self):
        return self.bins + 3

    def encode(self, actions: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        values = actions.detach()[mask]
        if not bool(torch.isfinite(values).all()) or bool((values.abs() > 1.000001).any()):
            raise ValueError("aux codec requires explicitly normalized active actions in [-1,1]")
        ids = ((actions.detach().float() + 1) * 0.5 * (self.bins - 1)).round().long()
        sequences = [torch.cat((ids[i][mask[i]], ids.new_tensor([self.eos]))) for i in range(len(ids))]
        max_len = max(map(len, sequences))
        labels = ids.new_full((len(ids), max_len), -100)
        shifted = ids.new_full((len(ids), max_len), self.pad)
        valid = torch.zeros_like(shifted, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            labels[i, :len(seq)] = seq
            shifted[i, 0] = self.bos
            shifted[i, 1:len(seq)] = seq[:-1]
            valid[i, :len(seq)] = True
        return shifted, labels, valid


class AuxiliaryARHead(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.codec = ScalarAuxiliaryCodec(cfg.aux_bins)
        self.embed = nn.Embedding(self.codec.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.horizon * cfg.action_dim + 1, cfg.width)
        self.decoder = nn.TransformerDecoderLayer(cfg.width, cfg.heads, 2 * cfg.width, dropout=0, batch_first=True, norm_first=True)
        self.head = nn.Linear(cfg.width, self.codec.vocab_size)

    def forward(self, context: Tensor, context_mask: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
        shifted, labels, valid = self.codec.encode(actions, action_mask)
        n = shifted.shape[1]
        x = self.embed(shifted) + self.position(torch.arange(n, device=shifted.device))[None]
        causal = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.decoder(x, context, tgt_mask=causal, tgt_key_padding_mask=~valid, memory_key_padding_mask=~context_mask)
        logits = self.head(x)
        return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)


class ReferenceVLA(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.brain = ReferenceBrain(cfg)
        self.mixer = LayerMixer(cfg.brain_layers if cfg.multilayer else 1, cfg.width)
        self.motor = FlowMotor(cfg)
        self.aux = AuxiliaryARHead(cfg)
        if cfg.mode == "frozen":
            self.brain.requires_grad_(False)

    def forward(self, batch: dict[str, Tensor], noise: Tensor, time: Tensor):
        return self.losses(batch, noise, time)

    def encode(self, batch: dict[str, Tensor], *, allow_goals: bool = True):
        mask = batch["image_mask"]
        if not allow_goals:
            mask = mask & (batch["image_roles"] != 2)
        return self.brain(batch["input_ids"], batch["text_mask"], batch["images"], mask, batch["image_roles"])

    def flow_context(self, layers: list[Tensor]):
        layers = layers if self.cfg.multilayer else [layers[-1]]
        if self.cfg.mode in {"ki", "frozen"}:
            layers = [x.detach() for x in layers]  # detach BEFORE trainable bridge/mixer
        return self.mixer(layers)

    def losses(self, batch: dict[str, Tensor], noise: Tensor, time: Tensor) -> dict[str, Tensor]:
        layers, cmask = self.encode(batch)
        context = self.flow_context(layers)
        x, target = flow_sample(batch["actions"], batch["action_mask"], noise, time)
        pred = self.motor(x, time, context, cmask, batch["state"], batch["state_mask"], batch["action_mask"])
        flow = masked_mse(pred, target, batch["action_mask"])
        # Future targets must not leak into the auxiliary backbone objective.
        # Same prefix without goal tokens is recomputed only if goals are present.
        has_goals = bool(((batch["image_roles"] == 2) & batch["image_mask"]).any())
        aux_layers, aux_mask = self.encode(batch, allow_goals=False) if has_goals else (layers, cmask)
        aux = self.aux(aux_layers[-1], aux_mask, batch["actions"], batch["action_mask"]) if self.cfg.aux_weight > 0 else flow.new_zeros(())
        return {"total": flow + self.cfg.aux_weight * aux, "flow": flow, "aux_ce": aux, "velocity": pred}

    @torch.no_grad()
    def sample(self, batch: dict[str, Tensor], noise: Tensor, steps: int = 4, *, return_trace=False):
        layers, mask = self.encode(batch)
        context = self.flow_context(layers)
        def velocity(x, t):
            return self.motor(x, t, context, mask, batch["state"], batch["state_mask"], batch["action_mask"])
        return euler_sample(velocity, noise, batch["action_mask"], steps, return_trace=return_trace)
