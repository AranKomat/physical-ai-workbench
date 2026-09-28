"""Versioned action contract. Our research schema is not claimed byte-identical
with InternW0, Tau0, Qwen-RobotManip or any checkpoint's native slot ordering.
Native/checkpoint layouts MUST use explicit mapping specs, never padding guesses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SCHEMA_ID = "pai.eef80.v1"
ACTION_DIM = 80
GROUPS = {
    "left_eef": (0, 6), "left_gripper": (6, 7),
    "right_eef": (7, 13), "right_gripper": (13, 14),
    "base": (14, 17), "left_hand": (17, 37), "right_hand": (37, 57),
    "native": (57, 80),
}
VALID_FRAMES = {"camera", "base", "native_joint"}

@dataclass(frozen=True)
class ActionContract:
    frame: str
    control_hz: float
    representation: str = SCHEMA_ID
    reference_camera: str | None = None
    calibration_valid: bool = False
    rotation: str = "spatial_rotvec_radians"
    delta_reference: str = "successive_target"
    gripper: str = "zero_closed_one_open"

    def validate(self) -> None:
        if self.frame not in VALID_FRAMES or not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("invalid frame or control frequency")
        if self.representation != SCHEMA_ID:
            raise ValueError("explicit conversion to pai.eef80.v1 required")
        if self.frame == "camera" and (not self.reference_camera or not self.calibration_valid):
            raise ValueError("camera-frame labels require verified calibration and reference camera")
        if self.rotation != "spatial_rotvec_radians" or self.delta_reference not in {"successive_target", "anchor_state"}:
            raise ValueError("unsupported rotation/delta convention")
        if self.gripper != "zero_closed_one_open":
            raise ValueError("normalize gripper convention explicitly")

@dataclass(frozen=True)
class ChannelMap:
    """Explicit native field/channel -> canonical slot affine mapping."""
    source_index: int
    target_index: int
    scale: float = 1.0
    offset: float = 0.0


def map_channels(values: np.ndarray, mappings: list[ChannelMap], dim: int = ACTION_DIM) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("native values must be finite [T,D]")
    targets = [m.target_index for m in mappings]
    if len(set(targets)) != len(targets) or not mappings:
        raise ValueError("mapping must be nonempty and may not overwrite a destination")
    out = np.zeros((len(values), dim), dtype=np.float32)
    mask = np.zeros_like(out, dtype=bool)
    for m in mappings:
        if not 0 <= m.source_index < values.shape[1] or not 0 <= m.target_index < dim:
            raise ValueError("channel index outside source/destination")
        if not np.isfinite([m.scale, m.offset]).all() or m.scale == 0:
            raise ValueError("invalid affine mapping")
        out[:, m.target_index] = values[:, m.source_index] * m.scale + m.offset
        mask[:, m.target_index] = True
    return out, mask

@dataclass
class PhysicalScaler:
    """Global per-semantic-channel scales, never per-robot quantiles for aligned channels.

    Ranges are configured/calibrated on train data, not guessed from test data.
    This implementation does not silently clip outliers.
    """
    center: np.ndarray
    scale: np.ndarray
    version: str = "unconfigured"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        if self.center.ndim != 1 or self.scale.shape != self.center.shape or not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("scaler requires finite vector center and positive scales")

    def encode(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x, mask = np.asarray(x), np.asarray(mask, dtype=bool)
        if x.shape != mask.shape or x.shape[-1] != len(self.scale) or not np.isfinite(x[mask]).all():
            raise ValueError("invalid scaler input")
        return np.where(mask, (np.where(mask, x, self.center) - self.center) / self.scale, 0).astype(np.float32)

    def decode(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if x.shape != mask.shape or x.shape[-1] != len(self.scale) or not np.isfinite(x[mask]).all():
            raise ValueError("invalid scaler input")
        return np.where(mask, x * self.scale + self.center, 0).astype(np.float32)
