"""Planning arithmetic, not a GPU throughput benchmark. Fractions have explicit units."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class TokenBudget:
    width: int = 256
    height: int = 256
    image_count: int = 9
    text_tokens: int = 150
    anchor_hz: float = 1.0
    patch_size: int = 16
    merge_size: int = 2
    measured_total_tokens_per_day: float = 2e9
    robot_token_fraction: float = 0.9

    def estimate(self) -> dict:
        values = [self.width, self.height, self.image_count, self.anchor_hz, self.patch_size, self.merge_size, self.measured_total_tokens_per_day]
        if any(not math.isfinite(v) or v <= 0 for v in values) or self.text_tokens < 0 or not 0 < self.robot_token_fraction <= 1:
            raise ValueError("invalid token budget")
        divisor = self.patch_size * self.merge_size
        visual = math.ceil(self.width / divisor) * math.ceil(self.height / divisor)
        per_window = self.image_count * visual + self.text_tokens
        per_hour = 3600 * self.anchor_hz * per_window
        return {**asdict(self), "visual_tokens_per_image_estimate": visual,
                "context_tokens_per_window_estimate": per_window,
                "robot_context_tokens_per_source_hour_one_pass": per_hour,
                "source_hours_per_day_one_pass": self.measured_total_tokens_per_day * self.robot_token_fraction / per_hour,
                "warning": "2B/day is the user's hypothetical 27B throughput, NOT measured here. Processor counts and motor/vision FLOPs must be profiled. 90% EXAMPLES is not 90% TOKENS."}


def token_fraction_from_example_mix(robot_example_fraction: float, robot_length: float, vl_length: float) -> float:
    if not 0 <= robot_example_fraction <= 1 or robot_length <= 0 or vl_length <= 0:
        raise ValueError("invalid example mix")
    r = robot_example_fraction
    return r * robot_length / (r * robot_length + (1 - r) * vl_length)


def generation_cost(unique_requests: int, seconds_per_request: float, gpus_per_request: int = 1) -> dict:
    if unique_requests < 0 or seconds_per_request <= 0 or gpus_per_request < 1:
        raise ValueError("invalid generation estimate")
    return {"unique_requests": unique_requests, "serial_wall_hours": unique_requests * seconds_per_request / 3600,
            "gpu_hours": unique_requests * seconds_per_request * gpus_per_request / 3600,
            "warning": "Latency from another GPU/model is not a throughput estimate for MI300X. Count unique requests, views, seeds, resampling and epochs separately."}
