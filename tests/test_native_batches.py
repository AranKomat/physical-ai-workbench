from dataclasses import replace

import numpy as np
from PIL import Image
import pytest

from physical_ai.native_batches import NativeWindow, NativeWindowCollator
from test_collation import Processor


def window():
    return NativeWindow("pinned-source", "ep0", "native-test", "norm-hash", 0, 3,
                        np.array([0., .1, .2]), ("front",), (0.,),
                        (Image.new("RGB", (8, 8)),), "lift", "action", "codec-rev",
                        np.zeros(2), np.ones(2, bool), np.zeros((3, 2)), np.ones((3, 2), bool))


def test_batch_preserves_native_fields():
    batch = NativeWindowCollator(Processor())([window(), replace(window(), episode="ep1")])
    assert batch["actions"].shape == (2, 3, 2)
    assert batch["provenance"][1]["episode"] == "ep1"
    assert "labels" not in batch["prefix_inputs"]


@pytest.mark.parametrize("changes,match", [
    ({"anchor": 1}, "episode boundary"),
    ({"camera_times": (.1,)}, "future"),
    ({"timestamps": np.zeros(3)}, "increasing"),
    ({"action_mask": np.zeros((3, 2), bool)}, "all timesteps"),
    ({"state": np.array([np.nan, 0.])}, "nonfinite"),
])
def test_invalid_native_windows(changes, match):
    with pytest.raises(ValueError, match=match):
        replace(window(), **changes).validate()


def test_contract_mix_rejected():
    with pytest.raises(ValueError, match="mix"):
        NativeWindowCollator(Processor())([window(), replace(window(), schema="other")])
