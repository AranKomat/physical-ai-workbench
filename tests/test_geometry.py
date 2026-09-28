import numpy as np
import pytest
from physical_ai.geometry import (so3_exp, so3_log, eef_delta, apply_eef_delta, check_transform,
                                  check_rotation, invert_transform, transform_points, project_points)

@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.001, 0.7, 2.3, np.pi - 1e-7, np.pi])
def test_rotation_roundtrip(angle):
    axis = np.array([1, 2, -3], float); axis /= np.linalg.norm(axis)
    r = so3_exp(axis * angle)
    assert np.allclose(so3_exp(so3_log(r)), r, atol=2e-7)

@pytest.mark.parametrize("camera_rot", [0, 0.8, 2.1])
def test_camera_delta_roundtrip(camera_rot):
    current = np.eye(4); current[:3, :3] = so3_exp(np.array([.1, .2, .3])); current[:3, 3] = [.5, .1, .2]
    target = np.eye(4); target[:3, :3] = so3_exp(np.array([.2, .5, -.2])); target[:3, 3] = [.51, .08, .23]
    cam = np.eye(4); cam[:3, :3] = so3_exp(np.array([0, 0, camera_rot])); cam[:3, 3] = [10, 20, 30]
    d = eef_delta(current, target, cam)
    assert np.allclose(apply_eef_delta(current, d, cam), target, atol=1e-7)
    cam2 = cam.copy(); cam2[:3, 3] = 0
    assert np.allclose(eef_delta(current, target, cam2), d)  # no incorrect adjoint cross-term

@pytest.mark.parametrize("bad", [np.ones((3, 3)), np.diag([1, 1, -1]), np.eye(4)])
def test_invalid_rotation(bad):
    with pytest.raises(ValueError): check_rotation(bad)


def test_transform_inverse():
    t = np.eye(4); t[:3, :3] = so3_exp(np.array([.2, -.3, .5])); t[:3, 3] = [4, 2, 1]
    points = np.array([[1, 2, 3], [2, 4, 6]])
    assert np.allclose(transform_points(transform_points(points, t), invert_transform(t)), points)


def test_projection_rejects_behind_camera():
    uv, valid = project_points(np.array([[1., 2., 2.], [1., 2., -1.]]), np.diag([10, 10, 1]))
    assert valid.tolist() == [True, False]
    assert np.allclose(uv[0], [5, 10]) and np.isnan(uv[1]).all()
