"""Explicit SE(3) conventions for action conversion.

T_camera_from_base maps a base-frame POINT to camera coordinates. An EEF-origin
translation difference and a spatial rotation increment rotate by R only (no
translation cross term). This is NOT an origin-referenced spatial twist.
For a moving wrist camera, freeze the reference transform at observation t.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def check_rotation(r: NDArray) -> NDArray:
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation must be finite 3x3")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1.0, atol=1e-5):
        raise ValueError("rotation must be orthonormal with determinant +1")
    return r


def check_transform(t: NDArray) -> NDArray:
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("transform must be a finite homogeneous 4x4 matrix")
    check_rotation(t[:3, :3])
    return t


def skew(v: NDArray) -> NDArray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def so3_exp(rotvec: NDArray) -> NDArray:
    v = np.asarray(rotvec, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("rotvec must be finite length 3")
    theta = np.linalg.norm(v)
    k = skew(v)
    if theta < 1e-7:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + np.sin(theta) / theta * k + (1 - np.cos(theta)) / theta**2 * (k @ k)


def so3_log(r: NDArray) -> NDArray:
    r = check_rotation(r)
    cosine = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-7:
        return v * 0.5
    if np.pi - theta < 1e-5:
        # The eigenvector of eigenvalue 1 is the rotation axis. At exactly pi
        # either sign is valid; near pi choose the sign consistent with sin(theta).
        _, vecs = np.linalg.eigh((r + r.T) / 2)
        axis = vecs[:, -1]
        if np.dot(axis, v) < 0:
            axis = -axis
        return theta * axis
    return theta / (2 * np.sin(theta)) * v


def invert_transform(t: NDArray) -> NDArray:
    t = check_transform(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def eef_delta(current: NDArray, target: NDArray, camera_from_base: NDArray | None = None) -> NDArray:
    """Return [delta position (m), spatial rotation vector (rad)].

    Rotation increment is R_target @ R_current.T, never Euler subtraction.
    Both poses are base-from-EEF transforms; optional camera is held fixed.
    """
    current, target = check_transform(current), check_transform(target)
    dp = target[:3, 3] - current[:3, 3]
    dr = so3_log(target[:3, :3] @ current[:3, :3].T)
    if camera_from_base is not None:
        r = check_transform(camera_from_base)[:3, :3]
        dp, dr = r @ dp, r @ dr
    return np.concatenate([dp, dr])


def apply_eef_delta(current: NDArray, delta: NDArray, camera_from_base: NDArray | None = None) -> NDArray:
    current = check_transform(current)
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != (6,) or not np.isfinite(d).all():
        raise ValueError("delta must be finite length 6")
    dp, dr = d[:3], d[3:]
    if camera_from_base is not None:
        r = check_transform(camera_from_base)[:3, :3]
        dp, dr = r.T @ dp, r.T @ dr
    target = current.copy()
    target[:3, 3] += dp
    target[:3, :3] = so3_exp(dr) @ current[:3, :3]
    return target


def transform_points(points: NDArray, t: NDArray) -> NDArray:
    t = check_transform(t)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must end in dimension 3 and be finite")
    return points @ t[:3, :3].T + t[:3, 3]


def project_points(points_camera: NDArray, intrinsics: NDArray) -> tuple[NDArray, NDArray]:
    p, k = np.asarray(points_camera), np.asarray(intrinsics)
    if p.shape[-1] != 3 or k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(p).all():
        raise ValueError("invalid points/intrinsics")
    valid = p[..., 2] > 1e-8
    uvw = p @ k.T
    uv = np.full((*p.shape[:-1], 2), np.nan)
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, valid
