"""Camera models: projection, unprojection, undistortion."""

from __future__ import annotations

import numpy as np

from ego2dex.io.camera import (
    Fisheye624Camera,
    FisheyeKBCamera,
    PinholeCamera,
    build_camera,
    gopro_default_intrinsics,
    undistort_image,
)
from ego2dex.schema.core import CameraModelType, CameraParams


def test_pinhole_no_distortion_matches_K():
    cam = PinholeCamera(fx=100, fy=120, cx=32, cy=24)
    pts = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 2.0]])
    uv = cam.project(pts)
    # for the on-axis point, projection == principal point
    assert np.allclose(uv[0], [32, 24])
    # second point matches the pinhole equation
    assert np.allclose(uv[1], [100 * 0.05 + 32, 120 * -0.1 + 24])


def test_pinhole_unproject_roundtrip():
    cam = PinholeCamera(fx=80, fy=80, cx=20, cy=15)
    pts = np.array([[0.2, -0.1, 1.5], [-0.3, 0.25, 2.0]])
    uv = cam.project(pts)
    back = cam.unproject(uv, depth=pts[:, 2])
    assert np.allclose(back, pts, atol=1e-6)


def test_fisheye_kb_center_and_monotonic():
    cam = FisheyeKBCamera(fx=120, fy=120, cx=64, cy=48, distortion=np.array([0.0, 0, 0, 0]))
    assert np.allclose(cam.project(np.array([[0, 0, 1.0]]))[0], [64, 48])
    # larger angle -> farther from center
    near = np.linalg.norm(cam.project(np.array([[0.1, 0, 1.0]]))[0] - [64, 48])
    far = np.linalg.norm(cam.project(np.array([[0.5, 0, 1.0]]))[0] - [64, 48])
    assert far > near


def test_fisheye624_from_15_params_center():
    params = np.zeros(15)
    params[0] = 150.0  # f
    params[1], params[2] = 70.0, 50.0  # cx, cy
    cam = Fisheye624Camera.from_params(params)
    assert np.allclose(cam.project(np.array([[0, 0, 1.0]]))[0], [70, 50])


def test_build_camera_dispatch():
    p = CameraParams(
        model=CameraModelType.FISHEYE624,
        intrinsics={"f": 150.0, "cx": 70, "cy": 50},
        distortion=[0.0] * 12,
    )
    cam = build_camera(p)
    assert isinstance(cam, Fisheye624Camera)


def test_undistort_pinhole_zero_distortion_is_noop_shape():
    img = (np.random.RandomState(0).rand(48, 64, 3) * 255).astype(np.uint8)
    params = gopro_default_intrinsics(64, 48)
    out, new_K = undistort_image(img, params)
    assert out.shape == img.shape
    assert new_K.shape == (3, 3)
