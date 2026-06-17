"""Camera models: projection, unprojection, undistortion.

Three families, matching the ingestion targets:

  * **Pinhole** (OpenCV) -- ``fx,fy,cx,cy`` + Brown-Conrady ``[k1,k2,p1,p2,k3,...]``.
  * **Fisheye / Kannala-Brandt** (OpenCV) -- ``fx,fy,cx,cy`` + ``D=[k1,k2,k3,k4]``.
  * **Fisheye624 / RadTanThinPrism** (Project Aria) -- single focal, 15 params:
    ``[f, cx, cy, k0..k5, p0, p1, s0..s3]`` (6 radial + 2 tangential + 4 thin-prism).

Forward projection for all three is pure numpy (no OpenCV needed). Image-level
undistortion uses OpenCV where a closed form exists (pinhole, KB). GoPro is a
wide-FOV camera -> use :func:`undistort_image` to rectify to a pinhole view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..schema.core import CameraModelType, CameraParams

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
@dataclass
class CameraModel:
    """Abstract-ish camera. Subclasses implement :meth:`project`."""

    width: int | None = None
    height: int | None = None

    def project(self, points_cam: NDArray) -> NDArray:  # pragma: no cover - abstract
        raise NotImplementedError

    def K(self) -> NDArray:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Pinhole (Brown-Conrady)
# --------------------------------------------------------------------------- #
@dataclass
class PinholeCamera(CameraModel):
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    # Brown-Conrady [k1, k2, p1, p2, k3, (k4, k5, k6)]
    distortion: NDArray = field(default_factory=lambda: np.zeros(5))

    def K(self) -> NDArray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def project(self, points_cam: NDArray) -> NDArray:
        p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = np.clip(p[:, 2], EPS, None)
        x = p[:, 0] / z
        y = p[:, 1] / z
        d = np.zeros(8)
        d[: len(self.distortion)] = self.distortion[:8]
        k1, k2, p1, p2, k3, k4, k5, k6 = d
        r2 = x * x + y * y
        radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
        x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        u = self.fx * x_d + self.cx
        v = self.fy * y_d + self.cy
        return np.stack([u, v], axis=-1)

    def unproject(self, pixels: NDArray, depth: NDArray | float = 1.0) -> NDArray:
        """Pixel(s) + depth -> 3D camera points (assumes distortion already removed)."""
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        z = np.broadcast_to(np.asarray(depth, dtype=np.float64), (px.shape[0],))
        x = (px[:, 0] - self.cx) / self.fx
        y = (px[:, 1] - self.cy) / self.fy
        return np.stack([x * z, y * z, z], axis=-1)


# --------------------------------------------------------------------------- #
# Fisheye / Kannala-Brandt (OpenCV fisheye)
# --------------------------------------------------------------------------- #
@dataclass
class FisheyeKBCamera(CameraModel):
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: NDArray = field(default_factory=lambda: np.zeros(4))  # [k1,k2,k3,k4]

    def K(self) -> NDArray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def project(self, points_cam: NDArray) -> NDArray:
        p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = np.clip(p[:, 2], EPS, None)
        x = p[:, 0] / z
        y = p[:, 1] / z
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        k1, k2, k3, k4 = (list(self.distortion) + [0, 0, 0, 0])[:4]
        theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        scale = np.where(r > EPS, theta_d / np.clip(r, EPS, None), 1.0)
        u = self.fx * (x * scale) + self.cx
        v = self.fy * (y * scale) + self.cy
        return np.stack([u, v], axis=-1)


# --------------------------------------------------------------------------- #
# Aria Fisheye624 / FisheyeRadTanThinPrism
# --------------------------------------------------------------------------- #
@dataclass
class Fisheye624Camera(CameraModel):
    """Aria 15-param model: [f, cx, cy, k0..k5, p0, p1, s0..s3].

    6 radial (on the angle theta), 2 tangential, 4 thin-prism. Single focal.
    Follows projectaria-tools' FisheyeRadTanThinPrism projection.
    """

    f: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    radial: NDArray = field(default_factory=lambda: np.zeros(6))  # k0..k5
    tangential: NDArray = field(default_factory=lambda: np.zeros(2))  # p0,p1
    thin_prism: NDArray = field(default_factory=lambda: np.zeros(4))  # s0..s3

    @classmethod
    def from_params(cls, params: NDArray, width=None, height=None) -> Fisheye624Camera:
        p = np.asarray(params, dtype=np.float64).reshape(-1)
        if p.shape[0] < 15:
            p = np.concatenate([p, np.zeros(15 - p.shape[0])])
        return cls(
            width=width,
            height=height,
            f=float(p[0]),
            cx=float(p[1]),
            cy=float(p[2]),
            radial=p[3:9],
            tangential=p[9:11],
            thin_prism=p[11:15],
        )

    def K(self) -> NDArray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], dtype=np.float64)

    def project(self, points_cam: NDArray) -> NDArray:
        p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = np.clip(p[:, 2], EPS, None)
        a = p[:, 0] / z
        b = p[:, 1] / z
        r = np.sqrt(a * a + b * b)
        th = np.arctan(r)
        k = (list(self.radial) + [0] * 6)[:6]
        th_d = th * (
            1
            + k[0] * th**2
            + k[1] * th**4
            + k[2] * th**6
            + k[3] * th**8
            + k[4] * th**10
            + k[5] * th**12
        )
        scale = np.where(r > EPS, th_d / np.clip(r, EPS, None), 1.0)
        xr = a * scale
        yr = b * scale
        rd2 = xr * xr + yr * yr
        p0, p1 = (list(self.tangential) + [0, 0])[:2]
        s0, s1, s2, s3 = (list(self.thin_prism) + [0, 0, 0, 0])[:4]
        x_d = xr + (2 * p0 * xr * yr + p1 * (rd2 + 2 * xr * xr)) + s0 * rd2 + s1 * rd2**2
        y_d = yr + (p0 * (rd2 + 2 * yr * yr) + 2 * p1 * xr * yr) + s2 * rd2 + s3 * rd2**2
        u = self.f * x_d + self.cx
        v = self.f * y_d + self.cy
        return np.stack([u, v], axis=-1)


# --------------------------------------------------------------------------- #
# Factory + image undistortion
# --------------------------------------------------------------------------- #
def build_camera(params: CameraParams) -> CameraModel:
    """Instantiate a concrete camera model from :class:`CameraParams`."""
    i = params.intrinsics
    if params.model in (CameraModelType.PINHOLE,):
        return PinholeCamera(
            width=params.width,
            height=params.height,
            fx=i.get("fx", 0.0),
            fy=i.get("fy", 0.0),
            cx=i.get("cx", 0.0),
            cy=i.get("cy", 0.0),
            distortion=np.asarray(params.distortion or [0, 0, 0, 0, 0], dtype=np.float64),
        )
    if params.model == CameraModelType.FISHEYE_KB:
        return FisheyeKBCamera(
            width=params.width,
            height=params.height,
            fx=i.get("fx", 0.0),
            fy=i.get("fy", 0.0),
            cx=i.get("cx", 0.0),
            cy=i.get("cy", 0.0),
            distortion=np.asarray(params.distortion or [0, 0, 0, 0], dtype=np.float64),
        )
    if params.model in (CameraModelType.FISHEYE624, CameraModelType.RADTAN_THINPRISM):
        # intrinsics may carry {f,cx,cy}; distortion carries the 12 coeffs.
        f = i.get("f", i.get("fx", 0.0))
        cx, cy = i.get("cx", 0.0), i.get("cy", 0.0)
        full = np.concatenate([[f, cx, cy], np.asarray(params.distortion, dtype=np.float64)])
        return Fisheye624Camera.from_params(full, width=params.width, height=params.height)
    raise ValueError(f"Unsupported camera model: {params.model}")


def undistort_image(image: NDArray, params: CameraParams, balance: float = 0.0):
    """Rectify an image to a pinhole view. Returns ``(undistorted, new_K)``.

    Supports pinhole (Brown-Conrady) and fisheye/KB via OpenCV. Fisheye624 is
    not closed-form-invertible here -- use :class:`Fisheye624Camera` for points.
    """
    import cv2  # core dep

    img = np.asarray(image)
    h, w = img.shape[:2]
    if params.model == CameraModelType.PINHOLE:
        cam = build_camera(params)
        K = cam.K()
        dist = np.asarray(params.distortion or [0, 0, 0, 0, 0], dtype=np.float64)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), balance, (w, h))
        return cv2.undistort(img, K, dist, None, new_K), new_K
    if params.model == CameraModelType.FISHEYE_KB:
        cam = build_camera(params)
        K = cam.K()
        D = np.asarray(params.distortion or [0, 0, 0, 0], dtype=np.float64).reshape(4, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=balance
        )
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR), new_K
    raise NotImplementedError(
        f"Image undistortion for {params.model} is not implemented; project points "
        "with the corresponding CameraModel instead."
    )


def gopro_default_intrinsics(width: int, height: int, hfov_deg: float = 118.0) -> CameraParams:
    """A reasonable pinhole guess for a GoPro 'Wide' frame when EXIF is missing.

    Real intrinsics should come from calibration or SLAM. fx is derived from the
    horizontal FOV; distortion is left zero (use SLAM/COLMAP to recover it).
    """
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return CameraParams(
        model=CameraModelType.PINHOLE,
        width=width,
        height=height,
        intrinsics={"fx": float(fx), "fy": float(fx), "cx": width / 2.0, "cy": height / 2.0},
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
