"""COLMAP camera-pose stage (offline SfM).

Runs / reads a COLMAP reconstruction to recover GoPro intrinsics + per-frame
extrinsics. COLMAP is a system binary (BSD), not a pip package. Provide an
existing reconstruction via ``params.sparse_dir`` (``cameras.bin``, ``images.bin``)
or let the stage shell out to ``colmap`` if installed.
"""

from __future__ import annotations

from pathlib import Path

from ...schema.core import ClipAnnotation
from ..base import POSE
from .base import PoseStageBase


@POSE.register("colmap")
class COLMAP(PoseStageBase):
    name = "colmap"
    requires = ()  # external `colmap` binary; pycolmap optional
    extra = "pose"
    license = "BSD (COLMAP)"
    license_url = "https://colmap.github.io/"

    def estimate(self, clip: ClipAnnotation) -> None:
        sparse = self.param("sparse_dir")
        if not sparse or not Path(sparse).exists():
            raise NotImplementedError(
                "Provide a prebuilt COLMAP model via params.sparse_dir (cameras.bin "
                "+ images.bin), or run COLMAP first. Then map each image's "
                "qvec/tvec (world->cam) and the camera intrinsics into CameraParams. "
                "Use `pycolmap.Reconstruction(sparse_dir)` to read the model."
            )
        self._load_reconstruction(Path(sparse), clip)

    def _load_reconstruction(self, sparse: Path, clip: ClipAnnotation) -> None:  # pragma: no cover
        import pycolmap

        rec = pycolmap.Reconstruction(str(sparse))
        # Map rec.images[*].cam_from_world (world->cam) + rec.cameras into our schema.
        raise NotImplementedError(
            f"Reconstruction with {rec.num_images()} images loaded; map COLMAP "
            "cam_from_world poses + intrinsics onto clip.frames[*].camera."
        )
