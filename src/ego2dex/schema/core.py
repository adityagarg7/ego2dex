"""The unified ego2dex annotation schema (pydantic v2).

Three tiers, modeled as a superset of what Ego4D / Ego-Exo4D / H2O / DexYCB /
HOI4D / OakInk2 / AssemblyHands / ARCTIC / HOT3D provide (see docs/datasets.md):

  * per-frame geometric  (HandPose, Detection, Mask, CameraParams, Gaze, ...)
  * interaction          (HandObjectInteraction, ActiveObject, ...)
  * semantic / temporal  (Tags, Caption, ActionSegment, Narration, ...)

Containers: ``FrameAnnotation`` and ``ClipAnnotation``.

All numeric fields are plain Python lists for clean JSON. Pass numpy arrays to
constructors freely -- a ``model_validator`` coerces ``ndarray``/np-scalars to
builtins. Use ``.as_numpy()`` helpers to get arrays back.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..topology import (
    MANO_NUM_BETAS,
    MANO_POSE_DIM,
    NUM_HAND_KEYPOINTS,
    HandConvention,
)

SCHEMA_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class ContactState(str, Enum):
    """100DOH 5-class contact taxonomy (hand_object_detector)."""

    NO_CONTACT = "no_contact"
    SELF_CONTACT = "self_contact"
    OTHER_PERSON = "other_person"
    PORTABLE_OBJECT = "portable_object"
    STATIONARY_OBJECT = "stationary_object"


class CameraModelType(str, Enum):
    PINHOLE = "pinhole"  # fx,fy,cx,cy + Brown-Conrady [k1,k2,p1,p2,k3,...]
    FISHEYE_KB = "fisheye_kb"  # OpenCV fisheye / Kannala-Brandt, D=[k1,k2,k3,k4]
    FISHEYE624 = "fisheye624"  # Aria Fisheye624 (15 params, single focal)
    RADTAN_THINPRISM = "radtan_thinprism"  # Aria FisheyeRadTanThinPrism


# --------------------------------------------------------------------------- #
# Base model with numpy coercion
# --------------------------------------------------------------------------- #
def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


class Ego2DexModel(BaseModel):
    """Base: forbids unknown fields and coerces numpy inputs to builtins."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_numpy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: _to_builtin(v) for k, v in data.items()}
        return data

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, text: str | bytes) -> Ego2DexModel:
        return cls.model_validate_json(text)


# --------------------------------------------------------------------------- #
# Per-frame geometric
# --------------------------------------------------------------------------- #
class MANOParams(Ego2DexModel):
    """MANO parameters. ``pose`` is the 45-d articulated pose (NO global)."""

    global_orient: list[float] = Field(..., description="axis-angle, len 3")
    pose: list[float] = Field(..., description="15x3 axis-angle, len 45 (no global)")
    betas: list[float] = Field(..., description="shape, len 10")
    trans: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    is_pca: bool = Field(default=False, description="True if `pose` is in PCA subspace")
    num_pca_comps: int | None = None

    @field_validator("global_orient", "trans")
    @classmethod
    def _len3(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"expected length 3, got {len(v)}")
        return v

    @field_validator("pose")
    @classmethod
    def _pose_dim(cls, v: list[float]) -> list[float]:
        # 45 = full articulated axis-angle. PCA poses may be shorter; allow that.
        if len(v) not in (MANO_POSE_DIM,) and len(v) > MANO_POSE_DIM:
            raise ValueError(
                f"MANO pose must be {MANO_POSE_DIM} (or <= for PCA), got {len(v)}. "
                "Did you accidentally include the 3-d global_orient (=> 48)?"
            )
        return v

    @field_validator("betas")
    @classmethod
    def _betas_dim(cls, v: list[float]) -> list[float]:
        if len(v) != MANO_NUM_BETAS:
            raise ValueError(f"betas must be {MANO_NUM_BETAS}, got {len(v)}")
        return v

    def full_pose(self) -> list[float]:
        """Concatenate to the 48-d ``[global(3), pose(45)]`` representation."""
        return list(self.global_orient) + list(self.pose)


class HandPose(Ego2DexModel):
    """A single hand in one frame: 2D/3D keypoints (+ optional MANO + mesh)."""

    side: HandSide = HandSide.UNKNOWN
    keypoints_2d: list[list[float]] = Field(..., description="21 x [x, y, conf] in pixels")
    keypoints_3d: list[list[float]] | None = Field(
        default=None, description="21 x [x, y, z] (camera frame, meters)"
    )
    mano: MANOParams | None = None
    vertices: list[list[float]] | None = Field(
        default=None, description="778 x 3 MANO mesh vertices (optional)"
    )
    keypoint_convention: HandConvention = HandConvention.STANDARD21
    score: float = 1.0

    @field_validator("keypoints_2d")
    @classmethod
    def _kp2d(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != NUM_HAND_KEYPOINTS:
            raise ValueError(f"keypoints_2d must have {NUM_HAND_KEYPOINTS} points, got {len(v)}")
        for p in v:
            if len(p) != 3:
                raise ValueError("each 2D keypoint must be [x, y, conf]")
        return v

    @field_validator("keypoints_3d")
    @classmethod
    def _kp3d(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return v
        if len(v) != NUM_HAND_KEYPOINTS:
            raise ValueError(f"keypoints_3d must have {NUM_HAND_KEYPOINTS} points, got {len(v)}")
        for p in v:
            if len(p) != 3:
                raise ValueError("each 3D keypoint must be [x, y, z]")
        return v

    def kp2d_array(self) -> np.ndarray:
        return np.asarray(self.keypoints_2d, dtype=np.float64)

    def kp3d_array(self) -> np.ndarray | None:
        return (
            None if self.keypoints_3d is None else np.asarray(self.keypoints_3d, dtype=np.float64)
        )

    @classmethod
    def from_arrays(
        cls,
        keypoints_2d: np.ndarray,
        side: HandSide | str = HandSide.UNKNOWN,
        keypoints_3d: np.ndarray | None = None,
        **kwargs: Any,
    ) -> HandPose:
        return cls(
            side=HandSide(side) if not isinstance(side, HandSide) else side,
            keypoints_2d=np.asarray(keypoints_2d),
            keypoints_3d=None if keypoints_3d is None else np.asarray(keypoints_3d),
            **kwargs,
        )


class Detection(Ego2DexModel):
    """An open-vocabulary detection box (COCO bbox = [x, y, w, h], top-left)."""

    label: str
    score: float = 1.0
    bbox: list[float] = Field(..., description="[x, y, w, h], top-left, 0-indexed")
    instance_id: int | None = None

    @field_validator("bbox")
    @classmethod
    def _bbox4(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox must be [x, y, w, h]")
        return v


class Mask(Ego2DexModel):
    """An instance mask stored as COCO RLE; ``instance_id`` is stable across frames."""

    size: list[int] = Field(..., description="[height, width]")
    counts: Any = Field(..., description="COCO RLE counts: str (compressed) or list[int]")
    instance_id: int | None = None
    label: str | None = None
    score: float = 1.0

    @field_validator("size")
    @classmethod
    def _size2(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError("size must be [height, width]")
        return v

    def rle(self) -> dict[str, Any]:
        return {"size": list(self.size), "counts": self.counts}

    def decode(self) -> np.ndarray:
        from .rle import decode_mask

        return decode_mask(self.rle())

    @classmethod
    def from_binary(
        cls,
        binary: np.ndarray,
        instance_id: int | None = None,
        label: str | None = None,
        score: float = 1.0,
    ) -> Mask:
        from .rle import encode_mask

        rle = encode_mask(binary)
        return cls(
            size=rle["size"],
            counts=rle["counts"],
            instance_id=instance_id,
            label=label,
            score=score,
        )


class CameraParams(Ego2DexModel):
    """Intrinsics + distortion + extrinsics with EXPLICIT conventions.

    Extrinsics default: COLMAP **world->cam**, quaternion Hamilton scalar-first.
    """

    model: CameraModelType = CameraModelType.PINHOLE
    width: int | None = None
    height: int | None = None
    intrinsics: dict[str, float] = Field(
        default_factory=dict,
        description="pinhole: {fx,fy,cx,cy}; fisheye624: {f,cx,cy}",
    )
    distortion: list[float] = Field(
        default_factory=list,
        description="pinhole: [k1,k2,p1,p2,k3,...]; KB: [k1..k4]; fisheye624: 12 params",
    )
    extrinsics: list[list[float]] | None = Field(default=None, description="4x4 SE(3)")
    extrinsics_direction: str = Field(
        default="world_to_cam", description="'world_to_cam' (COLMAP) or 'cam_to_world'"
    )
    quaternion_convention: str = "hamilton_wxyz"  # scalar-first

    @field_validator("extrinsics")
    @classmethod
    def _ext44(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return v
        if len(v) != 4 or any(len(r) != 4 for r in v):
            raise ValueError("extrinsics must be 4x4")
        return v

    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix (single-focal models duplicate f)."""
        i = self.intrinsics
        fx = i.get("fx", i.get("f", 0.0))
        fy = i.get("fy", i.get("f", 0.0))
        cx = i.get("cx", 0.0)
        cy = i.get("cy", 0.0)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class Gaze(Ego2DexModel):
    """Eye gaze: 2D point + optional 3D ray (Aria general_eye_gaze / Ego4D subset)."""

    point_2d: list[float] | None = None
    origin_3d: list[float] | None = None
    direction_3d: list[float] | None = None
    depth_m: float | None = None


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #
class HandObjectInteraction(Ego2DexModel):
    """Per-hand interaction state (100DOH contact + EgoHOS contact mask)."""

    hand_side: HandSide = HandSide.UNKNOWN
    hand_bbox: list[float] | None = Field(default=None, description="[x,y,w,h]")
    contact_state: ContactState = ContactState.NO_CONTACT
    contact_mask: Mask | None = None
    held_object_id: int | None = None
    held_object_bbox: list[float] | None = None
    grasp_label: str | None = None
    score: float = 1.0


class ActiveObject(Ego2DexModel):
    """Active / next-active object + state-change keyframes (Ego4D FHO style)."""

    object_id: int | None = None
    label: str | None = None
    bbox: list[float] | None = None
    is_next_active: bool = False
    time_to_contact_s: float | None = None
    state_change: str | None = Field(
        default=None, description="one of pre/contact/pnr/post (Ego4D FHO)"
    )


# --------------------------------------------------------------------------- #
# Semantic / temporal
# --------------------------------------------------------------------------- #
class Tags(Ego2DexModel):
    """Open-set image tags (RAM++); double as auto-prompts for Grounding DINO."""

    tags: list[str] = Field(default_factory=list)
    scores: list[float] | None = None
    source: str | None = None


class Caption(Ego2DexModel):
    """VLM outputs: a frame caption, an action phrase, region captions, points."""

    frame_caption: str | None = None
    action: str | None = None
    region_captions: list[dict[str, Any]] = Field(
        default_factory=list, description="[{bbox|label, text}]"
    )
    points: list[dict[str, Any]] = Field(
        default_factory=list, description="[{x, y, label}] e.g. Molmo contact/affordance points"
    )
    source: str | None = None


class ActionSegment(Ego2DexModel):
    """A temporal action span. Stores BOTH seconds and frame indices + fps."""

    start_sec: float
    end_sec: float
    start_frame: int | None = None
    end_frame: int | None = None
    fps: float | None = None
    verb: str | None = None
    noun: str | None = None
    text: str | None = None

    @model_validator(mode="after")
    def _fill_frames(self) -> ActionSegment:
        if self.fps and self.start_frame is None:
            object.__setattr__(self, "start_frame", int(round(self.start_sec * self.fps)))
        if self.fps and self.end_frame is None:
            object.__setattr__(self, "end_frame", int(round(self.end_sec * self.fps)))
        return self


class Narration(Ego2DexModel):
    """Free-form narration with a timestamp (Ego4D #C/#O style)."""

    timestamp_sec: float
    text: str
    annotator: str | None = None


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
class FrameAnnotation(Ego2DexModel):
    frame_id: int
    timestamp: float | None = Field(default=None, description="seconds")
    camera: CameraParams | None = None
    hands: list[HandPose] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    masks: list[Mask] = Field(default_factory=list)
    interactions: list[HandObjectInteraction] = Field(default_factory=list)
    active_objects: list[ActiveObject] = Field(default_factory=list)
    tags: Tags | None = None
    caption: Caption | None = None
    gaze: Gaze | None = None


class VideoMeta(Ego2DexModel):
    path: str
    fps: float
    width: int
    height: int
    num_frames: int
    source: str = Field(default="gopro", description="gopro | aria | generic")
    codec: str | None = None
    duration_sec: float | None = None
    sampled_fps: float | None = Field(
        default=None, description="effective fps after keyframe sampling"
    )


class Track(Ego2DexModel):
    """instance_id -> label table entry for video object tracks."""

    instance_id: int
    label: str
    score: float = 1.0
    first_frame: int | None = None
    last_frame: int | None = None


class RetargetingResult(Ego2DexModel):
    """A robot joint trajectory retargeted from the recovered hand motion."""

    robot: str = Field(..., description="e.g. 'orca', 'allegro', 'shadow'")
    hand_side: HandSide = HandSide.RIGHT
    optimizer: str = Field(default="position", description="position|vector|dexpilot")
    joint_names: list[str] = Field(default_factory=list)
    joint_trajectory: list[list[float]] = Field(
        default_factory=list, description="num_frames x num_dof joint angles"
    )
    frame_ids: list[int] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    urdf_path: str | None = None

    def trajectory_array(self) -> np.ndarray:
        return np.asarray(self.joint_trajectory, dtype=np.float64)


class ClipAnnotation(Ego2DexModel):
    """The top-level container: everything ego2dex produces for one video."""

    schema_version: str = SCHEMA_VERSION
    video_meta: VideoMeta
    frames: list[FrameAnnotation] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    action_segments: list[ActionSegment] = Field(default_factory=list)
    narrations: list[Narration] = Field(default_factory=list)
    retargeting: list[RetargetingResult] = Field(default_factory=list)

    # ---- convenience -----------------------------------------------------
    def frame(self, frame_id: int) -> FrameAnnotation | None:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f
        return None

    def save(self, path: str | Path, indent: int | None = 2) -> Path:
        """Write the whole clip to a single JSON file (round-trippable)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=indent))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ClipAnnotation:
        return cls.model_validate_json(Path(path).read_text())


def dumps_clip(clip: ClipAnnotation, indent: int | None = 2) -> str:
    return clip.model_dump_json(indent=indent)


def loads_clip(text: str) -> ClipAnnotation:
    return ClipAnnotation.model_validate_json(text)


def dumps_dict(model: Ego2DexModel) -> dict[str, Any]:
    """Plain-dict view (e.g. for embedding in other JSON)."""
    return json.loads(model.model_dump_json())
