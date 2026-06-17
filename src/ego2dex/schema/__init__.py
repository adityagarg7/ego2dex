"""Typed annotation schema + (de)serialization + COCO + JSON Schema."""

from __future__ import annotations

from .coco import (
    coco_keypoints_to_handpose,
    handpose_to_coco_keypoints,
    save_coco,
    to_coco,
)
from .core import (
    SCHEMA_VERSION,
    ActionSegment,
    ActiveObject,
    CameraModelType,
    CameraParams,
    Caption,
    ClipAnnotation,
    ContactState,
    Detection,
    Ego2DexModel,
    FrameAnnotation,
    Gaze,
    HandObjectInteraction,
    HandPose,
    HandSide,
    MANOParams,
    Mask,
    Narration,
    RetargetingResult,
    Tags,
    Track,
    VideoMeta,
    dumps_clip,
    loads_clip,
)
from .jsonschema import clip_json_schema, validate_clip_json, write_json_schema
from .rle import decode_mask, encode_mask, mask_area, mask_to_bbox

__all__ = [
    "SCHEMA_VERSION",
    # enums
    "HandSide",
    "ContactState",
    "CameraModelType",
    # geometric
    "MANOParams",
    "HandPose",
    "Detection",
    "Mask",
    "CameraParams",
    "Gaze",
    # interaction
    "HandObjectInteraction",
    "ActiveObject",
    # semantic
    "Tags",
    "Caption",
    "ActionSegment",
    "Narration",
    # containers
    "FrameAnnotation",
    "VideoMeta",
    "Track",
    "RetargetingResult",
    "ClipAnnotation",
    "Ego2DexModel",
    # (de)serialize
    "dumps_clip",
    "loads_clip",
    # coco
    "to_coco",
    "save_coco",
    "handpose_to_coco_keypoints",
    "coco_keypoints_to_handpose",
    # jsonschema
    "clip_json_schema",
    "write_json_schema",
    "validate_clip_json",
    # rle
    "encode_mask",
    "decode_mask",
    "mask_area",
    "mask_to_bbox",
]
