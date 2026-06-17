"""COCO import/export for the ego2dex schema.

Follows COCO conventions exactly: bbox ``[x, y, w, h]`` (top-left, 0-indexed),
keypoints flattened to ``[x, y, v]`` triplets (v in {0,1,2}), masks as RLE in
``segmentation``, and a **1-based** skeleton. A "hand" keypoint category carries
the standard-21 names + skeleton from :mod:`ego2dex.topology`.

Round-trippable for keypoints + boxes (see tests/test_coco.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..topology import COCO_HAND_SKELETON, NUM_HAND_KEYPOINTS, STANDARD21_NAMES
from .core import ClipAnnotation, Detection, HandPose, HandSide

HAND_CATEGORY_ID = 1
KP_CONF_VISIBLE = 0.5  # conf >= this -> v=2 (visible), else v=1 (labeled, occluded)


# --------------------------------------------------------------------------- #
# Keypoint <-> COCO flat triplets
# --------------------------------------------------------------------------- #
def handpose_to_coco_keypoints(hand: HandPose) -> list[float]:
    """21 ``[x,y,conf]`` -> flat ``[x,y,v]*21`` COCO list (len 63)."""
    flat: list[float] = []
    for x, y, c in hand.keypoints_2d:
        if c <= 0:
            v = 0
        elif c >= KP_CONF_VISIBLE:
            v = 2
        else:
            v = 1
        flat.extend([float(x), float(y), float(v)])
    return flat


def coco_keypoints_to_handpose(
    flat: list[float], side: HandSide | str = HandSide.UNKNOWN, **kwargs: Any
) -> HandPose:
    """Inverse of :func:`handpose_to_coco_keypoints`. v maps back to a coarse conf."""
    if len(flat) != 3 * NUM_HAND_KEYPOINTS:
        raise ValueError(f"expected {3 * NUM_HAND_KEYPOINTS} values, got {len(flat)}")
    kp2d = []
    for i in range(NUM_HAND_KEYPOINTS):
        x, y, v = flat[3 * i : 3 * i + 3]
        conf = {0: 0.0, 1: 0.5, 2: 1.0}.get(int(v), float(v))
        kp2d.append([float(x), float(y), conf])
    side = HandSide(side) if not isinstance(side, HandSide) else side
    return HandPose(side=side, keypoints_2d=kp2d, **kwargs)


def _keypoint_bbox(hand: HandPose) -> list[float]:
    xs = [p[0] for p in hand.keypoints_2d if p[2] > 0]
    ys = [p[1] for p in hand.keypoints_2d if p[2] > 0]
    if not xs:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


# --------------------------------------------------------------------------- #
# Detection <-> COCO annotation
# --------------------------------------------------------------------------- #
def detection_to_coco_ann(det: Detection, ann_id: int, image_id: int, category_id: int) -> dict:
    x, y, w, h = det.bbox
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(w), float(h)],
        "area": float(w * h),
        "iscrowd": 0,
        "score": float(det.score),
    }


def coco_ann_to_detection(ann: dict, category_name: str) -> Detection:
    return Detection(
        label=category_name,
        bbox=[float(v) for v in ann["bbox"]],
        score=float(ann.get("score", 1.0)),
        instance_id=ann.get("id"),
    )


# --------------------------------------------------------------------------- #
# RLE helper for segmentation field
# --------------------------------------------------------------------------- #
def _coco_segmentation(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    if isinstance(counts, (list, tuple)):
        try:  # compress uncompressed RLE if pycocotools is present
            import pycocotools.mask as mask_utils

            comp = mask_utils.frPyObjects(rle, int(rle["size"][0]), int(rle["size"][1]))
            c = comp["counts"]
            if isinstance(c, bytes):
                c = c.decode("ascii")
            return {"size": list(rle["size"]), "counts": c}
        except Exception:
            return {"size": list(rle["size"]), "counts": list(counts)}
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


# --------------------------------------------------------------------------- #
# Whole-clip export
# --------------------------------------------------------------------------- #
def hand_keypoint_category() -> dict:
    return {
        "id": HAND_CATEGORY_ID,
        "name": "hand",
        "supercategory": "person",
        "keypoints": list(STANDARD21_NAMES),
        "skeleton": [list(e) for e in COCO_HAND_SKELETON],
    }


def to_coco(clip: ClipAnnotation, file_name_fmt: str = "{:06d}.jpg") -> dict:
    """Convert a :class:`ClipAnnotation` to a COCO-format dict.

    ``categories`` = the hand keypoint category (id 1) + one object category per
    distinct detection/mask label (ids 2..). Hands become keypoint annotations;
    detections become bbox annotations; masks become RLE ``segmentation``.
    """
    images: list[dict] = []
    annotations: list[dict] = []

    # Build object category table.
    labels: set[str] = set()
    for f in clip.frames:
        labels.update(d.label for d in f.detections)
        labels.update(m.label for m in f.masks if m.label)
    label_to_id: dict[str, int] = {}
    categories = [hand_keypoint_category()]
    for i, label in enumerate(sorted(labels), start=HAND_CATEGORY_ID + 1):
        label_to_id[label] = i
        categories.append({"id": i, "name": label, "supercategory": "object"})

    ann_id = 1
    for f in clip.frames:
        width = clip.video_meta.width
        height = clip.video_meta.height
        if f.camera and f.camera.width:
            width = f.camera.width
            height = f.camera.height or height
        images.append(
            {
                "id": f.frame_id,
                "file_name": file_name_fmt.format(f.frame_id),
                "width": int(width),
                "height": int(height),
                "timestamp": f.timestamp,
            }
        )
        for hand in f.hands:
            kps = handpose_to_coco_keypoints(hand)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": f.frame_id,
                    "category_id": HAND_CATEGORY_ID,
                    "keypoints": kps,
                    "num_keypoints": sum(
                        1 for i in range(NUM_HAND_KEYPOINTS) if kps[3 * i + 2] > 0
                    ),
                    "bbox": _keypoint_bbox(hand),
                    "area": float(_keypoint_bbox(hand)[2] * _keypoint_bbox(hand)[3]),
                    "iscrowd": 0,
                    "score": float(hand.score),
                    "side": hand.side if isinstance(hand.side, str) else hand.side.value,
                }
            )
            ann_id += 1
        for det in f.detections:
            annotations.append(
                detection_to_coco_ann(det, ann_id, f.frame_id, label_to_id[det.label])
            )
            ann_id += 1
        for mask in f.masks:
            cat_id = (
                label_to_id.get(mask.label, HAND_CATEGORY_ID) if mask.label else HAND_CATEGORY_ID
            )
            seg = _coco_segmentation(mask.rle())
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": f.frame_id,
                    "category_id": cat_id,
                    "segmentation": seg,
                    "bbox": [],
                    "area": 0.0,
                    "iscrowd": 0,
                    "instance_id": mask.instance_id,
                    "score": float(mask.score),
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "ego2dex COCO export",
            "version": clip.schema_version,
            "source": clip.video_meta.source,
        },
        "licenses": [{"id": 1, "name": "see docs/licenses.md"}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def save_coco(clip: ClipAnnotation, path: str | Path, indent: int | None = 2) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_coco(clip), indent=indent))
    return path
