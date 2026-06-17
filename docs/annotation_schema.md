# Annotation schema

Typed with **pydantic v2** in `ego2dex/schema/core.py`; numpy inputs are coerced
to builtins, every model round-trips to/from JSON, and a JSON Schema is derived
for language-agnostic validation. Masks are COCO **RLE**. Three tiers, modeled as
a superset of the dataset survey ([`datasets.md`](datasets.md)).

`schema_version = 0.1.0`.

## Tier 1 — per-frame geometric

### `HandPose`
| field | type | notes |
|---|---|---|
| `side` | `left`/`right`/`unknown` | |
| `keypoints_2d` | `21 × [x, y, conf]` | pixels |
| `keypoints_3d` | `21 × [x, y, z]` (opt) | camera frame, meters |
| `mano` | `MANOParams` (opt) | |
| `vertices` | `778 × 3` (opt) | MANO mesh |
| `keypoint_convention` | enum | `standard21` (= MediaPipe = OpenPose), `mano_native` |
| `score` | float | |

### `MANOParams`
`global_orient[3]` (axis-angle) · `pose[45]` (15×3 axis-angle, **no global**) ·
`betas[10]` · `trans[3]` · `is_pca` / `num_pca_comps`. `full_pose()` →
`[global(3), pose(45)] = 48`. **The #1 interop bug is 45 vs 48** — validators
reject a 48-length `pose`. See [`../src/ego2dex/topology.py`](../src/ego2dex/topology.py).

### `Detection`, `Mask`, `CameraParams`, `Gaze`
- `Detection`: `label`, `score`, `bbox = [x, y, w, h]` (COCO, top-left, 0-indexed),
  optional `instance_id`.
- `Mask`: COCO RLE `{size:[h,w], counts}` (compressed `str` via pycocotools, or
  uncompressed `list[int]` fallback), `instance_id` (stable across frames),
  `label`, `score`. `from_binary()` / `decode()`.
- `CameraParams`: `model` (`pinhole` / `fisheye_kb` / `fisheye624` /
  `radtan_thinprism`), `intrinsics`, `distortion`, `extrinsics` (4×4 SE(3)). The
  **conventions are explicit**: `extrinsics_direction` defaults to
  `world_to_cam` (COLMAP) and `quaternion_convention` to `hamilton_wxyz`
  (scalar-first).
- `Gaze`: 2D point + optional 3D ray + depth.

## Tier 2 — interaction

- `HandObjectInteraction`: `hand_side`, `hand_bbox`, `contact_state` (100DOH
  5-class), `contact_mask` (RLE, EgoHOS), `held_object_id` / `_bbox`,
  `grasp_label`.
- `ActiveObject`: next-active-object + `time_to_contact_s` + `state_change`
  (`pre`/`contact`/`pnr`/`post`, Ego4D FHO).

## Tier 3 — semantic / temporal

- `Tags` (RAM++), `Caption` (`frame_caption`, `action`, `region_captions`,
  `points`), `ActionSegment` (keeps **both** seconds and frame indices + fps),
  `Narration`.

## Containers

```text
ClipAnnotation
├── video_meta : VideoMeta            (path, fps, w, h, num_frames, source, sampled_fps)
├── frames[]   : FrameAnnotation      (frame_id, timestamp, camera, hands[], detections[],
│                                       masks[], interactions[], active_objects[], tags, caption, gaze)
├── tracks[]   : Track                (instance_id -> label table, first/last frame)
├── action_segments[] : ActionSegment
├── narrations[]      : Narration
└── retargeting[]     : RetargetingResult   (robot, hand_side, optimizer, joint_names,
                                             joint_trajectory [T×DOF], frame_ids, urdf_path)
```

## On-disk formats

1. **Native ego2dex JSON** (`export/json`): one `frames/<id>.json` per frame +
   `clip.json` manifest + `clip_full.json` + `ego2dex.schema.json`.
2. **COCO** (`export/coco`): images/annotations/categories; a `hand` keypoint
   category carries the 21 names + 1-based skeleton; detections → bbox anns;
   masks → RLE `segmentation`.
3. **EgoDex-style HDF5** (`export/hdf5`) and **LeRobotDataset** (`export/lerobot`)
   — see [`pretraining.md`](pretraining.md).

## Validation

```python
from ego2dex.schema import validate_clip_json, write_json_schema
write_json_schema("ego2dex.schema.json")          # derive JSON Schema
validate_clip_json(open("clip_full.json").read()) # raises on invalid
```

## Keypoint topology (must-get-right constants)

`ego2dex/topology.py` pins: the standard-21 order (wrist=0; thumb→pinky,
base→tip), MANO constants (778 verts, 16 joints, 45/48 pose, 10 betas), MANO's
native finger order (index, middle, **pinky, ring**, thumb), tip vertex ids
(manopth `[745,317,444,556,673]`), and the manopth↔standard-21 remap. Use
`remap_keypoints(src, dst)` / `apply_remap(...)` — never reorder fingers by hand.
