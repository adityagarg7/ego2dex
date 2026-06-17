# SOTA survey — the `ego2dex` model stack

This is the literature backbone for `ego2dex`: for each pipeline role we name the
**primary** model, registered **fallbacks**, and the reasoning, with arXiv +
GitHub + license. Cross-verified June 2026. Licenses are summarized here and
detailed in [`licenses.md`](licenses.md).

The annotation taxonomy is deliberately a **superset** of what the major
egocentric datasets provide (see [`datasets.md`](datasets.md)) so `ego2dex`
output stays compatible with that ecosystem.

---

## A. Ingestion / IO

- **GoPro** `.mp4` (and 360 `.insv`, which must be flattened first) → OpenCV /
  ffmpeg frame extraction (fps, timestamps, resolution, keyframe sampling).
- **Project Aria** `.vrs` + **MPS** via
  [`projectaria-tools`](https://github.com/facebookresearch/projectaria_tools):
  RGB stream `214-1`, SLAM `1201-1/2`, eye `211-1`; MPS files
  `closed_loop_trajectory.csv`, `semidense_points.csv.gz`,
  `online_calibration.jsonl`, `general_eye_gaze.csv`.
- **Camera models**: OpenCV pinhole (Brown–Conrady `[k1,k2,p1,p2,k3,…]`), OpenCV
  fisheye / Kannala–Brandt (`D=[k1..k4]`), Aria **Fisheye624 /
  FisheyeRadTanThinPrism** (15 params, single focal). GoPro is wide-FOV → expose
  an undistortion path. Implemented in `ego2dex/io/camera.py`.

## B. Hand pose — 3D finger joints + MANO  *(default: HaMeR)*

| Model | Venue / arXiv | License | Why |
|---|---|---|---|
| **HaMeR** ⭐ | CVPR 2024 · [2312.05251](https://arxiv.org/abs/2312.05251) · [code](https://github.com/geopavlakos/hamer) | MIT + MANO | ViT-H per-hand-crop → MANO (48 pose + 10 shape) + 778-vtx mesh + 21 joints. Egocentric-proven (EgoExo4D Ego-Pose Hands 2nd; tested on EPIC-KITCHENS/Ego4D). ~27 FPS batch, GPU. Bundles ViTPose detection. Weights via `fetch_demo_data.sh`. |
| **WiLoR** | [2409.12259](https://arxiv.org/abs/2409.12259) · [code](https://github.com/rolpotamias/WiLoR) | CC-BY-NC-ND + AGPL + MANO | End-to-end detect+reconstruct, multi-hand, >130 FPS detector / ~156 FPS refiner. Weights on HF; community [`WiLoR-mini`](https://github.com/warmshao/WiLoR-mini). |
| **MediaPipe Hands / Hand Landmarker** | [docs](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | 21 landmarks (`hand_landmarks` normalized 2.5D + `hand_world_landmarks` metric + handedness). Real-time CPU, **no MANO**. The dependency-free default / smoke path. |
| **Hamba** | NeurIPS 2024 · [2407.09646](https://arxiv.org/abs/2407.09646) · [code](https://github.com/humansensinglab/Hamba) | CC-BY-NC 4.0 + MANO | ViT-H + graph Mamba; best FreiHAND accuracy; needs Mamba SSM kernels. |
| **WildHands** | ECCV 2024 · [2312.06583](https://arxiv.org/abs/2312.06583) · [code](https://github.com/ap229997/hands) | CC-BY-NC + MANO | Camera-intrinsics keypoint embedding; SOTA on ARCTIC ego; strong zero-shot on H2O / AssemblyHands / EPIC / EgoExo4D. |
| **Dyn-HaMR** | CVPR 2025 · [2412.12861](https://arxiv.org/abs/2412.12861) · [code](https://github.com/ZhengdiYu/Dyn-HaMR) | MIT + MANO | The reference **video** pipeline: SLAM (VIPE/DROID) + HaMeR + WiLoR + ViTPose + interacting-hand motion prior → 4D global MANO. Batch/offline. |

Two-hand / occlusion specialists (registered as options): **ACR** (CVPR 2023,
[2303.05938](https://arxiv.org/abs/2303.05938)), **IntagHand** (CVPR 2022, GPL-3.0),
**HandOccNet** (CVPR 2022, research-only), **FrankMocap hand** (CC-BY-NC, archived,
has an egocentric mode + the 100DOH detector).

**Temporal smoothing** (post-process any per-frame estimator): **SmoothNet**
(ECCV 2022, [code](https://github.com/cure-lab/SmoothNet)) for batch; **One-Euro
(1€) filter** for real-time (bundled, pure-python). Both exposed as the
`hands/smoothing` sub-stage.

## C. Open-vocabulary detection — boxes  *(default: Grounding DINO)*

| Model | Venue / arXiv | License | Why |
|---|---|---|---|
| **Grounding DINO** ⭐ | ECCV 2024 · [2303.05499](https://arxiv.org/abs/2303.05499) · [code](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 | Text prompt → boxes+phrases+scores. ~52.5 AP COCO (Swin-L). Open weights, local. |
| **MM-Grounding-DINO** | mmdetection | Apache-2.0 | Open training code, fine-tunable. |
| **YOLO-World** | [2401.17270](https://arxiv.org/abs/2401.17270) | GPLv3/commercial | Real-time (~52 FPS). |
| **OWLv2** | — | Apache-2.0 | Image-query / one-shot. |
| **Detic** | ECCV 2022 | Apache-2.0 | 21k-class, box+mask. |
| GDINO 1.5/1.6, DINO-X, T-Rex2 | — | API / non-commercial | Offered as optional, flagged. |

## D. Segmentation + video tracking — masks + temporal IDs  *(default: SAM 2)*

| Model | Venue / arXiv | License | Why |
|---|---|---|---|
| **SAM 2** ⭐ | [2408.00714](https://arxiv.org/abs/2408.00714) · [code](https://github.com/facebookresearch/sam2) | Apache-2.0 | Box/point → mask + streaming-memory video propagation, multi-object. |
| **Grounded-SAM-2** ⭐ | [code](https://github.com/IDEA-Research/Grounded-SAM-2) | Apache-2.0 | The integration to mirror: text→GDINO boxes→SAM2 masks→tracked IDs. |
| **DEVA** | ICCV 2023 · [2309.03903](https://arxiv.org/abs/2309.03903) | GPLv3 | Open-world long-clip multi-object ID consistency. |
| **SAMURAI** | [2411.11922](https://arxiv.org/abs/2411.11922) | Apache-2.0 | Motion-aware SAM2; robust single-object tracking under fast egocentric motion/occlusion. |

## E. Hand–object interaction

| Model | Venue / arXiv | License | Output |
|---|---|---|---|
| **100DOH `hand_object_detector`** | CVPR 2020 · [2006.06669](https://arxiv.org/abs/2006.06669) · [code](https://github.com/ddshan/hand_object_detector) | research-only | per hand: box, side (L/R), **contact state** {no-contact, self, other-person, portable-object, stationary-object}, held-object box. |
| **EgoHOS** | ECCV 2022 · [2208.03826](https://arxiv.org/abs/2208.03826) · [code](https://github.com/owenzlz/EgoHOS) | MIT | pixel masks of hands + interacted objects + contact boundaries (egocentric). |

## F. Tags + captions (VLM)  *(primary VLM: Qwen2.5-VL)*

| Model | Venue / arXiv | License | Role |
|---|---|---|---|
| **RAM++** | [2310.15200](https://arxiv.org/abs/2310.15200) · [code](https://github.com/xinyu1205/recognize-anything) | Apache-2.0 | open-set tagging; tags double as Grounding DINO auto-prompts. |
| **Qwen2.5-VL** ⭐ | [2502.13923](https://arxiv.org/abs/2502.13923) · [code](https://github.com/QwenLM/Qwen2.5-VL) | Apache-2.0 (3B/7B) | captions + actions (and grounded boxes/points via prompting). |
| **Florence-2** | [2311.06242](https://arxiv.org/abs/2311.06242) · [HF](https://huggingface.co/microsoft/Florence-2-large) | MIT | tiny/fast/local; boxes/region-captions/OV-det/ref-seg via task tokens. |
| **Molmo** | [2409.17146](https://arxiv.org/abs/2409.17146) | Apache-2.0 | pointing → contact/affordance points. |
| **InternVL3** | [2504.10479](https://arxiv.org/abs/2504.10479) | MIT (code) | alternative general VLM. |

Optional **API VLMs** (Claude / GPT-4o / Gemini) can do captions/QA/JSON
normalization but are **not reliable pixel localizers** — use the dedicated
detectors for boxes/masks.

## G. Camera pose / SLAM (GoPro has no MPS)

- **VIPE** (Dyn-HaMR's recommended SLAM) or **DROID-SLAM** (NeurIPS 2021,
  [code](https://github.com/princeton-vl/DROID-SLAM)); **COLMAP** for offline SfM.
- The Aria path uses **MPS** directly (no SLAM needed).

## H. Retargeting — hand → robot joints  *(library: `dex-retargeting`)*

[`dex-retargeting`](https://github.com/dexsuite/dex-retargeting) (MIT, `pip
install dex_retargeting`) — Pinocchio FK + NLopt SLSQP. Three optimizers, all
with temporal smoothing via `SeqRetargeting`:

- **PositionOptimizer** — match absolute 3D keypoints (DexMV-style; best for
  offline dataset retargeting / imitation).
- **VectorOptimizer** — match keypoint direction vectors, translation-invariant
  (AnyTeleop-style; teleop/streaming).
- **DexPilotOptimizer** — fingertip-to-fingertip + palm-to-tip vectors with
  contact snapping (precise pinching).

Papers: AnyTeleop (RSS 2023, [2307.04577](https://arxiv.org/abs/2307.04577)),
DexMV (ECCV 2022, [2108.05877](https://arxiv.org/abs/2108.05877)), DexPilot (ICRA
2020, [1910.03135](https://arxiv.org/abs/1910.03135)). Bundled hands: Allegro,
Shadow, SVH, LEAP, Ability, Inspire, Barrett, DClaw, Panda. **ORCA needs its own
URDF/config** (user-supplied; 16 active DOF) — first-class in `retarget/robots.py`.
See [`retargeting.md`](retargeting.md).

## I. Export & pretraining pathways

- **Visual-representation pretraining** (R3M / MVP / VC-1 / Voltron): frame
  sequences + rich annotations → webdataset/parquet manifest.
- **Retargeted-action / imitation** (DexMV / DexCap / EgoDex): per-frame 21
  keypoints + camera + **retargeted robot joint trajectories** + object states →
  LeRobotDataset / EgoDex-style HDF5 / COCO / JSON. See [`pretraining.md`](pretraining.md).

## J. VLAs are downstream — NOT in the labeling loop

A VLA (RT-2, **OpenVLA** [2406.09246](https://arxiv.org/abs/2406.09246), Octo,
π0/openpi, GR00T) outputs robot **actions** and is a **policy** (parallel to the
user's PPO controller in `fv-orca-hand-rl`). The `ego2dex` annotation pipeline
produces perception data that a VLA/RL policy later *consumes*. A **VLM** is the
perception-side counterpart and **is** used (tags/captions/pointing).

## Prior systems: human/egocentric video → dexterous learning

- **EgoDex** (Apple, [2505.11709](https://arxiv.org/abs/2505.11709)) — 829 h,
  68-joint SE(3) HDF5, robot-centric joint-space alignment.
- **EgoZero** ([2505.20290](https://arxiv.org/abs/2505.20290), MIT) — Aria→gripper,
  point-based.
- **ARCap** ([2410.08464](https://arxiv.org/abs/2410.08464)) — glove→dexterous + AR.
- **DexMimicGen** ([2410.24185](https://arxiv.org/abs/2410.24185)), **DexCap**,
  **Robotic Telekinesis**, **RSRD** ([2409.18121](https://arxiv.org/abs/2409.18121),
  object-part-centric).

Most of these do **not** use `dex-retargeting`; `ego2dex` deliberately
standardizes on it for ORCA so the retarget step is a one-line config swap across
robot hands.
