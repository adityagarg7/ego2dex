# Licensing — per-component matrix

`ego2dex`'s **own code is MIT** (© 2026 Aditya Garg). But `ego2dex` is a
*wrapper/pipeline*: it orchestrates third-party models it does **not**
redistribute, many of which are **non-permissive** (non-commercial, copyleft, or
API-only). **You are responsible for complying with each model's license for
your use case.** At runtime, enabling a non-permissive component prints a
one-time warning (`ego2dex.utils.logging.warn_license`).

## ⚠️ MANO is the load-bearing caveat

[MANO](https://mano.is.tue.mpg.de) is licensed for **non-commercial scientific
research only** and is **gated behind registration**. Every MANO-based hand
estimator (HaMeR, WiLoR, Hamba, WildHands, Dyn-HaMR, FrankMocap, …) and every
MANO-parameterized dataset annotation **inherits this restriction**. `ego2dex`
**never vendors MANO**; you must download `MANO_RIGHT.pkl` / `MANO_LEFT.pkl`
yourself after accepting the license, and point stages at them via
`mano.model_dir`.

## Legend

- 🟢 **Permissive** (MIT / Apache-2.0 / BSD): commercial use generally OK.
- 🟡 **Attribution / weak copyleft caveats** or research-friendly but check terms.
- 🔴 **Non-permissive** (CC-BY-NC / GPL / AGPL / research-only / API-only):
  **not** free for commercial use.

## Hand pose (3D joints + MANO)

| Component | Role | License | Commercial? | Source |
|---|---|---|---|---|
| **HaMeR** | primary hand model | 🟢 MIT (code) + 🔴 MANO | ❌ via MANO | <https://github.com/geopavlakos/hamer> |
| **WiLoR** | real-time multi-hand | 🔴 CC-BY-NC-ND (models) + 🔴 AGPL (detector) + 🔴 MANO | ❌ | <https://github.com/rolpotamias/WiLoR> |
| **MediaPipe Hands** | light CPU, no MANO | 🟢 Apache-2.0 | ✅ | <https://github.com/google-ai-edge/mediapipe> |
| **Hamba** | best FreiHAND | 🔴 CC-BY-NC 4.0 + 🔴 MANO | ❌ | <https://github.com/humansensinglab/Hamba> |
| **WildHands** | egocentric single-image | 🔴 CC-BY-NC + 🔴 MANO | ❌ | <https://github.com/ap229997/hands> |
| **Dyn-HaMR** | temporal world-frame | 🟢 MIT (code) + 🔴 MANO | ❌ via MANO | <https://github.com/ZhengdiYu/Dyn-HaMR> |
| ACR | two-hand | 🟡 MIT-ish | ⚠️ | <https://github.com/ZhengyiLuo/ACR> |
| IntagHand | two-hand | 🔴 GPL-3.0 | ❌ | <https://github.com/Dw1010/IntagHand> |
| HandOccNet | occlusion | 🔴 none (research) | ❌ | <https://github.com/namepllet/HandOccNet> |
| FrankMocap (hand) | egocentric mode | 🔴 CC-BY-NC + 🔴 MANO | ❌ | <https://github.com/facebookresearch/frankmocap> |
| SmoothNet | smoothing | 🟢 MIT | ✅ | <https://github.com/cure-lab/SmoothNet> |
| 1€ filter | smoothing | 🟢 (public/MIT) | ✅ | bundled (pure-python) |
| **MANO** | hand model | 🔴 **research-only, gated** | ❌ | <https://mano.is.tue.mpg.de> |

## Open-vocab detection

| Component | License | Commercial? | Source |
|---|---|---|---|
| **Grounding DINO** | 🟢 Apache-2.0 | ✅ | <https://github.com/IDEA-Research/GroundingDINO> |
| MM-Grounding-DINO | 🟢 Apache-2.0 | ✅ | mmdetection `configs/mm_grounding_dino` |
| YOLO-World | 🔴 GPLv3 / commercial | ❌ (GPL) | <https://github.com/AILab-CVC/YOLO-World> |
| OWLv2 | 🟢 Apache-2.0 | ✅ | `google/owlv2-*` |
| Detic | 🟢 Apache-2.0 | ✅ | <https://github.com/facebookresearch/Detic> |
| GDINO 1.5/1.6, DINO-X, T-Rex2 | 🔴 API / non-commercial | ❌ | DeepDataSpace API |

## Segmentation + tracking

| Component | License | Commercial? | Source |
|---|---|---|---|
| **SAM 2** | 🟢 Apache-2.0 | ✅ | <https://github.com/facebookresearch/sam2> |
| **Grounded-SAM-2** | 🟢 Apache-2.0 | ✅ | <https://github.com/IDEA-Research/Grounded-SAM-2> |
| DEVA | 🔴 GPLv3 | ❌ | <https://github.com/hkchengrex/Tracking-Anything-with-DEVA> |
| SAMURAI | 🟢 Apache-2.0 | ✅ | <https://github.com/yangchris11/samurai> |

## Hand-object interaction

| Component | License | Commercial? | Source |
|---|---|---|---|
| 100DOH `hand_object_detector` | 🔴 research-only | ❌ | <https://github.com/ddshan/hand_object_detector> |
| EgoHOS | 🟢 MIT | ✅ | <https://github.com/owenzlz/EgoHOS> |

## Tags + captions (VLM)

| Component | License | Commercial? | Source |
|---|---|---|---|
| RAM++ | 🟢 Apache-2.0 | ✅ | <https://github.com/xinyu1205/recognize-anything> |
| Qwen2.5-VL (3B/7B) | 🟢 Apache-2.0 | ✅ | <https://github.com/QwenLM/Qwen2.5-VL> |
| Florence-2 | 🟢 MIT | ✅ | <https://huggingface.co/microsoft/Florence-2-large> |
| Molmo | 🟢 Apache-2.0 | ✅ | <https://huggingface.co/allenai/Molmo-7B-D-0924> |
| InternVL3 | 🟢 MIT (code) | ✅ | <https://github.com/OpenGVLab/InternVL> |

## Camera pose / SLAM

| Component | License | Commercial? | Source |
|---|---|---|---|
| COLMAP | 🟢 BSD | ✅ | <https://colmap.github.io/> |
| DROID-SLAM | 🟡 research/BSD-ish | ⚠️ | <https://github.com/princeton-vl/DROID-SLAM> |
| VIPE | 🟡 check repo | ⚠️ | Dyn-HaMR dependency |
| Project Aria tools / MPS | 🟡 Aria data license | ⚠️ | <https://www.projectaria.com> |

## Retargeting + export

| Component | License | Commercial? | Source |
|---|---|---|---|
| **dex-retargeting** | 🟢 MIT | ✅ | <https://github.com/dexsuite/dex-retargeting> |
| dex-urdf (robot URDFs) | 🟢 MIT | ✅ | <https://github.com/dexsuite/dex-urdf> |
| LeRobot | 🟢 Apache-2.0 | ✅ | <https://github.com/huggingface/lerobot> |
| h5py / pyarrow | 🟢 BSD / Apache-2.0 | ✅ | — |

## A "fully permissive, commercial-OK" subset

If you need a commercially-usable stack, prefer: **MediaPipe** (hands, no MANO) +
**Grounding DINO** + **SAM 2 / Grounded-SAM-2** + **EgoHOS** + **RAM++ /
Qwen2.5-VL / Florence-2 / Molmo** + **COLMAP** + **dex-retargeting** + the JSON /
COCO / HDF5 / LeRobot writers. Note this path yields **no MANO parameters** (only
21 keypoints), which is sufficient for keypoint-based retargeting and most
visual-representation pretraining.
