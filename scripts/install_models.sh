#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install_models.sh — source-install heavy models + fetch weights (GPU env).
#
# ego2dex's core is light and ships NO weights. This script documents and (with
# args) automates the per-model installs that are NOT on PyPI. Run only the
# components you need, in a CUDA-enabled environment with plenty of disk.
#
#   bash scripts/install_models.sh --help
#   bash scripts/install_models.sh hamer sam2 grounding_dino
#   bash scripts/install_models.sh all
#
# Weights / datasets are written under ./_DATA (gitignored). NEVER commit them.
# MANO is research-only + gated and is NOT downloaded here — see the note below.
# ---------------------------------------------------------------------------
set -euo pipefail

DATA_DIR="${EGO2DEX_DATA_DIR:-$(pwd)/_DATA}"
mkdir -p "$DATA_DIR"
PIP="${PIP:-pip}"

log()  { printf '\033[1;36m[install-models]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install-models]\033[0m %s\n' "$*"; }

mano_note() {
  warn "MANO is research-only and GATED. ego2dex never vendors it."
  warn "  1) register at https://mano.is.tue.mpg.de"
  warn "  2) download MANO_RIGHT.pkl / MANO_LEFT.pkl"
  warn "  3) set pipeline config: mano.model_dir=/path/to/mano"
}

install_mediapipe() {
  log "MediaPipe Hands (Apache-2.0, CPU)"
  $PIP install "mediapipe>=0.10"
  log "fetching hand_landmarker.task (~7 MB)"
  curl -L -o "$DATA_DIR/hand_landmarker.task" \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  log "set params.model_path=$DATA_DIR/hand_landmarker.task (or EGO2DEX_MEDIAPIPE_TASK)"
}

install_hamer() {
  log "HaMeR (MIT code + MANO). Cloning + fetching ViT-H/ViTPose weights."
  [ -d hamer ] || git clone --recursive https://github.com/geopavlakos/hamer.git
  ( cd hamer && $PIP install -e ".[all]" && bash fetch_demo_data.sh )
  mano_note
}

install_wilor() {
  log "WiLoR (CC-BY-NC-ND models + AGPL detector + MANO) via WiLoR-mini"
  $PIP install "git+https://github.com/warmshao/WiLoR-mini"
  mano_note
}

install_dynhamr() {
  log "Dyn-HaMR (MIT + MANO). Clone + follow its env setup (SLAM + HaMeR/WiLoR)."
  [ -d Dyn-HaMR ] || git clone --recursive https://github.com/ZhengdiYu/Dyn-HaMR.git
  mano_note
}

install_grounding_dino() {
  log "Grounding DINO (Apache-2.0) via transformers (weights pulled on first use)"
  $PIP install "transformers>=4.40" "supervision>=0.18"
  # Original repo alternative:
  # git clone https://github.com/IDEA-Research/GroundingDINO && (cd GroundingDINO && pip install -e .)
}

install_sam2() {
  log "SAM 2 (Apache-2.0)"
  $PIP install "git+https://github.com/facebookresearch/sam2.git"
  log "download checkpoints with sam2's download_ckpts.sh (sam2.1_hiera_large.pt, ...)"
}

install_grounded_sam2() { install_grounding_dino; install_sam2; }

install_yolo_world() { log "YOLO-World (GPLv3/commercial)"; $PIP install ultralytics; }
install_detic()      { warn "Detic needs detectron2; see github.com/facebookresearch/Detic"; }
install_deva()       { warn "DEVA (GPLv3): github.com/hkchengrex/Tracking-Anything-with-DEVA"; }
install_samurai()    { warn "SAMURAI (Apache-2.0): github.com/yangchris11/samurai"; }

install_hoi() {
  warn "100DOH hand_object_detector (research): github.com/ddshan/hand_object_detector (CUDA RoI build)"
  warn "EgoHOS (MIT): github.com/owenzlz/EgoHOS (mmsegmentation)"
}

install_caption() {
  log "VLMs (Qwen2.5-VL/Florence-2/Molmo, Apache/MIT) + RAM++"
  $PIP install "transformers>=4.49" accelerate qwen-vl-utils einops
  $PIP install recognize-anything   # RAM++ (`ram` package)
}

install_pose() {
  warn "COLMAP is a system binary (apt/brew). DROID-SLAM: github.com/princeton-vl/DROID-SLAM"
  $PIP install scipy
}

install_aria()     { log "Project Aria tools"; $PIP install "projectaria-tools>=1.5"; }
install_retarget() { log "dex-retargeting (MIT)"; $PIP install "dex_retargeting>=0.4.0"; warn "ORCA needs YOUR urdf"; }
install_export()   { log "HDF5/parquet writers"; $PIP install "h5py>=3.8" "pyarrow>=14.0"; }

ALL=(mediapipe hamer wilor dynhamr grounding_dino sam2 yolo_world detic deva samurai
     hoi caption pose aria retarget export)

usage() {
  cat <<EOF
Usage: bash scripts/install_models.sh [component ...] | all | --help
Components: ${ALL[*]}
Weights go to: $DATA_DIR  (gitignored). MANO is NOT downloaded (gated).
EOF
}

main() {
  [ $# -eq 0 ] && { usage; exit 0; }
  [ "$1" = "--help" ] && { usage; exit 0; }
  local targets=("$@")
  [ "$1" = "all" ] && targets=("${ALL[@]}")
  for t in "${targets[@]}"; do
    if declare -f "install_$t" >/dev/null; then "install_$t"; else warn "unknown component: $t"; fi
  done
  log "done. See docs/install.md and docs/licenses.md."
}

main "$@"
