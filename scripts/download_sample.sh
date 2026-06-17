#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_sample.sh — get a sample egocentric clip to try ego2dex on.
#
# By default this just (re)generates the tiny bundled SYNTHETIC clip so the
# smoke pipeline runs offline. Real egocentric datasets are large and gated;
# pointers are printed below. Downloaded media goes under ./data (gitignored).
# ---------------------------------------------------------------------------
set -euo pipefail

log() { printf '\033[1;36m[download-sample]\033[0m %s\n' "$*"; }

log "regenerating the bundled synthetic clip -> assets/synthetic/"
python "$(dirname "$0")/make_synthetic.py"

cat <<'EOF'

Real egocentric sources (large / often gated — fetch per their terms):
  • Ego4D        : pip install ego4d ; ego4d --output_directory data/ego4d ...
  • Ego-Exo4D    : pip install ego-exo4d ; egoexo -o data/egoexo ...
  • EPIC-KITCHENS: https://epic-kitchens.github.io/
  • H2O / DexYCB / HOI4D / ARCTIC / HOT3D: see docs/datasets.md for links
  • Project Aria : https://www.projectaria.com/datasets/ (VRS + MPS)
  • Your own GoPro: just point ego2dex at the .mp4

Then run, e.g.:
  ego2dex run -c configs/pipeline/default.yaml -i data/my_clip.mp4 -o outputs/run
  ego2dex run -c configs/pipeline/smoke.yaml   -i assets/synthetic -o outputs/smoke   # offline
EOF
