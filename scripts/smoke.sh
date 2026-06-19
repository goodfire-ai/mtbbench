#!/usr/bin/env bash
# Run one agent-loop smoke / reproduction run (a few cases) for a single track.
#
# Usage:
#   scripts/smoke.sh <dataset: msk|hancock> <doctor_model> <use_tools: 0|1> <output_dir> [max_cases]
#
# Assumes the runtime venv is active (scripts/setup_runtime.sh) and the repo is pointed
# at staged data/models (scripts/link_data.sh). Reads, all optional:
#   HF_TOKEN                 token for the gated MahmoodLab/conch download (HANCOCK + tools)
#   MTBBENCH_HF_TOKEN_FILE   file to read HF_TOKEN from if HF_TOKEN is unset
#   DRUGBANK_USERNAME        defaulted to a placeholder (pubmed.py reads it at import time)
set -euo pipefail

DATASET="${1:?dataset: msk|hancock}"
MODEL="${2:?doctor_model (HF id or local path)}"
USETOOLS="${3:?use_tools: 0|1}"
OUTDIR="${4:?output_dir}"
MAXCASES="${5:-2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# pubmed.py raises at IMPORT time unless DRUGBANK_USERNAME is set (only used as Entrez.email;
# the pubmed/drugbank tools are never invoked by these agents).
export DRUGBANK_USERNAME="${DRUGBANK_USERNAME:-placeholder@example.com}"

# Optional: load an HF token from a file for the gated CONCH download.
if [ -z "${HF_TOKEN:-}" ] && [ -n "${MTBBENCH_HF_TOKEN_FILE:-}" ] && [ -r "${MTBBENCH_HF_TOKEN_FILE}" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${MTBBENCH_HF_TOKEN_FILE}")"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "[smoke] HF token loaded (len ${#HF_TOKEN})"
else
  echo "[smoke] WARNING: no HF_TOKEN set; the gated CONCH download may 403 for tool runs"
fi

cd "$REPO_ROOT"
mkdir -p "$OUTDIR"

EXTRA=""
[ "$USETOOLS" = "1" ] && EXTRA="--use-tools"

echo "[smoke] dataset=$DATASET model=$MODEL use_tools=$USETOOLS outdir=$OUTDIR max_cases=$MAXCASES"
echo "[smoke] python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m neurips25.benchmarks.run_agent_benchmark \
  --dataset "$DATASET" \
  --doctor_model "$MODEL" \
  --max-cases "$MAXCASES" \
  --output_dir "$OUTDIR" \
  $EXTRA

echo "[smoke] DONE. Logs in $OUTDIR:"
ls -la "$OUTDIR"
