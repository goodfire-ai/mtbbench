#!/usr/bin/env bash
#SBATCH --job-name=smoke-mtb-hancock
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=smoke_%x_%j.out
# HANCOCK smoke / reproduction run (tool-augmented: CONCH + IHC density tool), 2 cases.
#
# Submit from the repo root after pointing at staged data/models (scripts/link_data.sh).
# sbatch forwards your environment (--export=ALL is the default); set before submitting:
#   MTBBENCH_VENV          path to the runtime venv (from scripts/setup_runtime.sh)  [optional]
#   MTBBENCH_MODELS_ROOT   models root (for the --doctor_model path)
#   HF_TOKEN / MTBBENCH_HF_TOKEN_FILE   token authorized for gated MahmoodLab/conch
#   MTBBENCH_OUT           output dir (default: ./agent_logs_hancock_smoke)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -n "${MTBBENCH_VENV:-}" ] && source "$MTBBENCH_VENV/bin/activate"

MODEL="${MTBBENCH_HANCOCK_MODEL:-${MTBBENCH_MODELS_ROOT:?set MTBBENCH_MODELS_ROOT}/Qwen2.5-VL-32B-Instruct}"
OUT="${MTBBENCH_OUT:-$REPO_ROOT/agent_logs_hancock_smoke}"

bash "$REPO_ROOT/scripts/smoke.sh" hancock "$MODEL" 1 "$OUT" "${MTBBENCH_MAX_CASES:-2}"
