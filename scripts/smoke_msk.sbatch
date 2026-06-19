#!/usr/bin/env bash
#SBATCH --job-name=smoke-mtb-msk
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=smoke_%x_%j.out
# MSK-CHORD smoke / reproduction run (no tools), 2 cases.
#
# Submit from the repo root after pointing at staged data (scripts/link_data.sh).
# Submit from the repo root (so SLURM_SUBMIT_DIR is the repo) or set MTBBENCH_REPO.
# sbatch forwards your environment (--export=ALL is the default); set before submitting:
#   MTBBENCH_REPO    repo root (default: SLURM_SUBMIT_DIR, i.e. the submit dir)   [optional]
#   MTBBENCH_VENV    path to the runtime venv (from scripts/setup_runtime.sh)    [optional]
#   MTBBENCH_MODELS_ROOT  models root (for the --doctor_model path)
#   MTBBENCH_OUT     output dir (default: ./agent_logs_msk_smoke)
set -euo pipefail

# SLURM copies the batch script to a spool dir, so $0 is not the repo. Resolve the repo
# from MTBBENCH_REPO, else the submit dir, else cwd.
REPO_ROOT="${MTBBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -n "${MTBBENCH_VENV:-}" ] && source "$MTBBENCH_VENV/bin/activate"

MODEL="${MTBBENCH_MSK_MODEL:-${MTBBENCH_MODELS_ROOT:?set MTBBENCH_MODELS_ROOT}/Qwen3-32B}"
OUT="${MTBBENCH_OUT:-$REPO_ROOT/agent_logs_msk_smoke}"

bash "$REPO_ROOT/scripts/smoke.sh" msk "$MODEL" 0 "$OUT" "${MTBBENCH_MAX_CASES:-2}"
