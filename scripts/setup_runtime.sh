#!/usr/bin/env bash
# Build the validated runtime venv for the agent benchmark with uv.
#
# Mirrors the build order that produced the validated stack: install vllm first (it
# anchors torch 2.11+cu130 and transformers 5), then the rest of the pinned runtime
# requirements, then CONCH with --no-deps (so it does not drag incompatible deps).
#
# Usage:
#   bash scripts/setup_runtime.sh [venv_dir]   # default venv dir: .venv
#
# Requires `uv` (https://docs.astral.sh/uv/). Activate with: source <venv_dir>/bin/activate
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$REPO_ROOT/.venv}"
CONCH_REV="141cc09c7d4ff33d8eda562bd75169b457f71a62"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found (https://docs.astral.sh/uv/)"; exit 1; }

uv venv --python 3.12 "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# vllm first: it pins torch/transformers for the rest of the resolve.
uv pip install vllm==0.23.0
uv pip install -r "$REPO_ROOT/requirements-runtime.txt"
uv pip install --no-deps "conch @ git+https://github.com/mahmoodlab/conch@${CONCH_REV}"

echo
echo "runtime venv ready at $VENV_DIR"
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
