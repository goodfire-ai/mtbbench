#!/usr/bin/env bash
# Point the repo at staged data + models by creating `data/` and `models/` symlinks
# at the repo root. The benchmark and the dataset's question JSONs reference data via
# RELATIVE paths (e.g. `data/hancock/cases/...`), so a `data/` dir must exist at the
# repo root; this is the portable, no-hardcoded-path way to wire it up.
#
# Usage:
#   MTBBENCH_DATA_ROOT=/abs/path/to/staged/data \
#   MTBBENCH_MODELS_ROOT=/abs/path/to/models \
#   bash scripts/link_data.sh
#
# MTBBENCH_DATA_ROOT is required. MTBBENCH_MODELS_ROOT is optional (only needed for the
# CONCH/UNI tool models; doctor models can instead be passed by path via --doctor_model).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

link() {
  local target="$1" name="$2"
  if [ -z "$target" ]; then return 0; fi
  if [ ! -e "$target" ]; then
    echo "ERROR: $name target does not exist: $target" >&2
    exit 1
  fi
  local dest="$REPO_ROOT/$name"
  # Replace only an existing symlink; refuse to clobber a real directory.
  if [ -L "$dest" ]; then
    rm -f "$dest"
  elif [ -e "$dest" ]; then
    echo "ERROR: $dest exists and is not a symlink; remove/move it first." >&2
    exit 1
  fi
  ln -s "$target" "$dest"
  echo "linked $name -> $target"
}

: "${MTBBENCH_DATA_ROOT:?set MTBBENCH_DATA_ROOT to the staged data directory}"
link "$MTBBENCH_DATA_ROOT" data
link "${MTBBENCH_MODELS_ROOT:-}" models

echo "done. base.yaml relative paths now resolve from $REPO_ROOT."
