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
#
# The repo tracks a small data/ directory (the shipped cell_density_measurements.csv and the
# ABMIL checkpoint). The staged data root is a superset of it, so to point the repo at staged
# data set MTBBENCH_LINK_FORCE=1: an existing real data/ (or models/) is moved aside to
# <name>.bak before the symlink is created (nothing is deleted). Without it, a real directory
# is left untouched and the script errors.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE="${MTBBENCH_LINK_FORCE:-0}"

link() {
  local target="$1" name="$2"
  if [ -z "$target" ]; then return 0; fi
  if [ ! -e "$target" ]; then
    echo "ERROR: $name target does not exist: $target" >&2
    exit 1
  fi
  local dest="$REPO_ROOT/$name"
  if [ -L "$dest" ]; then
    rm -f "$dest"                       # replace a previous symlink in place
  elif [ -e "$dest" ]; then
    if [ "$FORCE" = "1" ]; then
      local bak="$dest.bak"
      [ -e "$bak" ] && { echo "ERROR: $bak already exists; resolve it first." >&2; exit 1; }
      mv "$dest" "$bak"                 # move the tracked dir aside (recoverable; nothing deleted)
      echo "moved existing $name -> $(basename "$bak")"
    else
      echo "ERROR: $dest exists and is not a symlink. Set MTBBENCH_LINK_FORCE=1 to move it aside." >&2
      exit 1
    fi
  fi
  ln -s "$target" "$dest"
  echo "linked $name -> $target"
}

: "${MTBBENCH_DATA_ROOT:?set MTBBENCH_DATA_ROOT to the staged data directory}"
link "$MTBBENCH_DATA_ROOT" data
link "${MTBBENCH_MODELS_ROOT:-}" models

echo "done. base.yaml relative paths now resolve from $REPO_ROOT."
