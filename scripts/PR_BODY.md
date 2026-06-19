## Make the fork runnable: setup, compatibility shims, data prep, and a smoke reproduction

This branch lands the runnable setup developed across Silico experiments #1–#4 onto the fork. Before this, `goodfire-ai/mtbbench` was vanilla upstream (HEAD `ac8151bc`) and every fix lived only on experiment branches — a teammate cloning the fork re-derived all of it. After this, the fork is the runnable artifact: clone, install pinned deps, point at staged data/models, run.

Changes are committed in **logical chunks** (one concern per commit) so the diff is reviewable. No hardcoded `/mnt/data/artifacts/...` staging paths — data/model roots are parameterized via env vars and a helper script.

### Commits (in order)

1. **`--use-tools` / `--max-cases` runner flags** — replaces the commented-out agent-selection block in `run_agent_benchmark.py` with a clean flag-driven selector, plus the arg-parser additions. (#3)
2. **Newer-stack compatibility shims** — env-compat fixes for vLLM 0.23 / torch 2.11+cu130 / transformers 5: `rope_scaling` removal, the sdpa attention fallback, and the transformers-5 CONCH tokenizer alias. (#3)
3. **MSK-CHORD preprocessing** — `scripts/preprocess_msk_chord.py` reshapes the raw cBioPortal export into the comma-CSV + melted-CNA format the loader's `Patient()` expects. (#1)
4. **HANCOCK IHC tool-data prep** — `scripts/build_cell_density_csv.py` assembles/validates the agent's `cell_density_measurements.csv` (code only, not the data or weights), plus `scripts/validate_ihc_tool_path.py` which asserts the tool returns real measurements end-to-end. Documents that the shipped all-predicted (ABMIL) CSV is the faithful tool output by design. (#2)
5. **Parameterized config** — `base.yaml` tool paths read a data root from an env var / relative path instead of the absolute staging path; `scripts/link_data.sh` points a checkout at staged data/models. (#1/#4)
6. **Setup docs + manifest** — README data-acquisition steps for the gated sources, the dep-pinning note, how to point at staged data and the consolidated models, `requirements-runtime.txt` (pinned working stack), `scripts/setup_runtime.sh`, and `MODELS.md`. (#4)

Tests: unit tests for the MSK preprocessing output schema (`tests/test_preprocess_msk_chord.py`) and the HANCOCK CSV builder schema + coverage (`tests/test_build_cell_density_csv.py`). Smoke/reproduction harness under `scripts/` (`smoke.sh`, `smoke_{msk,hancock}.sbatch`, `analyze_smoke.py`).

### Dependencies

All public (vllm, transformers, accelerate, conch, uni, trident, qwen_vl_utils) — **no internal Goodfire libraries**. `requirements-runtime.txt` pins the working stack (vLLM 0.23, torch 2.11+cu130, transformers 5) so the compatibility shims and the deps agree.

### Reproduction check (pre-registered bar)

From a clean checkout of this branch with pinned deps, the runner was run at `--max-cases 2` on both tracks against the already-staged data (not re-downloading gated sources), using the consolidated models:

- **MSK** (no tools): completes, writes parseable scored logs.
- **HANCOCK** (tools): completes, writes parseable scored logs, and the IHC tool + CONCH fire with **zero fallbacks** on the sampled cases — reproducing #3's smoke outcome.

Per-question accuracy is **not** pinned (data-dependent / non-deterministic / GPU). See the PR's linked results page for the scored bar.

### Provenance

Synthesized from Silico experiments #1 (MSK-CHORD staging + preprocessing), #2 (HANCOCK IHC tool data), #3 (runner flags + compatibility shims + smoke), #4 (models consolidation). Upstream `bunnelab/mtbbench` history is preserved.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
