# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology

**MTBBench** is a benchmark designed to evaluate the reasoning capabilities of multimodal large language models (LLMs) in complex clinical decision-making scenarios. It focuses on two core challenges in oncology: multimodal integration (e.g., pathology, genomics, radiology) and longitudinal reasoning across patient timelines. The benchmark includes agentic tasks requiring interaction with external foundation model-based tools and datasets.

---

## 🛠 Getting Started

To install all required dependencies, simply run:

```bash
bash setup.sh
```

> **Note**: If you want to evaluate the agent on **IHC data**, you will need to additionally clone and install [TRIDENT](https://github.com/mahmoodlab/TRIDENT/tree/main) from source.

---

## ⚙️ Configuration

Before running the benchmark, configure your paths and credentials in:

```
neurips25/configs/base.yaml
```

The config file specifies paths to datasets, tool credentials, and output directories. Below, we provide guidance for acquiring the necessary external datasets.

---

## 📁 Datasets

### HANCOCK (Multimodal Tissue Microarrays)

The **HANCOCK** dataset contains SVS-format tissue micro arrays (TMAs). To prepare it:

1. Follow the original [HANCOCK GitHub repository](https://github.com/ankilab/HANCOCK_MultimodalDataset) to extract tiles and compute cell densities using QuPath.
2. Reproduce our ABMIL training by extracting tumor centers and cell density measurements for **Blocks 1 and 2**.
3. Download the dataset files from the [HANCOCK project page](https://hancock.research.fau.eu/download) to replicate the question curation used in the benchmark.

---

### MSK-CHORD (Longitudinal Genomic Profiles)

The **MSK-CHORD** dataset is available on [cBioPortal](https://www.cbioportal.org/study/summary?id=msk_chord_2024). To use it:

* Download the ZIP archive from the cBioPortal page.
* Extract it and update the dataset path in `base.yaml`.

---

### DrugBank API

To enable the **DrugBank tool** for longitudinal drug lookups:

1. [Register for a DrugBank account](https://www.drugbank.com).
2. Apply for a license to access their API.
3. Download and locally host the dataset following their documentation.
4. Update your API path and credentials in the config file.

---

## 📑 Agent Logs

We provide full logs of agent interactions for all models evaluated in the paper:

* `agent_logs_hancock/`: Multimodal evaluation logs (HANCOCK)
* `agent_logs_msk/`: Longitudinal evaluation logs (MSK-CHORD)

Each log includes all agent–LLM conversations, intermediate reasoning steps, and generated answers.

---

## ▶️ Running the Benchmark

Make sure you have:

* Installed dependencies
* Configured Hugging Face access tokens (for model download)
* Set paths in `base.yaml`

To run an evaluation with `Qwen/Qwen2.5-VL-7B-Instruct` on the HANCOCK dataset:

```bash
python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --output_dir "./agent_logs_hancock/" \
  --dataset "hancock"
```

---

## 🔁 Reproducible runtime (Goodfire fork)

This fork adds a pinned, validated runtime for the **agent benchmark** plus the data-prep
and reproduction tooling needed to run both tracks end-to-end. The original `setup.sh` /
`requirements.txt` (conda, full WSI/ABMIL stack) still describe the upstream install; the
section below is the leaner, version-pinned path that the in-tree compatibility shims target.

### What's new in this fork

| Area | Change |
|------|--------|
| Runner | `--use-tools` (tool-augmented agent) and `--max-cases N` (stop after N cases) flags |
| Compatibility | shims for vLLM 0.23 / torch 2.11+cu130 / transformers 5 (`neurips25/eval`, `neurips25/tools/conch.py`) |
| Data prep | `scripts/preprocess_msk_chord.py`, `scripts/build_cell_density_csv.py`, `scripts/validate_ihc_tool_path.py` |
| Config | `tools.conch` / `tools.uni` in `base.yaml`; `scripts/link_data.sh` to wire staged data/models |
| Reproduction | `scripts/setup_runtime.sh`, `scripts/smoke.sh`, `scripts/analyze_smoke.py`, sbatch templates |
| Models | `MODELS.md` (the five models, sizes, load mechanism, gated-access notes) |

### 1. Install the pinned runtime

```bash
bash scripts/setup_runtime.sh        # builds .venv with uv: vllm 0.23, torch 2.11+cu130, transformers 5.12, CONCH
source .venv/bin/activate
```

The pins live in `requirements-runtime.txt`. This is the **lean runtime set** — it omits the
optional offline paths (UNI/TRIDENT ABMIL training, openslide/QuPath WSI processing,
pyserini/faiss PubMed retrieval); those are not needed to run the benchmark.

### 2. Point at staged data + models

The dataset's question JSONs embed **relative** `data/...` paths, so a `data/` directory must
exist at the repo root. The repo already tracks a small `data/` (the shipped
`cell_density_measurements.csv` and the ABMIL checkpoint); the staged root is a superset, so
set `MTBBENCH_LINK_FORCE=1` to move the tracked `data/` aside to `data.bak` (nothing is
deleted) before symlinking. Wire it (and `models/`) up with no hardcoded paths:

```bash
MTBBENCH_LINK_FORCE=1 \
MTBBENCH_DATA_ROOT=/abs/path/to/staged/data \
MTBBENCH_MODELS_ROOT=/abs/path/to/models \
bash scripts/link_data.sh
```

On the CoreWeave reno cluster the staged root is `/mnt/data/artifacts/tumor_board`
(`.../data`, `.../models`), group-readable to `slurm-users`. See `MODELS.md` for the models.

### 3. Run both tracks

```bash
export DRUGBANK_USERNAME="placeholder@example.com"   # pubmed.py reads this at import time (Entrez.email); the tool is never called

# MSK-CHORD (longitudinal, no tools)
python -m neurips25.benchmarks.run_agent_benchmark \
  --dataset msk --doctor_model "$MTBBENCH_MODELS_ROOT/Qwen3-32B" \
  --output_dir ./agent_logs_msk/

# HANCOCK (multimodal, tool-augmented: CONCH + IHC density tool)
python -m neurips25.benchmarks.run_agent_benchmark \
  --dataset hancock --use-tools --doctor_model "$MTBBENCH_MODELS_ROOT/Qwen2.5-VL-32B-Instruct" \
  --output_dir ./agent_logs_hancock/
```

CONCH downloads gated `MahmoodLab/conch` weights at first use; set `HF_TOKEN` to an account
that has accepted the gate (see `MODELS.md`).

### 4. Smoke / reproduction

`scripts/smoke.sh` runs the runner at `--max-cases 2` on a track; `scripts/analyze_smoke.py`
scores the logs (logs written, no parse errors, valid answers, and — for HANCOCK — the IHC
tool + CONCH fire with zero fallbacks). SLURM templates: `scripts/smoke_{msk,hancock}.sbatch`.

```bash
bash scripts/smoke.sh hancock "$MTBBENCH_MODELS_ROOT/Qwen2.5-VL-32B-Instruct" 1 ./agent_logs_hancock_smoke
# Score the bar. Pass the run's stdout (.out) so IHC/CONCH fires are counted from the
# unmutated logger output -- the agent rewrites its conversation between questions, so the
# JSON blob alone is only a lower bound and the zero-fallback check is not airtight.
python scripts/analyze_smoke.py ./agent_logs_msk_smoke ./agent_logs_hancock_smoke smoke_summary.json \
  --hancock-stdout ./smoke_smoke-mtb-hancock_<jobid>.out
```

### Data acquisition (gated / external sources)

| Source | How to obtain | Prep |
|--------|---------------|------|
| MSK-CHORD | cBioPortal `msk_chord_2024` ZIP ([study](https://www.cbioportal.org/study/summary?id=msk_chord_2024)); CC BY-NC-ND 4.0 | `scripts/preprocess_msk_chord.py --in-dir <raw> --out-dir data/msk_chord_processed` |
| HANCOCK IHC tool CSV | the shipped `data/hancock/cell_density_measurements.csv` holds the **all-predicted (ABMIL)** values, which is the faithful tool output by design (the IHC tool *is* the ABMIL predictor) | see `scripts/build_cell_density_csv.py` to build the measured comparison CSV from a FAU QuPath export |
| CONCH / UNI | gated `MahmoodLab/CONCH`, `MahmoodLab/UNI` (accept the gate on your HF account) | staged under `models/{conch,uni}` |
| DrugBank | licensed account ([drugbank.com](https://go.drugbank.com)) — **blocked** without credentials; not needed for the two tracks above | set `DRUGBANK_USERNAME`/`DRUGBANK_PASSWORD` |
| Question JSONs + cases | generated over case data (`generate_questions.py` / `msk_question_generation.py`, needs an OpenAI key); not in upstream | place at `data/questions_{hancock,msk}_bench.json` + `data/{hancock,msk_bench}/cases` |

Licenses: MSK-CHORD (CC BY-NC-ND 4.0) and HANCOCK (CC BY-NC) are non-commercial; MSK-CHORD
also restricts derivatives. Confirm this fits your intended use before relying on the data.

### Provenance

Synthesized from Goodfire tumor-board experiments #1 (data staging + MSK preprocessing),
#2 (HANCOCK IHC tool data), #3 (runner flags + compatibility shims + smoke validation), and
#4 (model consolidation). See the PR description for details.

---
