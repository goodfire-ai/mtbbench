# Models manifest

The benchmark uses five models. None are committed to the repo (weights are large and,
for the pathology models, gated). They live under a single models root that you point the
repo at via `MTBBENCH_MODELS_ROOT` (see `scripts/link_data.sh` and the README).

On the CoreWeave reno cluster they are staged, group-readable to `slurm-users`, at:

```
/mnt/data/artifacts/tumor_board/models/
```

| Model | Subdir | Size | Repo ID | Loads via |
|-------|--------|------|---------|-----------|
| Qwen2.5-VL-32B-Instruct | `Qwen2.5-VL-32B-Instruct/` | 64 GB | `Qwen/Qwen2.5-VL-32B-Instruct` | explicit local path (`--doctor_model`) |
| Qwen3-32B | `Qwen3-32B/` | 62 GB | `Qwen/Qwen3-32B` | explicit local path (`--doctor_model`) |
| CONCH | `conch/` | 766 MB | `MahmoodLab/CONCH` | `tools.conch` in base.yaml (pathology tool) |
| UNI | `uni/` | 1.2 GB | `MahmoodLab/UNI` | `tools.uni` (optional; offline ABMIL only) |
| TRIDENT | `trident/` | 6.5 MB | code package | import path (optional; WSI processing) |

## Doctor models (Qwen)

The two doctor LLMs load **by explicit local path** — pass the directory to
`--doctor_model`. No HuggingFace env var (`HF_HOME` / `HF_HUB_CACHE`) is required:

```bash
python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model "$MTBBENCH_MODELS_ROOT/Qwen2.5-VL-32B-Instruct" \
  --dataset hancock --use-tools
```

`neurips25/utils/load_model.py` routes by **substring**: `"Qwen2.5-VL" in model_name`
→ `Qwen25VLEval` (HF transformers, 4-bit bnb), `"Qwen3" in model_name` →
`BaseTextVLLMEval` (vLLM, 4-bit bnb). The directory names preserve those substrings, so
the router selects the correct wrapper even when given a full local path. The two doctor
models can also be resolved by HF repo ID (`Qwen/Qwen2.5-VL-32B-Instruct`, `Qwen/Qwen3-32B`)
if you prefer a normal HuggingFace download.

Snapshots staged on reno (HF `refs/main`, 2026-06-19):
Qwen2.5-VL-32B-Instruct `7cfb30d71a1f4f49a57592323337a4a4727301da` (18 shards);
Qwen3-32B `9216db5781bf21249d130ec9da846c4624c16137` (17 shards).

## Pathology tool models (CONCH / UNI / TRIDENT)

- **CONCH** (`MahmoodLab/CONCH`) is the only pathology model used at benchmark **runtime**
  — the H&E cancer-type / invasion tool. base.yaml `tools.conch -> models/conch`. The
  CONCH python package (pinned in `requirements-runtime.txt`) downloads the gated weights
  at first use via its hardcoded `hf_hub:MahmoodLab/conch` path; staging `models/conch`
  is the offline copy.
- **UNI** (`MahmoodLab/UNI`, ViT-L/16) and **TRIDENT** are **not** used at benchmark
  runtime. They are consumed only by the optional offline ABMIL reproduction
  (`neurips25/utils/train_abmil.py`) and WSI feature extraction. Note: the HANCOCK
  `features_uni_v2` path actually uses **UNI2-h** (`MahmoodLab/UNI2-h`, a separate gated
  repo), not UNI v1 — UNI v1 weights will not load into the uni_v2 architecture.

## Gated access (CONCH / UNI / UNI2-h)

These are gated HuggingFace repos. A token only works if its account has accepted each
repo's gate; `model_info` succeeding is a **false positive** — test with a real file
download. On reno the authorized account is `chingfang17`. CONCH and UNI v1 gates are
accepted there; the UNI2-h gate (needed only for the optional ABMIL path) was not.
