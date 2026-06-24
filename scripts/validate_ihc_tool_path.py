#!/usr/bin/env python3
"""End-to-end validation of the tool-augmented HANCOCK IHC path (CPU-only, no doctor model).

Runs the REAL ``DoctorAgentWithTools.run_case`` loop on a real HANCOCK case, driven by a
scripted mock doctor LLM that requests IHC images and then answers. This exercises the
actual ``run_case`` -> ``_attach_files`` / ``[IHCTool:]`` -> ``_use_ihctool`` -> cell-density
CSV lookup code path against the canonical (all-predicted) ``data/hancock/cell_density_measurements.csv``.

Stubbed (necessarily / orthogonally):
  - the doctor LLM (the mock drives the tool protocol deterministically — no model needed)
  - CONCH (the H&E foundation-model tool — separate from the IHC CSV deliverable; needs GPU+weights)
The IHC tool, the CSV, ``run_case``, ``_attach_files``, and the real case files are all genuine.

Usage (the data root must contain ``data/hancock/...`` and ``data/questions_hancock_bench.json``):
    python scripts/validate_ihc_tool_path.py --data-root /path/to/staged --case-id 296
"""
import argparse
import json
import os
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Stub heavy / orthogonal deps BEFORE importing the agent so no GPU / vLLM / weights are needed.
sys.modules["neurips25.eval"] = types.ModuleType("neurips25.eval")  # avoids vllm/transformers import
_conch = types.ModuleType("neurips25.tools.conch")


class _StubConch:
    def __init__(self, *a, **k):
        pass

    def image_to_text_retrieval(self, *a, **k):
        return ("stub", [[0.5]])


_conch.Conch = _StubConch
sys.modules["neurips25.tools.conch"] = _conch

from neurips25.models.agent_with_tools import DoctorAgentWithTools  # noqa: E402


class MockDoctorLLM:
    """Deterministically drives both IHC tool entrypoints, then answers."""

    def __init__(self):
        self.step = 0

    def evaluate(self, messages=None):
        self.step += 1
        if self.step == 1:
            # exercises _attach_files .png branch (auto-runs the IHC tool on the requested image)
            return "I will inspect the CD3 stain at the tumor center. [REQUEST: TMA_IHC_TumorCenter_CD3_0.png]"
        if self.step == 2:
            # exercises the explicit [IHCTool:] branch
            return "Now the CD8 invasion front. [IHCTool: TMA_IHC_InvasionFront_CD8_0.png]"
        return "[ANSWER: A) Tumor center]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="Dir containing data/hancock/... and data/questions_hancock_bench.json.")
    ap.add_argument("--case-id", default="296", help="HANCOCK case id with CD3/CD8 IHC questions.")
    args = ap.parse_args()

    os.chdir(args.data_root)  # the IHC tool + case image paths are read relative to cwd

    # capture every IHC tool result the agent produces during the real run
    ihc_calls = []
    orig = DoctorAgentWithTools._use_ihctool

    def spy(self, image_name, current_file_paths):
        r = orig(self, image_name, current_file_paths)
        ihc_calls.append((image_name.replace(" ", ""), r))
        return r

    DoctorAgentWithTools._use_ihctool = spy

    qs = json.loads(open("data/questions_hancock_bench.json").readlines()[0])
    assert args.case_id in qs, f"case {args.case_id} not in questions"
    case_data = qs[args.case_id]
    # keep only entries up to and including the first IHC (CD3) question -> one focused question
    trimmed = []
    for e in case_data:
        trimmed.append(e)
        if "question" in e and ("CD3" in e["question"] or "IHC" in e["question"]):
            break
    case_data = trimmed

    with tempfile.TemporaryDirectory() as out:
        agent = DoctorAgentWithTools(MockDoctorLLM(), MockDoctorLLM(), model_name="mock-doctor", output_dir=out)
        chat = agent.run_case(case_data=case_data, case_id=args.case_id)

    print("\n==== IHC tool calls made during run_case ====")
    real = 0
    for name, resp in ihc_calls:
        is_real = "positively stained" in resp
        real += is_real
        print(f"  [{'REAL' if is_real else 'FALLBACK'}] {name}: {resp}")
    print(f"\nchat_history questions answered: {len(chat) if chat else 0}")
    print(f"IHC tool calls: {len(ihc_calls)}; returning a real measurement: {real}")

    assert ihc_calls, "IHC tool was never invoked during run_case"
    assert real == len(ihc_calls), "some IHC calls hit the fallback (expected real measurements)"
    print("\nVALIDATION PASSED: tool-augmented HANCOCK path runs end-to-end; IHC tool returns real measurements.")


if __name__ == "__main__":
    main()
