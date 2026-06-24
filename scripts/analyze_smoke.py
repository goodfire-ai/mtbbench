"""Score smoke / reproduction agent logs into a pass/fail summary.

Usage:
  python scripts/analyze_smoke.py <msk_log_dir> <hancock_log_dir> [out_json] \
      [--hancock-stdout PATH] [--msk-stdout PATH]

For each track it reports: #cases with a parseable log, #questions, #with a `correct`
flag, #valid `[ANSWER:]`, and (HANCOCK) whether the IHC tool and CONCH actually fired
(non-fallback). Accuracy is NOT a pass criterion — a couple of cases is plumbing
validation only. Pass criteria: logs written, no parse errors, every question carries a
`correct` flag, valid answers present, and (HANCOCK) the IHC tool + CONCH fired with zero
fallbacks.

IMPORTANT — where the tool-fire counts come from. The agent mutates its in-memory
conversation between questions (`_dettach_files` strips `[IHCTool: ...]` segments from
earlier questions before the log is written), so counting tool fires/fallbacks in the
stored JSON conversation blob only sees the LAST question of each case — it is a LOWER
BOUND and can hide an earlier-question fallback. The authoritative source is the run's
**stdout** (the SLURM `.out`), which the agent's `logger.info` writes per invocation and
which is never mutated: "Using IHCTool model for image" (every IHC call), "not found in
the IHCTool data" (every fallback), "CONCH response:" (every CONCH fire). Pass
`--hancock-stdout` to score the bar from stdout; without it, the JSON-derived counts are
reported but flagged as a lower bound and the no-fallback criterion is not airtight.
"""
import json
import os
import re
import sys

# Unmutated per-invocation log markers emitted to stdout by the agent's logger.
_IHC_CALL = "Using IHCTool model for image"
_IHC_FALLBACK = "not found in the IHCTool data"
_CONCH_FIRE = "CONCH response:"


def tool_counts_from_stdout(path):
    """Authoritative IHC/CONCH counts from a run's stdout (.out); None if unreadable."""
    if not path or not os.path.isfile(path):
        return None
    text = open(path, errors="replace").read()
    return {
        "ihc_fired": text.count(_IHC_CALL),
        "ihc_fallback": text.count(_IHC_FALLBACK),
        "conch_fired": text.count(_CONCH_FIRE),
    }


def load_logs(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        case_id = fn.split("_")[0]
        try:
            with open(os.path.join(d, fn)) as f:
                out[case_id] = json.load(f)
        except Exception as e:  # noqa: BLE001
            out[case_id] = {"_parse_error": str(e)}
    return out


def analyze_case(chat_history):
    """chat_history: list of per-question dicts + a trailing {'conversation': [...]}."""
    info = {"questions": 0, "with_correct": 0, "valid_answer": 0,
            "ihc_fired": 0, "ihc_fallback": 0, "conch_fired": 0, "n_correct": 0}
    if isinstance(chat_history, dict) and "_parse_error" in chat_history:
        info["parse_error"] = chat_history["_parse_error"]
        return info
    conversation = None
    for entry in chat_history:
        if "conversation" in entry:
            conversation = entry["conversation"]
            continue
        if "question" in entry:
            info["questions"] += 1
            if "correct" in entry:
                info["with_correct"] += 1
                if entry["correct"]:
                    info["n_correct"] += 1
            resp = str(entry.get("response", ""))
            # The stored `response` is the extracted answer letter for valid answers.
            if re.match(r"^\s*[a-fA-F]\s*[)\] ]?", resp):
                info["valid_answer"] += 1
    text = json.dumps(conversation) if conversation else ""
    info["ihc_fired"] = text.count("positively stained")
    info["ihc_fallback"] = text.count("cannot be analyzed by IHCTool")
    info["conch_fired"] = text.count("The image resembles")
    return info


def summarize(name, logs, is_hancock, stdout_counts=None):
    cases = {}
    agg = {"cases": 0, "questions": 0, "with_correct": 0, "valid_answer": 0,
           "ihc_fired": 0, "ihc_fallback": 0, "conch_fired": 0, "n_correct": 0,
           "parse_errors": 0}
    for cid, ch in logs.items():
        ci = analyze_case(ch)
        cases[cid] = ci
        if "parse_error" in ci:
            agg["parse_errors"] += 1
            continue
        agg["cases"] += 1
        for k in ("questions", "with_correct", "valid_answer", "ihc_fired",
                  "ihc_fallback", "conch_fired", "n_correct"):
            agg[k] += ci[k]

    # Tool-fire counts: prefer the unmutated stdout when available (authoritative);
    # otherwise the JSON-derived counts are only a lower bound (last question per case).
    tool_source = "conversation_blob (lower bound)"
    ihc_fired, ihc_fallback, conch_fired = agg["ihc_fired"], agg["ihc_fallback"], agg["conch_fired"]
    if stdout_counts is not None:
        tool_source = "stdout (authoritative)"
        ihc_fired = stdout_counts["ihc_fired"]
        ihc_fallback = stdout_counts["ihc_fallback"]
        conch_fired = stdout_counts["conch_fired"]
        agg["ihc_fired_stdout"] = ihc_fired
        agg["ihc_fallback_stdout"] = ihc_fallback
        agg["conch_fired_stdout"] = conch_fired

    crit = {
        "logs_written": agg["cases"] > 0,
        "no_parse_errors": agg["parse_errors"] == 0,
        "questions_have_correct_flags": agg["questions"] > 0 and agg["with_correct"] == agg["questions"],
        "valid_answers_present": agg["valid_answer"] > 0,
    }
    if is_hancock:
        crit["ihc_tool_fired"] = ihc_fired > 0
        crit["conch_fired"] = conch_fired > 0
        # The no-fallback bar is only airtight from stdout; gate it on having that source.
        crit["ihc_no_fallback_airtight"] = (stdout_counts is not None) and ihc_fallback == 0
    return {"name": name, "is_hancock": is_hancock, "per_case": cases,
            "aggregate": agg, "criteria": crit, "tool_count_source": tool_source,
            "all_pass": all(crit.values())}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Score smoke / reproduction agent logs.")
    ap.add_argument("msk_dir", help="MSK agent-log directory")
    ap.add_argument("hancock_dir", help="HANCOCK agent-log directory")
    ap.add_argument("out_json", nargs="?", default=None, help="optional path to write the summary JSON")
    ap.add_argument("--hancock-stdout", default=None,
                    help="HANCOCK run stdout (.out) for authoritative IHC/CONCH fire counts")
    ap.add_argument("--msk-stdout", default=None, help="MSK run stdout (.out) (no tools; informational)")
    args = ap.parse_args()

    han_stdout = tool_counts_from_stdout(args.hancock_stdout)
    if args.hancock_stdout and han_stdout is None:
        print(f"WARNING: --hancock-stdout not readable: {args.hancock_stdout}", file=sys.stderr)
    result = {
        "msk": summarize("MSK (no-tools)", load_logs(args.msk_dir), False),
        "hancock": summarize("HANCOCK (tools)", load_logs(args.hancock_dir), True, stdout_counts=han_stdout),
    }
    out_json = args.out_json
    print(json.dumps(result, indent=2))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {out_json}")
    print("\n=== SUMMARY ===")
    for run in ("msk", "hancock"):
        r = result[run]
        print(f"\n{r['name']}: {'ALL PASS' if r['all_pass'] else 'CHECK'}")
        for k, v in r["criteria"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        a = r["aggregate"]
        if r["is_hancock"]:
            print(f"  tool counts from {r['tool_count_source']}: "
                  f"ihc_fired={a.get('ihc_fired_stdout', a['ihc_fired'])} "
                  f"ihc_fallback={a.get('ihc_fallback_stdout', a['ihc_fallback'])} "
                  f"conch_fired={a.get('conch_fired_stdout', a['conch_fired'])}")
        print(f"  cases={a['cases']} questions={a['questions']} valid_answers={a['valid_answer']}")
    # non-zero exit if either track fails, so CI / harness callers can gate on it
    sys.exit(0 if all(result[r]["all_pass"] for r in ("msk", "hancock")) else 1)


if __name__ == "__main__":
    main()
