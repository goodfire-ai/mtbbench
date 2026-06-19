"""Score smoke / reproduction agent logs into a pass/fail summary.

Usage: python scripts/analyze_smoke.py <msk_log_dir> <hancock_log_dir> [out_json]

For each track it reports: #cases with a parseable log, #questions, #with a `correct`
flag, #valid `[ANSWER:]`, and (HANCOCK) whether the IHC tool and CONCH actually fired
(non-fallback). Accuracy is NOT a pass criterion — a couple of cases is plumbing
validation only. Pass criteria: logs written, no parse errors, every question carries a
`correct` flag, valid answers present, and (HANCOCK) the IHC tool + CONCH fired.
"""
import json
import os
import re
import sys


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


def summarize(name, logs, is_hancock):
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
    crit = {
        "logs_written": agg["cases"] > 0,
        "no_parse_errors": agg["parse_errors"] == 0,
        "questions_have_correct_flags": agg["questions"] > 0 and agg["with_correct"] == agg["questions"],
        "valid_answers_present": agg["valid_answer"] > 0,
    }
    if is_hancock:
        crit["ihc_tool_fired"] = agg["ihc_fired"] > 0
        crit["ihc_no_fallback"] = agg["ihc_fallback"] == 0
        crit["conch_fired"] = agg["conch_fired"] > 0
    return {"name": name, "is_hancock": is_hancock, "per_case": cases,
            "aggregate": agg, "criteria": crit, "all_pass": all(crit.values())}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    msk_dir, han_dir = sys.argv[1], sys.argv[2]
    out_json = sys.argv[3] if len(sys.argv) > 3 else None
    result = {
        "msk": summarize("MSK (no-tools)", load_logs(msk_dir), False),
        "hancock": summarize("HANCOCK (tools)", load_logs(han_dir), True),
    }
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
        print(f"  cases={a['cases']} questions={a['questions']} valid_answers={a['valid_answer']} "
              f"ihc_fired={a['ihc_fired']} ihc_fallback={a['ihc_fallback']} conch_fired={a['conch_fired']}")
    # non-zero exit if either track fails, so CI / harness callers can gate on it
    sys.exit(0 if all(result[r]["all_pass"] for r in ("msk", "hancock")) else 1)


if __name__ == "__main__":
    main()
