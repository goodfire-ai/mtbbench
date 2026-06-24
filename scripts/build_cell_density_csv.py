#!/usr/bin/env python3
"""Assemble / audit the HANCOCK IHC tool data: ``data/hancock/cell_density_measurements.csv``.

The tool-augmented HANCOCK agent (``DoctorAgentWithTools._use_ihctool``) looks up
``file_name == f"{case_id}_{image_name}"`` in this CSV and returns the ``value`` column
(% positive cells). ``image_name`` has the form ``TMA_IHC_<view>_<marker>_<0|1>.png``.

IMPORTANT — what the SHIPPED CSV is, and why it is faithful by design
--------------------------------------------------------------------
In the benchmark, the IHC tool *is* the ABMIL predictor: UNI2 tiles -> ABMIL regresses
the % positive. So the canonical, shipped ``cell_density_measurements.csv`` holds the
**ALL-PREDICTED (ABMIL)** values, and that is the faithful reproduction of the tool's
runtime behaviour. Do NOT swap in QuPath-MEASURED values for the agent's runtime CSV:
the measured densities are ABMIL's *training labels*, and only CD3/CD8 were ever measured
(a small fraction of the queryable markers).

This script builds the **measured** comparison CSV from the FAU QuPath export and reports
its coverage against the staged cases and the shipped predicted baseline. It is an audit /
analysis tool (e.g. predicted-vs-measured fidelity), not a replacement for the runtime CSV.

Usage:
    # Build the measured CSV + coverage report
    python scripts/build_cell_density_csv.py \
        --fau-density /path/to/TMA_celldensity_measurements.csv \
        --cases-dir   /path/to/data/hancock/cases \
        --predicted-csv /path/to/data/hancock/cell_density_measurements.csv \
        --out /path/to/cell_density_measurements.measured.csv --write
"""
import argparse
import collections
import csv
import os
import re


def parse_img(img):
    """Parse a FAU ``Image`` field into (view, marker, block); (None, None, None) if it doesn't match."""
    m = re.match(r"(TumorCenter|InvasionFront)_([A-Za-z0-9]+)_block(\d+)\.svs", img)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


def name_key(name):
    """Map a FAU grid ``Name`` ("col-row") to an (col, row) int tuple for deterministic ordering."""
    try:
        c, r = name.split("-")
        return (int(c), int(r))
    except Exception:
        return (9999, 9999)


def build_measured(fau_density_csv):
    """Build {file_name -> % positive} from the FAU QuPath ``TMA_celldensity_measurements.csv``.

    Cores are assigned the 0/1 index per ``(Case ID, view, marker)`` in deterministic grid
    order (block ascending, then grid ``Name`` col-row), matching ankilab's
    ``export_centertiles`` TMA-grid traversal and the ``_<0|1>.png`` naming convention.
    """
    rows = []
    with open(fau_density_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    grp = collections.defaultdict(list)
    for r in rows:
        view, marker, block = parse_img(r["Image"])
        cid = r.get("Case ID", "")
        if view is None or not cid:
            continue
        if r.get("Missing", "").lower() == "true":
            continue  # core not detected -> no measured value
        pp = r.get("Positive %", "").strip()
        if pp == "":
            continue
        grp[(cid, view, marker)].append((block, name_key(r["Name"]), float(pp)))
    measured = {}  # file_name -> value
    for (cid, view, marker), cores in grp.items():
        cores.sort(key=lambda t: (t[0], t[1]))  # block asc, then grid pos
        for idx, (_, _, pp) in enumerate(cores):
            if idx > 1:  # the .png convention only has _0 and _1
                break
            measured[f"{cid}_TMA_IHC_{view}_{marker}_{idx}.png"] = pp
    return measured


def staged_marker_keys(cases_dir):
    """Agent lookup keys for every staged marker (non-H&E) IHC image: {case_id}_{image_name}."""
    staged = {}
    for c in sorted(os.listdir(cases_dir)):
        cdir = os.path.join(cases_dir, c)
        if not os.path.isdir(cdir):
            continue
        for im in os.listdir(cdir):
            if im.startswith("TMA_IHC_") and im.endswith(".png") and "_HE_" not in im:
                staged[f"{c}_{im}"] = im
    return staged


def read_keys(csv_path, col="file_name"):
    with open(csv_path) as f:
        return {r[col] for r in csv.DictReader(f)}


def _view_marker(key):
    m = re.search(r"_TMA_IHC_(TumorCenter|InvasionFront)_([A-Za-z0-9]+)_[01]\.png$", key)
    return (m.group(1), m.group(2)) if m else (None, None)


def coverage_report(measured, cases_dir=None, predicted_csv=None):
    """Print coverage of the measured CSV against staged cases and the predicted baseline."""
    print(f"measured file_names built: {len(measured)}")
    if cases_dir:
        staged = set(staged_marker_keys(cases_dir))
        print(f"staged marker (non-HE) agent keys: {len(staged)}")
        print(f"  covered by measured: {len(staged & set(measured))}")
        if predicted_csv:
            pred = read_keys(predicted_csv)
            print(f"  covered by predicted baseline: {len(staged & pred)}")
        tot, mez = collections.Counter(), collections.Counter()
        for k in staged:
            tot[_view_marker(k)] += 1
            if k in measured:
                mez[_view_marker(k)] += 1
        print("\ncoverage by view x marker (measured / total staged):")
        for key in sorted(tot):
            print(f"  {str(key[0]):13s} {str(key[1]):6s}: {mez[key]:3d}/{tot[key]:3d}")
    if predicted_csv:
        pred = read_keys(predicted_csv)
        extra = set(measured) - pred
        print(f"\nmeasured keys NOT in predicted baseline (sanity, expect ~0): {len(extra)}")
        for k in sorted(extra)[:10]:
            print("   ", k)


def write_csv(measured, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "value"])
        for k in sorted(measured):
            w.writerow([k, measured[k]])
    print(f"\nwrote {len(measured)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fau-density", required=True, help="FAU QuPath TMA_celldensity_measurements.csv (measured).")
    ap.add_argument("--cases-dir", default=None, help="data/hancock/cases dir (for coverage over staged keys).")
    ap.add_argument("--predicted-csv", default=None, help="Shipped all-predicted CSV (for the coverage/sanity report).")
    ap.add_argument("--out", default=None, help="Output path for the measured CSV (with --write).")
    ap.add_argument("--write", action="store_true", help="Write the measured CSV to --out (default: report only).")
    args = ap.parse_args()

    measured = build_measured(args.fau_density)
    coverage_report(measured, cases_dir=args.cases_dir, predicted_csv=args.predicted_csv)
    if args.write:
        assert args.out, "--out is required with --write"
        write_csv(measured, args.out)


if __name__ == "__main__":
    main()
