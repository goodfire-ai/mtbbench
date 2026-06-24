#!/usr/bin/env python3
"""Preprocess the raw cBioPortal ``msk_chord_2024`` study files into the form the
mtbbench loader (``neurips25/utils/patient.py``) expects under ``msk.metadata_path``.

The loader uses ``pd.read_csv(path)`` with the DEFAULT comma separator and reads a
*melted* ``data_cna.txt`` (a long ``SAMPLE_ID`` + ``Hugo_Symbol`` table, not the raw
gene x sample matrix). The raw cBioPortal files are tab-separated, and the clinical
files carry leading ``#`` metadata rows before the real header. This script:

  * clinical_patient / clinical_sample: drop the leading ``#`` rows, re-emit as comma-CSV
  * mutations / sv / all timeline tables: re-emit tab -> comma-CSV (pandas quotes fields
    containing commas, so the default ``read_csv`` round-trips exactly)
  * data_cna.txt: melt the gene x sample matrix to long (SAMPLE_ID, Hugo_Symbol, CNA),
    keeping only altered calls (CNA != 0)

With ``--validate`` it then imports the repo's ``Patient`` loader and runs it against a
few patient IDs to prove the staged files load. Run ``--validate`` from the repo root
(``base.yaml`` is loaded via a relative path) or pass ``--fork-root``.

Usage:
    python scripts/preprocess_msk_chord.py \
        --in-dir  /path/to/msk_chord_2024 \
        --out-dir /path/to/data/msk_chord_processed
"""
import argparse
import glob
import os
import sys

import pandas as pd


def count_leading_comment_lines(path):
    """Count the leading ``#``-prefixed metadata rows in a cBioPortal clinical file."""
    n = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                n += 1
            else:
                break
    return n


def passthrough(src, dst, has_comment_header):
    """Re-emit a tab-separated table as comma-CSV, dropping any ``#`` metadata header."""
    skip = count_leading_comment_lines(src) if has_comment_header else 0
    df = pd.read_csv(src, sep="\t", skiprows=skip, dtype=str,
                     keep_default_na=False, na_filter=False, low_memory=False)
    df.to_csv(dst, index=False)
    return df.shape


def melt_cna(src, dst, nrows=None):
    """Melt the gene x sample CNA matrix to long form, keeping only altered (CNA != 0) calls."""
    # gene x sample matrix; first column is Hugo_Symbol, rest are sample IDs
    df = pd.read_csv(src, sep="\t", low_memory=False, nrows=nrows)
    id_col = df.columns[0]  # Hugo_Symbol
    long = df.melt(id_vars=[id_col], var_name="SAMPLE_ID", value_name="CNA")
    long = long.rename(columns={id_col: "Hugo_Symbol"})
    long["CNA"] = pd.to_numeric(long["CNA"], errors="coerce")
    long = long[(long["CNA"].notna()) & (long["CNA"] != 0)]
    long = long[["SAMPLE_ID", "Hugo_Symbol", "CNA"]]
    long.to_csv(dst, index=False)
    return long.shape


def preprocess(in_dir, out_dir, smoke=False):
    """Convert every raw study file in ``in_dir`` into the loader-ready form in ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    clinical = ["data_clinical_patient.txt", "data_clinical_sample.txt"]
    plain = ["data_mutations.txt", "data_sv.txt"]
    plain += [os.path.basename(p) for p in glob.glob(os.path.join(in_dir, "data_timeline_*.txt"))]

    for name in clinical:
        shape = passthrough(os.path.join(in_dir, name), os.path.join(out_dir, name), has_comment_header=True)
        summary[name] = shape
        print(f"clinical  {name:42} rows={shape[0]:>8,} cols={shape[1]}")

    for name in sorted(set(plain)):
        src = os.path.join(in_dir, name)
        if not os.path.exists(src):
            print(f"MISSING   {name}")
            continue
        shape = passthrough(src, os.path.join(out_dir, name), has_comment_header=False)
        summary[name] = shape
        print(f"table     {name:42} rows={shape[0]:>8,} cols={shape[1]}")

    cna_shape = melt_cna(os.path.join(in_dir, "data_cna.txt"), os.path.join(out_dir, "data_cna.txt"),
                         nrows=50 if smoke else None)
    summary["data_cna.txt"] = cna_shape
    tag = " (SMOKE: 50 genes only)" if smoke else " (altered calls only)"
    print(f"melted    data_cna.txt{'':31} rows={cna_shape[0]:>8,} cols={cna_shape[1]}{tag}")

    # carry meta_study.txt for provenance, if present
    msrc = os.path.join(in_dir, "meta_study.txt")
    if os.path.exists(msrc):
        with open(msrc) as fr, open(os.path.join(out_dir, "meta_study.txt"), "w") as fw:
            fw.write(fr.read())
    return summary


def validate(fork_root, patient_ids):
    """Import the repo loader and run ``Patient()`` on a few patients to prove the staged files load."""
    sys.path.insert(0, fork_root)
    os.chdir(fork_root)  # base.yaml is loaded via a relative path
    from neurips25.utils.patient import Patient

    print("\n=== validation: repo Patient() loader ===")
    ok = 0
    for pid in patient_ids:
        try:
            p = Patient(pid)
            n_samples = len(p.samples)
            s0 = p.samples[0] if p.samples else None
            n_cna = len(s0.cna) if s0 is not None else 0
            n_mut = len(s0.mutation) if s0 is not None else 0
            n_sv = len(s0.sv) if s0 is not None else 0
            events = p.get_sorted_events(0, 10000)
            print(f"OK  {pid}: samples={n_samples} cna={n_cna} mut={n_mut} sv={n_sv} events={len(events)}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {pid}: {type(e).__name__}: {e}")
    print(f"\nvalidated {ok}/{len(patient_ids)} patients")
    return ok == len(patient_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="Raw cBioPortal msk_chord_2024 study directory.")
    ap.add_argument("--out-dir", required=True, help="Output dir for the loader-ready files (msk.metadata_path).")
    ap.add_argument("--fork-root", default=None, help="Repo root to import the Patient loader from (with --validate).")
    ap.add_argument("--validate", action="store_true", help="Import the repo loader and run Patient() on --patients.")
    ap.add_argument("--smoke", action="store_true", help="Truncate the CNA melt for a fast entrypoint check.")
    ap.add_argument("--patients", nargs="*", default=[
        "P-0004727", "P-0005708", "P-0006191", "P-0006687", "P-0009786"],
        help="Patient IDs to load under --validate (defaults to a handful of curated cases).")
    args = ap.parse_args()

    preprocess(args.in_dir, args.out_dir, smoke=args.smoke)
    if args.validate:
        assert args.fork_root, "--fork-root required with --validate"
        all_ok = validate(args.fork_root, args.patients)
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
