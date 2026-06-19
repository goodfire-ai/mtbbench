"""Schema tests for scripts/preprocess_msk_chord.py.

Builds tiny synthetic cBioPortal-style inputs and asserts the reshaped outputs match
what the repo loader (neurips25/utils/patient.py) reads: comma-CSV clinical/mutation
tables with the leading ``#`` metadata rows dropped, and a *melted* long
``data_cna.txt`` (SAMPLE_ID, Hugo_Symbol, CNA) containing only altered (CNA != 0) calls.

Runnable under pytest or standalone: ``python tests/test_preprocess_msk_chord.py``.
"""
import csv
import importlib.util
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "preprocess_msk_chord", os.path.join(REPO_ROOT, "scripts", "preprocess_msk_chord.py"))
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _make_raw_study(d):
    # clinical files: 4 leading '#' metadata rows, then a tab-separated header + rows
    _write(os.path.join(d, "data_clinical_patient.txt"),
           "#Patient Identifier\tOverall Survival\n"
           "#Patient Identifier\tOverall Survival\n"
           "#STRING\tNUMBER\n"
           "#1\t1\n"
           "PATIENT_ID\tOS_MONTHS\n"
           "P-0000001\t12.3\n"
           "P-0000002\t4.5\n")
    _write(os.path.join(d, "data_clinical_sample.txt"),
           "#Patient Identifier\tSample Identifier\n"
           "#Patient Identifier\tSample Identifier\n"
           "#STRING\tSTRING\n"
           "#1\t1\n"
           "PATIENT_ID\tSAMPLE_ID\n"
           "P-0000001\tP-0000001-T01\n"
           "P-0000002\tP-0000002-T01\n")
    # mutations: plain tab-separated, no comment header; include a field with a comma
    _write(os.path.join(d, "data_mutations.txt"),
           "Hugo_Symbol\tSAMPLE_ID\tHGVSp\n"
           "TP53\tP-0000001-T01\tp.R175H, pathogenic\n"
           "KRAS\tP-0000002-T01\tp.G12D\n")
    _write(os.path.join(d, "data_sv.txt"),
           "Sample_Id\tSite1_Hugo_Symbol\nP-0000001-T01\tALK\n")
    _write(os.path.join(d, "data_timeline_treatment.txt"),
           "PATIENT_ID\tSTART_DATE\tEVENT_TYPE\nP-0000001\t10\tTreatment\n")
    # CNA: gene x sample matrix; mix of altered and zero calls
    _write(os.path.join(d, "data_cna.txt"),
           "Hugo_Symbol\tP-0000001-T01\tP-0000002-T01\n"
           "TP53\t-2\t0\n"
           "KRAS\t0\t2\n"
           "EGFR\t0\t0\n")


def test_preprocess_schema():
    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
        _make_raw_study(raw)
        summary = pp.preprocess(raw, out, smoke=False)

        # clinical: '#' rows dropped, comma-CSV with the real header
        with open(os.path.join(out, "data_clinical_patient.txt")) as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == ["PATIENT_ID", "OS_MONTHS"], rows[0].keys()
        assert {r["PATIENT_ID"] for r in rows} == {"P-0000001", "P-0000002"}

        # mutations: comma-CSV that round-trips a field containing a comma
        with open(os.path.join(out, "data_mutations.txt")) as f:
            mut = list(csv.DictReader(f))
        assert mut[0]["HGVSp"] == "p.R175H, pathogenic", mut[0]

        # CNA: melted long form, altered calls only, exact schema and order
        with open(os.path.join(out, "data_cna.txt")) as f:
            cna = list(csv.DictReader(f))
        assert list(cna[0].keys()) == ["SAMPLE_ID", "Hugo_Symbol", "CNA"], cna[0].keys()
        pairs = {(r["SAMPLE_ID"], r["Hugo_Symbol"], float(r["CNA"])) for r in cna}
        assert pairs == {
            ("P-0000001-T01", "TP53", -2.0),
            ("P-0000002-T01", "KRAS", 2.0),
        }, pairs  # the two 0 / EGFR calls are dropped
        assert summary["data_cna.txt"][0] == 2

        # timeline table is carried through as comma-CSV
        assert os.path.exists(os.path.join(out, "data_timeline_treatment.txt"))
    print("test_preprocess_schema PASSED")


if __name__ == "__main__":
    test_preprocess_schema()
