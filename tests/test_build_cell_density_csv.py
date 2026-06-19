"""Schema / coverage tests for scripts/build_cell_density_csv.py.

Asserts the measured-CSV builder produces the agent's lookup schema
(``file_name`` = ``{case_id}_TMA_IHC_{view}_{marker}_{0|1}.png``, ``value`` = % positive),
assigns the 0/1 core index deterministically (block asc, then grid position), drops
Missing / empty / non-matching rows, and that the coverage helper counts staged keys.

Runnable under pytest or standalone: ``python tests/test_build_cell_density_csv.py``.
"""
import csv
import importlib.util
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "build_cell_density_csv", os.path.join(REPO_ROOT, "scripts", "build_cell_density_csv.py"))
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)

_FAU_HEADER = ["Image", "Name", "Missing", "Case ID", "Positive %"]
_FAU_ROWS = [
    # same (296, TumorCenter, CD3) group, block1 -> cores ordered by grid Name
    ["TumorCenter_CD3_block1.svs", "2-1", "false", "296", "8.0"],   # idx 1 (grid 2-1)
    ["TumorCenter_CD3_block1.svs", "1-1", "false", "296", "12.5"],  # idx 0 (grid 1-1)
    ["TumorCenter_CD3_block2.svs", "1-1", "false", "296", "5.0"],   # idx 2 -> dropped (>1)
    ["InvasionFront_CD8_block1.svs", "1-1", "true", "296", "9.9"],  # Missing -> skipped
    ["InvasionFront_CD8_block1.svs", "1-1", "false", "296", ""],    # empty Positive % -> skipped
    ["not_a_tma_image.svs", "1-1", "false", "296", "3.0"],          # non-matching Image -> skipped
]


def _write_fau(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FAU_HEADER)
        w.writerows(_FAU_ROWS)


def test_build_measured_schema_and_ordering():
    with tempfile.TemporaryDirectory() as d:
        fau = os.path.join(d, "TMA_celldensity_measurements.csv")
        _write_fau(fau)
        measured = bd.build_measured(fau)

        assert measured == {
            "296_TMA_IHC_TumorCenter_CD3_0.png": 12.5,  # grid 1-1 sorts first
            "296_TMA_IHC_TumorCenter_CD3_1.png": 8.0,   # grid 2-1 second
        }, measured  # block2 core dropped; Missing/empty/non-matching skipped

        # round-trip through the writer and confirm the agent's columns
        out = os.path.join(d, "measured.csv")
        bd.write_csv(measured, out)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == ["file_name", "value"], rows[0].keys()
        assert {r["file_name"] for r in rows} == set(measured)
    print("test_build_measured_schema_and_ordering PASSED")


def test_coverage_helper():
    with tempfile.TemporaryDirectory() as d:
        cases = os.path.join(d, "cases")
        case = os.path.join(cases, "296")
        os.makedirs(case)
        # one marker (queryable) image and one H&E (not queryable) image
        for fn in ["TMA_IHC_TumorCenter_CD3_0.png", "TMA_IHC_TumorCenter_HE_0.png"]:
            open(os.path.join(case, fn), "w").close()
        staged = bd.staged_marker_keys(cases)
        assert staged == {"296_TMA_IHC_TumorCenter_CD3_0.png": "TMA_IHC_TumorCenter_CD3_0.png"}, staged
    print("test_coverage_helper PASSED")


if __name__ == "__main__":
    test_build_measured_schema_and_ordering()
    test_coverage_helper()
