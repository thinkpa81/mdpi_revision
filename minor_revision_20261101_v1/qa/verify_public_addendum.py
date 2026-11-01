"""Verify this public addendum without reading source microdata or refitting models."""

from pathlib import Path
import ast
import csv
import hashlib
import json
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def verify():
    sums = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        sums[name] = expected
    released = {str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts}
    assert released == set(sums) | {"SHA256SUMS"}, "Manifest/file-set mismatch"
    counts = {"files": 0, "python_files": 0, "json_files": 0, "csv_files": 0}
    forbidden_extensions = {".docx", ".xlsx", ".xls", ".parquet", ".pkl", ".joblib", ".npy", ".npz"}
    forbidden_columns = {"id", "respondent_id", "participant_id", "person_id", "pid", "panel_id"}
    for name, expected in sums.items():
        path = ROOT / name
        assert path.suffix.lower() not in forbidden_extensions, name
        assert not {"private_raw", "private_data", "raw", "predictions", "models", "checkpoints"}.intersection(path.relative_to(ROOT).parts), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        counts["files"] += 1
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            counts["python_files"] += 1
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            counts["json_files"] += 1
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                assert not forbidden_columns.intersection(x.strip().lower() for x in fields), name
                assert fields and all(None not in row for row in reader), name
            counts["csv_files"] += 1

    bootstrap = ROOT / "bootstrap_verification"
    with tempfile.TemporaryDirectory(prefix="jtaer-aggregate-check-") as temp:
        command = [sys.executable, "code/verify_recomputed_segment.py",
                   "--new-tables", "exact_original_source/main_result/revision/tables",
                   "--historical-tables", "historical_tables",
                   "--historical-manifest", "historical_audit/round2_reselection_bootstrap_manifest.json",
                   "--environment", "rerun_environment.json", "--out", temp]
        subprocess.run(command, cwd=bootstrap, check=True, capture_output=True, text=True)
        report = json.loads((Path(temp) / "minor_reselection_software_verification.json").read_text())
        assert report["status"].lower() == "passed", report.get("status")

    import numpy as np
    from PIL import Image
    original = np.asarray(Image.open(ROOT / "figures/source_figure7.png").convert("RGB"))
    revised = np.asarray(Image.open(ROOT / "figures/Submission_Figure_07_Block_Heatmap.png").convert("RGB"))
    assert original.shape == revised.shape
    changed = np.flatnonzero(np.any(original != revised, axis=2).any(axis=1))
    assert np.all(((changed >= 226) & (changed <= 309)) | ((changed >= 2002) & (changed <= 2085)))
    assert (ROOT / "figures/data/submission_block_heatmap.csv").read_bytes() == (ROOT / "figures/data/original_submission_block_heatmap.csv").read_bytes()
    return {"status": "passed", **counts,
            "bootstrap_aggregate_verification": "passed",
            "figure_7_numeric_data_unchanged": True,
            "figure_7_changes_limited_to_panel_title_bands": True,
            "scope": "Integrity, syntax, public aggregate structure, bootstrap aggregate equality and Figure 7 pixel scope; no model fitting or raw-data access."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
