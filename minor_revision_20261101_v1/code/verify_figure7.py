from pathlib import Path
import hashlib
import json

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
QA = ROOT / "qa"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before_path = FIGURES / "source_figure7.png"
    after_path = FIGURES / "Submission_Figure_07_Block_Heatmap.png"
    before = np.array(Image.open(before_path).convert("RGB"))
    after = np.array(Image.open(after_path).convert("RGB"))
    assert before.shape == after.shape
    different = np.any(before != after, axis=2)
    changed_rows = np.flatnonzero(different.any(axis=1))
    assert np.all(((changed_rows >= 226) & (changed_rows <= 309)) | ((changed_rows >= 2002) & (changed_rows <= 2085)))
    source_path = FIGURES / "data/submission_block_heatmap.csv"
    original_path = FIGURES / "data/original_submission_block_heatmap.csv"
    assert source_path.read_bytes() == original_path.read_bytes()
    source = pd.read_csv(source_path)
    selected = source[source.outlined]
    assert len(source) == 84
    assert len(selected) == 7
    assert (selected.block == "D").sum() == 6
    assert (selected.block == "A").sum() == 1
    audit = {
        "status": "PASS",
        "scope": "Panel-name prefixes changed from A–D to (a)–(d); values, intervals and significance are unchanged.",
        "source_commit": "af06ecda65c0f5768cbe2b3449bb46ebf229edca",
        "original_script_url": "https://github.com/thinkpa81/mdpi_revision/blob/af06ecda65c0f5768cbe2b3449bb46ebf229edca/main_result/revision/code/plot_submission_block_heatmap.py",
        "original_script_sha256": digest(ROOT / "code/original_plot_submission_block_heatmap.py"),
        "regeneration_script_sha256": digest(ROOT / "code/regenerate_figure7.py"),
        "original_figure_sha256": digest(before_path),
        "revised_figure_sha256": digest(after_path),
        "pixel_dimensions": [before.shape[1], before.shape[0]],
        "changed_pixels": int(different.sum()),
        "changed_row_bands_zero_based_inclusive": [[226, 309], [2002, 2085]],
        "all_pixels_outside_panel_title_bands_identical": True,
        "all_84_source_rows_byte_identical": True,
        "full_precision_values_and_intervals_unchanged": True,
        "outlined_cells": 7,
        "outlined_D_cells": 6,
        "outlined_A_cells": 1,
        "source_sha256": {p.name: digest(p) for p in sorted((FIGURES / "data").glob("*.csv"))},
        "runtime": {"matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "figure7_relabel_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
