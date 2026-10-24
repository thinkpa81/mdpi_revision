from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def context(root, fitting_script, grids, seed):
    root = Path(root)
    files = [
        root / "private_data/development.csv",
        root / "private_data/evaluation.csv",
        Path(fitting_script),
        root / "revision/code/original_model_engine.py",
        Path(__file__),
    ]
    payload = {
        "schema": 1,
        "inputs_and_dependencies": {str(p.relative_to(root)): sha(p) for p in files},
        "grids": grids,
        "seed_base": seed,
        "budgets": [1, 4],
        "folds": 5,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def receipt_path(path):
    return Path(path).with_suffix(".receipt.json")


def verify_checkpoint(path, run_context):
    path = Path(path)
    receipt = receipt_path(path)
    if not receipt.is_file():
        raise RuntimeError(
            f"Unverified bootstrap checkpoint: {path.name}. Use a clean run directory; historical checkpoints are not adopted automatically."
        )
    value = json.loads(receipt.read_text())
    if value.get("run_fingerprint") != run_context["fingerprint"] or value.get(
        "output_sha256"
    ) != sha(path):
        raise RuntimeError(
            f"Bootstrap checkpoint provenance mismatch: {path.name}. Keep the historical run separately and recompute in a clean directory."
        )
    return value


def ensure_directory(directory, run_context):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = directory / "run_context.json"
    existing = list(directory.glob("replicate_[0-9][0-9][0-9][0-9].json"))
    if binding.exists():
        if json.loads(binding.read_text()) != run_context:
            raise RuntimeError(
                "Bootstrap input/code/grid/seed/environment changed. Refusing stale checkpoints; use a clean run directory."
            )
    elif existing:
        raise RuntimeError(
            "Historical bootstrap checkpoints lack a current run binding. Preserve them and use a clean run directory; no silent cache adoption is allowed."
        )
    else:
        atomic_json(binding, run_context)
    for path in existing:
        verify_checkpoint(path, run_context)


def record_checkpoint(path, run_context):
    path = Path(path)
    payload = json.loads(path.read_text())
    if len(payload.get("rows", [])) != 8:
        raise RuntimeError("Incomplete bootstrap result cannot receive a completion receipt.")
    rows = pd.DataFrame(payload["rows"])
    if not np.isfinite(rows.select_dtypes("number").to_numpy()).all():
        raise RuntimeError("Nonfinite bootstrap result cannot receive a completion receipt.")
    atomic_json(
        receipt_path(path),
        {
            "schema": 1,
            "run_fingerprint": run_context["fingerprint"],
            "output_sha256": sha(path),
            "replicate": payload["replicate"],
            "status": "complete",
        },
    )
