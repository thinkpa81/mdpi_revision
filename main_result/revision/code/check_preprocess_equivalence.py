import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_selection_bootstrap as bs
import holdout_calibration as hc


def run_checks(development, evaluation):
    development, evaluation = (Path(development), Path(evaluation))
    dev = pd.read_csv(development, dtype={"id": str})
    eva = pd.read_csv(evaluation, dtype={"id": str})
    x, z, columns = bs.prepare(dev, eva)
    a, target, indices = hc.prepared(dev, [eva])
    cases = {"H_B": ["H", "B"], "H_D": ["H", "D"], "D": ["D"], "NONE": [], "FULL": bs.BLOCK_ORDER}
    rows = []
    with threadpool_limits(limits=1):
        for algorithm, config in [
            ("ridge", 0.1),
            ("hgb", {"max_leaf_nodes": 7, "learning_rate": 0.05}),
        ]:
            for label, blocks in cases.items():
                expected, _ = bs.eng.fit_predict(dev, eva, blocks, algorithm, config)
                jj = bs.select_columns(blocks, columns)
                actual = (
                    bs.fit_array(
                        algorithm,
                        config,
                        bs.eng.SEED,
                        x[:, jj],
                        dev.outcome_any_use.to_numpy(int),
                        z[:, jj],
                    )
                    if blocks
                    else np.full(len(eva), (dev.outcome_any_use.sum() + 1) / (len(dev) + 2))
                )
                key = "_".join(blocks) or "NONE"
                kk = indices[key]
                holdout = hc.fit_arr(
                    a[:, kk],
                    dev.outcome_any_use.to_numpy(int),
                    target[0][:, kk],
                    algorithm,
                    config,
                )
                for implementation, prediction in [
                    ("selection_bootstrap", actual),
                    ("holdout_calibration", holdout),
                ]:
                    error = float(np.max(np.abs(prediction - expected)))
                    if not np.isfinite(error) or error >= 1e-10:
                        raise RuntimeError(
                            f"Preprocessing equivalence failed: {implementation}/{algorithm}/{label}: {error}"
                        )
                    rows.append(
                        {
                            "implementation": implementation,
                            "algorithm": algorithm,
                            "representation": label,
                            "blocks": blocks,
                            "config": config,
                            "maximum_absolute_prediction_difference": error,
                        }
                    )
    files = [
        development,
        evaluation,
        Path(__file__),
        Path(bs.__file__),
        Path(hc.__file__),
        Path(bs.eng.__file__),
    ]
    return {
        "status": "passed",
        "scope": "Specified configurations on complete development/evaluation inputs; not every CV fold or bootstrap draw.",
        "tolerance": 1e-10,
        "cases": rows,
        "maximum_absolute_prediction_difference": max(
            (r["maximum_absolute_prediction_difference"] for r in rows)
        ),
        "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("private_data"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("revision/audit/preprocessing_equivalence_validation.json"),
    )
    args = parser.parse_args()
    result = run_checks(args.data / "development.csv", args.data / "evaluation.csv")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cases", "source_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
