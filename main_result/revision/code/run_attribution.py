from pathlib import Path
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
import argparse, json, math, hashlib
import numpy as np
import pandas as pd

BLOCKS = ["D", "B", "M", "P", "C", "A", "K"]
COST = np.array([4, 1, 1, 11, 1, 1, 11])


def loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def exact_shapley(L):
    phi = np.zeros((L.shape[0], 7))
    for j in range(7):
        for m in range(128):
            if m & 1 << j:
                continue
            k = m.bit_count()
            phi[:, j] += (
                math.factorial(k)
                * math.factorial(6 - k)
                / math.factorial(7)
                * (L[:, m] - L[:, m | 1 << j])
            )
    assert np.allclose(phi.sum(1), L[:, 0] - L[:, -1], atol=1e-12)
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()
    r = args.root
    ev = pd.read_csv(r / "private_data/evaluation.csv", dtype={"id": str})
    dev = pd.read_csv(r / "private_data/development.csv", dtype={"id": str})
    fp = r / "private_data/refit_all_predictions.npz"
    pred = np.load(fp)
    y = ev.outcome_any_use.to_numpy()
    parts = []
    met = []
    checks = []
    for alg in ["ridge", "hgb"]:
        for h in [True, False]:
            keys = []
            for m in range(128):
                bs = (["H"] if h else []) + [b for j, b in enumerate(BLOCKS) if m & 1 << j]
                keys.append(alg + "::" + ("_".join(bs) if bs else "NONE"))
            L = np.column_stack([loss(y, pred[k]) for k in keys])
            ph = exact_shapley(L)
            parts.append(ph)
            for j, b in enumerate(BLOCKS):
                met.append(dict(algorithm=alg, history_fixed=h, block=b, items=int(COST[j])))
            checks.append(
                dict(
                    algorithm=alg,
                    history_fixed=h,
                    max_efficiency_error=float(np.max(abs(ph.sum(1) - (L[:, 0] - L[:, -1])))),
                )
            )
    allphi = np.column_stack(parts)
    mask = ~ev.id.isin(dev.id).to_numpy()
    means = np.r_[allphi.mean(0), allphi[mask].mean(0)]
    rng = np.random.default_rng(2026102021)
    boot = []
    for start in range(0, args.reps, 100):
        w = rng.multinomial(
            len(ev), np.full(len(ev), 1 / len(ev)), size=min(100, args.reps - start)
        )
        boot.append(
            np.c_[w @ allphi / len(ev), w[:, mask] @ allphi[mask] / w[:, mask].sum(1)[:, None]]
        )
    boot = np.vstack(boot)
    crit = float(np.quantile(np.max(abs(boot - means), axis=1), 0.95))
    rows = []
    for domain, offset, n in [("whole", 0, len(ev)), ("person_disjoint", 28, int(mask.sum()))]:
        for j, meta in enumerate(met):
            k = j + offset
            v = means[k]
            lo, hi = np.quantile(boot[:, k], [0.025, 0.975])
            cost = meta["items"]
            rows.append(
                dict(
                    **meta,
                    domain=domain,
                    n=n,
                    loss_reduction=v,
                    ci95_lower=lo,
                    ci95_upper=hi,
                    simultaneous95_lower=v - crit,
                    simultaneous95_upper=v + crit,
                    loss_reduction_per_item=v / cost,
                    per_item_ci95_lower=lo / cost,
                    per_item_ci95_upper=hi / cost,
                    per_item_simultaneous95_lower=(v - crit) / cost,
                    per_item_simultaneous95_upper=(v + crit) / cost
                )
            )
    out = r / "revision/tables"
    out.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(out / "attribution_per_item.csv", index=False)
    audit = dict(
        repetitions=args.reps,
        seed=2026102021,
        conditional_on_fitted_predictions=True,
        family_size=56,
        simultaneous_method="95th percentile of max absolute centered paired bootstrap deviation across 56 block/domain/model contrasts",
        critical_value=crit,
        efficiency=checks,
        positive="reduction in log loss",
        per_item="Each block allocation divided by original item count; not a cost-weighted Shapley game.",
        predictions_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),
    )
    (r / "revision/audit/attribution_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        pd.DataFrame(rows)
        .query("domain=='whole'")[
            ["algorithm", "history_fixed", "block", "loss_reduction", "loss_reduction_per_item"]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
