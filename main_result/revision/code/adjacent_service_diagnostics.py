from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm

YEARS = (2023, 2024, 2025)
DOMAINS = {"leisure": "q1_7", "consumer": "q1_8", "work": "q1_9"}
THRESHOLDS = {"past_year": (1, 2, 3, 4), "monthly": (1, 2, 3), "weekly": (1, 2)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def paired_contrasts(arr, n_boot, seed):
    vectors, labels = ([], [])
    for ti, threshold in enumerate(THRESHOLDS):
        for ai, anchor in [(0, "leisure"), (2, "work")]:
            gap = arr[:, :, 1, ti] - arr[:, :, ai, ti]
            a, b = (gap[:, 1] - gap[:, 0], gap[:, 2] - gap[:, 1])
            for period, z in [
                ("2023_to_2024", a),
                ("2024_to_2025", b),
                ("change_in_differential_change", b - a),
            ]:
                labels.append(
                    {"threshold": threshold, "adjacent_domain": anchor, "contrast": period}
                )
                vectors.append(z)
    z = np.column_stack(vectors).astype(float)
    n = len(z)
    point = z.mean(axis=0)
    se = z.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(se > 0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, z.shape[1]))
    for start in range(0, n_boot, 100):
        stop = min(n_boot, start + 100)
        indices = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = z[indices].mean(axis=1)
    standardized_max = np.max(np.abs((boot - point) / se), axis=1)
    critical = float(np.quantile(standardized_max, 0.95, method="linear"))
    pw = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")
    rows = []
    for i, label in enumerate(labels):
        rows.append(
            {
                **label,
                "n_participants": n,
                "estimate": point[i],
                "standard_error": se[i],
                "pointwise_percentile_lower95": pw[0, i],
                "pointwise_percentile_upper95": pw[1, i],
                "simultaneous_lower95": point[i] - critical * se[i],
                "simultaneous_upper95": point[i] + critical * se[i],
                "normal_two_sided_p": 2 * norm.sf(abs(point[i] / se[i])),
                "bootstrap_max_adjusted_p": (1 + np.sum(standardized_max >= abs(point[i] / se[i])))
                / (n_boot + 1),
                "bootstrap_repetitions": n_boot,
                "bootstrap_seed": seed,
                "family_size": len(labels),
                "max_standardized_critical95": critical,
            }
        )
    return rows


def logistic_cluster(x, y, cluster):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def objective(beta):
        eta = x @ beta
        return float(np.logaddexp(0, eta).sum() - y @ eta)

    def gradient(beta):
        return x.T @ (expit(x @ beta) - y)

    def hessian(beta):
        mu = expit(x @ beta)
        return x.T @ ((mu * (1 - mu))[:, None] * x)

    opt = minimize(
        objective,
        np.zeros(x.shape[1]),
        method="trust-exact",
        jac=gradient,
        hess=hessian,
        options={"gtol": 1e-08, "maxiter": 200},
    )
    beta = opt.x
    for _ in range(5):
        score = gradient(beta)
        candidate = beta - np.linalg.solve(hessian(beta), score)
        if np.max(np.abs(gradient(candidate))) < np.max(np.abs(score)):
            beta = candidate
        else:
            break
    g = gradient(beta)
    assert np.max(np.abs(g)) < 1e-06, (opt.message, g)
    H = hessian(beta)
    assert np.linalg.matrix_rank(H) == x.shape[1]
    assert np.max(np.abs(beta)) < 25, "Potential separation; do not report Wald inference."
    bread = np.linalg.inv(H)
    scores = np.zeros((int(np.max(cluster)) + 1, x.shape[1]))
    np.add.at(scores, cluster, x * (y - expit(x @ beta))[:, None])
    cov = bread @ (scores.T @ scores) @ bread
    assert np.isfinite(cov).all() and np.linalg.eigvalsh(cov).min() > -1e-10
    return (beta, cov, float(np.max(np.abs(g))), float(np.linalg.cond(H)))


def conditional_year_diagnostics(arr):
    n = len(arr)
    cluster = np.repeat(np.arange(n), 3)
    years = np.tile(np.array(YEARS), n)
    out, coefficients, checks = ([], [], [])
    for ti, threshold in enumerate(THRESHOLDS):
        y = arr[:, :, 1, ti].reshape(-1)
        pattern = (arr[:, :, 0, ti] * 2 + arr[:, :, 2, ti]).reshape(-1)
        anchors = np.column_stack([(pattern == i).astype(float) for i in (1, 2, 3)])
        periods = np.column_stack([(years == year).astype(float) for year in (2024, 2025)])
        names = [
            "intercept",
            "leisure0_work1",
            "leisure1_work0",
            "leisure1_work1",
            "year2024",
            "year2025",
        ]
        x = np.column_stack([np.ones(len(y)), anchors, periods])
        beta, cov, grad, cond = logistic_cluster(x, y, cluster)
        for name, coef, se in zip(names, beta, np.sqrt(np.diag(cov))):
            coefficients.append(
                {
                    "threshold": threshold,
                    "model": "additive_year_observed_anchor_pattern",
                    "term": name,
                    "coefficient": coef,
                    "cluster_robust_se": se,
                }
            )
        for contrast, r in [
            ("2023_to_2024", np.array([0, 0, 0, 0, 1, 0])),
            ("2024_to_2025", np.array([0, 0, 0, 0, -1, 1])),
            ("change_in_conditional_log_odds_change", np.array([0, 0, 0, 0, -2, 1])),
        ]:
            estimate = float(r @ beta)
            se = float(np.sqrt(r @ cov @ r))
            out.append(
                {
                    "threshold": threshold,
                    "test": contrast,
                    "n_participants": n,
                    "estimate_log_odds": estimate,
                    "cluster_robust_se": se,
                    "odds_ratio": np.exp(estimate),
                    "odds_ratio_lower95": np.exp(estimate - 1.96 * se),
                    "odds_ratio_upper95": np.exp(estimate + 1.96 * se),
                    "statistic": (estimate / se) ** 2,
                    "degrees_of_freedom": 1,
                    "p_value": float(2 * norm.sf(abs(estimate / se))),
                }
            )
        interactions = np.column_stack(
            [anchors[:, j] * periods[:, k] for k in range(2) for j in range(3)]
        )
        xfull = np.column_stack([x, interactions])
        bf, vf, gf, cf = logistic_cluster(xfull, y, cluster)
        expected = np.empty_like(y, dtype=float)
        for year in YEARS:
            for p in range(4):
                mask = (years == year) & (pattern == p)
                assert mask.sum() > 0 and 0 < y[mask].mean() < 1
                expected[mask] = y[mask].mean()
        max_saturated_error = float(np.max(np.abs(expit(xfull @ bf) - expected)))
        assert max_saturated_error < 1e-08
        stat = float(bf[6:] @ np.linalg.solve(vf[6:, 6:], bf[6:]))
        out.append(
            {
                "threshold": threshold,
                "test": "year_by_observed_anchor_pattern_interactions",
                "n_participants": n,
                "estimate_log_odds": None,
                "cluster_robust_se": None,
                "odds_ratio": None,
                "odds_ratio_lower95": None,
                "odds_ratio_upper95": None,
                "statistic": stat,
                "degrees_of_freedom": 6,
                "p_value": float(chi2.sf(stat, 6)),
            }
        )
        checks.append(
            {
                "threshold": threshold,
                "additive_max_abs_score": grad,
                "saturated_max_abs_score": gf,
                "additive_hessian_condition": cond,
                "saturated_hessian_condition": cf,
                "saturated_fitted_probability_max_abs_difference_from_group_means": max_saturated_error,
            }
        )
    adjusted = holm([r["p_value"] for r in out])
    for r, p in zip(out, adjusted):
        r["holm_p_12_tests_within_cohort"] = p
        r["interpretation"] = (
            "Observed-anchor-conditioned year association; not identified DIF or a wording effect"
        )
    return (out, coefficients, checks)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paired adjacent-domain response diagnostics from authorized raw waves.\n\nThese are domain/period and anchor-conditioned response-association diagnostics,\nnot a test that establishes measurement invariance or identifies wording effects.\nNo respondent identifiers or individual records are written to the public output.\n"
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026102131)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waves, sources = ({}, [])
    for year in YEARS:
        source = args.raw_dir / f"{year}_data.xlsx"
        needed = {"id", "q1_7", "q1_8", "q1_9", "q15_6" if year < 2025 else "q14_6"}
        raw = pd.read_excel(
            source, sheet_name="Numeric", usecols=lambda c: str(c).lower() in needed
        )
        raw.columns = raw.columns.str.lower()
        raw["id"] = raw.id.astype(str)
        assert raw.id.is_unique and raw.id.notna().all()
        waves[year] = raw.set_index("id")
        sources.append(
            {"year": year, "file": source.name, "sha256": sha(source), "rows": len(raw)}
        )
    common = sorted(set.intersection(*(set(w.index) for w in waves.values())))
    complete = [
        i
        for i in common
        if all(
            (waves[y].loc[i, list(DOMAINS.values())].isin([1, 2, 3, 4, 5, 9]).all() for y in YEARS)
        )
    ]
    restricted = [
        i
        for i in complete
        if waves[2023].loc[i, "q15_6"] in (1, 2) and waves[2024].loc[i, "q15_6"] in (1, 2)
    ]
    assert len(common) == len(complete) == 3333
    assert len(restricted) == 641
    rates, contrasts, logistic, coefficients, fit_checks = ([], [], [], [], {})
    for ci, (cohort, ids) in enumerate(
        [("restricted_shared", restricted), ("full_three_wave_panel", complete)]
    ):
        values = np.stack(
            [waves[y].loc[ids, list(DOMAINS.values())].to_numpy() for y in YEARS], axis=1
        )
        arr = np.stack(
            [np.isin(values, codes).astype(int) for codes in THRESHOLDS.values()], axis=3
        )
        for yi, year in enumerate(YEARS):
            for di, domain in enumerate(DOMAINS):
                for ti, threshold in enumerate(THRESHOLDS):
                    rates.append(
                        {
                            "cohort": cohort,
                            "year": year,
                            "domain": domain,
                            "threshold": threshold,
                            "n_participants": len(ids),
                            "events": int(arr[:, yi, di, ti].sum()),
                            "prevalence": float(arr[:, yi, di, ti].mean()),
                        }
                    )
        contrasts.extend(
            (
                {"cohort": cohort, **r}
                for r in paired_contrasts(arr, args.bootstrap_repetitions, args.seed + ci)
            )
        )
        tests, coefs, checks = conditional_year_diagnostics(arr)
        logistic.extend(({"cohort": cohort, **r} for r in tests))
        coefficients.extend(({"cohort": cohort, **r} for r in coefs))
        fit_checks[cohort] = checks
    rates = pd.DataFrame(rates)
    expected = {
        ("consumer", "past_year"): (427, 382),
        ("consumer", "monthly"): (366, 268),
        ("consumer", "weekly"): (189, 121),
        ("leisure", "past_year"): (204, 217),
        ("leisure", "monthly"): (131, 90),
        ("leisure", "weekly"): (58, 34),
        ("work", "past_year"): (243, 279),
        ("work", "monthly"): (183, 183),
        ("work", "weekly"): (107, 74),
    }
    s = rates[rates.cohort.eq("restricted_shared")].set_index(["domain", "threshold", "year"])
    for (domain, threshold), counts in expected.items():
        assert (
            tuple((int(s.loc[(domain, threshold, y), "events"]) for y in (2024, 2025))) == counts
        )
    outputs = {
        "adjacent_domain_rates.csv": rates,
        "adjacent_domain_differences.csv": pd.DataFrame(contrasts),
        "adjacent_conditional_year_tests.csv": pd.DataFrame(logistic),
        "adjacent_logistic_coefficients.csv": pd.DataFrame(coefficients),
    }
    display_rows = []
    for (cohort, domain, threshold), group in rates.groupby(
        ["cohort", "domain", "threshold"], sort=False
    ):
        by_year = group.set_index("year")
        display_rows.append(
            {
                "cohort": cohort,
                "domain": domain,
                "threshold": threshold,
                "n_participants": int(group.n_participants.iloc[0]),
                **{f"events_{year}": int(by_year.loc[year, "events"]) for year in YEARS},
                **{
                    f"percent_{year}": float(100 * by_year.loc[year, "prevalence"])
                    for year in YEARS
                },
                "change_2024_minus_2023_pp": float(
                    100 * (by_year.loc[2024, "prevalence"] - by_year.loc[2023, "prevalence"])
                ),
                "change_2025_minus_2024_pp": float(
                    100 * (by_year.loc[2025, "prevalence"] - by_year.loc[2024, "prevalence"])
                ),
            }
        )
    outputs["adjacent_service_table.csv"] = pd.DataFrame(display_rows)
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    audit = [
        {
            "year": 2023,
            "domain": "leisure",
            "pdf_page": 4,
            "finding": "Q1 leisure examples include travel assistant and metaverse games.",
        },
        {
            "year": 2024,
            "domain": "leisure",
            "pdf_page": 4,
            "finding": "Q1 leisure replaces metaverse games with conversational assistant; its definition appears on PDF page 5.",
        },
        {
            "year": 2025,
            "domain": "leisure",
            "pdf_page": 10,
            "finding": "Extended leisure explanation and examples include AiRSPACE, translation services, Netflix and Genie Music.",
        },
        {
            "year": 2025,
            "domain": "consumer",
            "pdf_page": 10,
            "finding": "Extended consumer explanation and examples include Naver Plus Store and Kurly basket recommendations.",
        },
        {
            "year": 2025,
            "domain": "work",
            "pdf_page": 4,
            "finding": "Q1 work examples add coding and transcription relative to 2024.",
        },
        {
            "year": 2025,
            "domain": "work",
            "pdf_page": 11,
            "finding": "Extended work explanation and examples include ChatGPT, Perplexity, Clova Note and Copilot.",
        },
        {
            "year": 2025,
            "domain": "all_Q1_domains",
            "pdf_page": 4,
            "finding": "Q1 stem explicitly adds frequency wording and the general introductory definition changes.",
        },
    ]
    pd.DataFrame(audit).to_csv(
        args.output_dir / "adjacent_questionnaire_anchor_audit.csv", index=False
    )
    manifest = {
        "status": "passed",
        "sources": sources,
        "questionnaire_sources": [
            {
                "year": y,
                "file": f"{y}_questionnaire.pdf",
                "sha256": sha(args.raw_dir / f"{y}_questionnaire.pdf"),
            }
            for y in YEARS
        ],
        "cohorts": {"restricted_shared": 641, "full_three_wave_panel": 3333},
        "complete_case_rule": "Observed in all three years with valid responses to Q1_7, Q1_8 and Q1_9 in every year.",
        "restricted_shared_rule": "The complete three-wave panel additionally has purchase-hesitation codes 1 or 2 in both 2023 and 2024.",
        "missing_adjacent_responses_among_three_wave_panel": len(common) - len(complete),
        "original_S31_exact_count_rows_reproduced": 9,
        "paired_contrast_definition": "Within each participant, change in consumer response minus change in adjacent-domain response; acceleration subtracts the 2023–2024 contrast from 2024–2025.",
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed_restricted": args.seed,
            "seed_full": args.seed + 1,
            "unit": "participant, retaining all nine year/domain responses and all thresholds",
            "simultaneous_family": "18 two-sided contrasts within each cohort: 3 thresholds x 2 adjacent domains x 3 period contrasts",
            "critical_value": "95th linear quantile of max absolute centered bootstrap mean divided by original standard error",
            "quantiles": "NumPy method=linear",
            "survey_weights": False,
        },
        "conditional_logistic": {
            "response": "consumer-service binary response",
            "covariates": "Categorical four-pattern leisure/work response and categorical year",
            "covariance": "Participant-cluster sandwich, independent working correlation, no finite-cluster correction",
            "interaction_diagnostic": "Wald joint test of six year-by-observed-pattern interaction coefficients",
            "multiplicity": "Holm across 12 conditional tests within each cohort",
            "fit_checks": fit_checks,
        },
        "identification": {
            "formal_measurement_invariance_established": False,
            "causal_wording_effect_identified": False,
            "adjacent_domains_are_validated_invariant_anchors": False,
            "reasons": [
                "Adjacent services are different behaviors, not validated parallel indicators of one latent construct.",
                "Leisure wording changes in 2024; work wording changes in 2025; the 2025 questionnaire contains extended explanations for all three domains. Individual exposure and administration are unobserved.",
                "Year-specific genuine use changes and measurement changes remain observationally confounded.",
                "The 2023–2024 interval is a historical comparison, not an untreated negative-control period.",
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "script_sha256": sha(Path(__file__)),
        "public_outputs": [
            {"file": name, "rows": len(frame), "sha256": sha(args.output_dir / name)}
            for name, frame in outputs.items()
        ],
    }
    (args.output_dir / "adjacent_methods.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "cohorts": manifest["cohorts"],
                "S31_rows_reproduced": 9,
                "response_contrasts": len(contrasts),
                "conditional_tests": len(logistic),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
