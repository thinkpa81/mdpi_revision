"""Independent arithmetic and cohort verification of the minor-revision additions.

Reads the Numeric worksheet directly and does not import the production analysis.
Writes aggregate checks only; no respondent records or identifiers are exported.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "analysis" / "raw"
OUT = ROOT / "analysis" / "outputs"
QA = Path(__file__).resolve().parent
Z = NormalDist().inv_cdf(0.975)


def wilson(k, n):
    p = k / n
    center = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return center - half, center + half


def read_wave(year):
    keys = {2023: ("ID", "q1_8", "q15_6", "q23"),
            2024: ("id", "Q1_8", "Q15_6", "Q24"),
            2025: ("id", "q1_8", "q14_6", "q20")}[year]
    path = RAW / f"panel_{year}.xlsx"
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book["Numeric"]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    indices = [header.index(key) for key in keys]
    result = {}
    for row in rows:
        identifier, outcome, hesitation, experience = [row[i] for i in indices]
        if identifier is None:
            continue
        if identifier in result:
            raise AssertionError(f"Duplicate ID in {year}")
        result[identifier] = (outcome, hesitation, experience)
    book.close()
    return result, {"year": year, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(result), "fields": list(keys)}


def transition(base, follow, restricted):
    result = {}
    for key, (prior, hesitation, experience) in base.items():
        if key not in follow or follow[key][0] not in (1, 2, 3, 4, 5, 9):
            continue
        if restricted and hesitation not in (1, 2):
            continue
        outcome = int(follow[key][0] in (1, 2, 3, 4))
        if experience not in (1, 2):
            raise AssertionError("A requires explicit invalid-code handling")
        result[key] = {"A": int(experience == 1), "Y": outcome, "hesitation": hesitation}
    return result


def expected_rows(name):
    with (OUT / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check_close(observed, expected, label, checks):
    ok = math.isclose(float(observed), float(expected), abs_tol=1e-9, rel_tol=1e-11)
    checks.append({"label": label, "observed": observed, "reported": float(expected), "pass": ok})
    if not ok:
        raise AssertionError(f"Mismatch: {label}")


def main():
    waves, source = {}, []
    for year in (2023, 2024, 2025):
        waves[year], audit = read_wave(year)
        source.append(audit)
    full_dev = transition(waves[2023], waves[2024], False)
    full_eval = transition(waves[2024], waves[2025], False)
    dev = transition(waves[2023], waves[2024], True)
    ev = transition(waves[2024], waves[2025], True)
    checks = []
    control = {
        "2023 annual rows": (len(waves[2023]), 4581),
        "2024 annual rows": (len(waves[2024]), 4420),
        "2025 annual rows": (len(waves[2025]), 4324),
        "full development n": (len(full_dev), 3895),
        "full development events": (sum(r["Y"] for r in full_dev.values()), 2215),
        "full evaluation n": (len(full_eval), 3741),
        "full evaluation events": (sum(r["Y"] for r in full_eval.values()), 2125),
        "restricted development n": (len(dev), 1565),
        "restricted development events": (sum(r["Y"] for r in dev.values()), 880),
        "restricted evaluation n": (len(ev), 1520),
        "restricted evaluation events": (sum(r["Y"] for r in ev.values()), 853),
        "restricted overlap": (len(set(dev) & set(ev)), 641),
        "full overlap": (len(set(full_dev) & set(full_eval)), 3333)
    }
    for label, (actual, expected) in control.items():
        check_close(actual, expected, label, checks)
    cohorts = {"restricted_development": dev, "restricted_evaluation": ev}
    calculated = {}
    for sample, cohort in cohorts.items():
        n = len(cohort)
        yes = [r for r in cohort.values() if r["A"]]
        no = [r for r in cohort.values() if not r["A"]]
        a = sum(r["Y"] for r in yes)
        b = len(yes) - a
        c = sum(r["Y"] for r in no)
        d = len(no) - c
        p1, p0 = a / (a + b), c / (c + d)
        odds = a * d / (b * c)
        se = math.sqrt(1/a + 1/b + 1/c + 1/d)
        ci1, ci0 = wilson(a, a+b), wilson(c, c+d)
        delta = p1 - p0
        lo = delta - math.sqrt((p1-ci1[0])**2 + (ci0[1]-p0)**2)
        hi = delta + math.sqrt((ci1[1]-p1)**2 + (p0-ci0[0])**2)
        pr = wilson(len(yes), n)
        values = {
            "n": n, "A_valid_n": n, "A_missing_n": 0, "A_yes_n": len(yes), "A_no_n": len(no),
            "A_yes_pct": 100*len(yes)/n, "A_yes_pct_pointwise_95_low": 100*pr[0],
            "A_yes_pct_pointwise_95_high": 100*pr[1], "outcome_events": a+c,
            "outcome_pct": 100*(a+c)/n, "complete_n": n, "A_yes_outcome_events": a,
            "A_yes_outcome_pct": 100*p1, "A_no_outcome_events": c, "A_no_outcome_pct":100*p0,
            "risk_difference_pp":100*delta,"risk_difference_pp_pointwise_95_low":100*lo,
            "risk_difference_pp_pointwise_95_high":100*hi,"unadjusted_odds_ratio": odds,
            "unadjusted_OR_pointwise_95_low": math.exp(math.log(odds)-Z*se),
            "unadjusted_OR_pointwise_95_high": math.exp(math.log(odds)+Z*se),
            "phi_correlation": (a*d-b*c)/math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
        }
        calculated[sample] = values
    for filename in ("A_prevalence.csv", "A_outcome_association.csv"):
        for reported in expected_rows(filename):
            for key, value in calculated[reported["sample"]].items():
                if key in reported:
                    check_close(value, reported[key], f"{filename}:{reported['sample']}:{key}", checks)

    groups = {
        "all_linked": lambda h: True,
        "hesitant": lambda h: h in (1,2),
        "other": lambda h: h not in (1,2),
        "rare_or_never": lambda h: h in (3,4),
        "no_experience": lambda h: h == 9
    }
    overlaps = {}
    for group, predicate in groups.items():
        members = {k:r for k,r in full_eval.items() if predicate(r["hesitation"])}
        shared = {k:r for k,r in members.items() if k in full_dev}
        separate = {k:r for k,r in members.items() if k not in full_dev}
        n, m = len(members), len(shared)
        overlaps[group] = {"evaluation_n":n,"overlap_with_full_development_n":m,
            "overlap_with_full_development_pct":100*m/n,
            "person_disjoint_from_full_development_n":n-m,
            "person_disjoint_from_full_development_pct":100*(n-m)/n,
            "overlap_with_restricted_development_n_for_reference_only":len(set(members)&set(dev)),
            "overlap_with_restricted_development_pct_for_reference_only":100*len(set(members)&set(dev))/n,
            "outcome_events":sum(r["Y"] for r in members.values()),
            "shared_outcome_events":sum(r["Y"] for r in shared.values()),
            "disjoint_outcome_events":sum(r["Y"] for r in separate.values())}
    aliases = {"other_respondents":"other","other_linked":"other","rare_never":"rare_or_never",
               "rare_or_never_hesitant":"rare_or_never","no_relevant_experience":"no_experience"}
    for reported in expected_rows("full_panel_group_overlap.csv"):
        group = aliases.get(reported["evaluation_group"], reported["evaluation_group"])
        for key,value in overlaps[group].items():
            check_close(value,reported[key],f"overlap:{group}:{key}",checks)
    cells = {(a,b):0 for a in (0,1) for b in (0,1)}
    for key in set(dev)&set(ev):
        cells[dev[key]["A"],ev[key]["A"]] += 1
    for reported in expected_rows("A_shared_participant_transitions.csv"):
        pair = int(reported["A_2023"]),int(reported["A_2024"])
        check_close(cells[pair],reported["count"],f"Apaired:{pair}",checks)
    result = {"status":"PASS", "method":"Independent direct openpyxl Numeric-worksheet joins; no production analysis functions imported",
              "sources":source,"historical_controls":13,"checks_passed":len(checks),
              "A_results":calculated,"full_panel_group_overlap":overlaps,
              "A_paired_transition_counts":[{"A2023":a,"A2024":b,"count":v} for (a,b),v in cells.items()],
              "checks":checks,"privacy":"Only aggregate results exported; raw records and identifiers remain local"}
    (QA/"independent_numeric_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"historical_controls":13,"checks_passed":len(checks),
                      "A":calculated,"overlap":overlaps},indent=2))


if __name__ == "__main__":
    main()
