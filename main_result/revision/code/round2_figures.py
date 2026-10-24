from pathlib import Path
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import json, hashlib, math, tempfile
from decimal import Decimal
import xml.etree.ElementTree as ET
from PIL import Image
from pypdf import PdfReader
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "JTAER_Figures"
NOTE = ROOT / "analysis/round2"
TAB = ROOT / "revision/tables"
ORIG = ROOT / "original_results/main_result/mdpi-main"
OLD = ORIG / "01_required_prior_analysis/revision_20260907/outputs"
BLUE, ORANGE, GRAY = ("#23618C", "#C06332", "#5E6871")
BLOCKS = ["D", "B", "M", "P", "C", "A", "K"]
COST = dict(zip(BLOCKS, [4, 1, 1, 11, 1, 1, 11]))
LABELS = {
    "D": "D · Demographics",
    "B": "B · Expected benefit",
    "M": "M · Data management",
    "P": "P · Safeguards",
    "C": "C · Consent response",
    "A": "A · Generative AI use",
    "K": "K · Digital skills",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9DA5AC",
        "axes.linewidth": 0.6,
        "grid.color": "#E0E5E8",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
MANIFEST, CHECKS, CAPTIONS = ([], {}, [])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    return {"file": str(path.relative_to(ROOT)), "sha256": sha(path)}


def sync_file(path):
    with path.open("rb") as f:
        os.fsync(f.fileno())


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.stem + "-",
        suffix=".tmp",
        delete=False,
    ) as f:
        stage = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            stage.unlink(missing_ok=True)
            raise
    try:
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def validate_exports(paths):
    png, pdf, svg, preview = paths
    for p in (png, preview):
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            assert im.mode in ("RGB", "RGBA") and min(im.size) > 100
    with Image.open(png) as im:
        pixel_size = im.size
        dpi = im.info["dpi"]
    assert all((abs(d - 600) < 0.02 for d in dpi))
    reader = PdfReader(str(pdf), strict=True)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    pdf_pt = (float(page.mediabox.width), float(page.mediabox.height))
    assert all((x > 0 for x in pdf_pt))
    assert len(page.extract_text().strip()) > 20
    assert len(page.get_contents().get_data()) > 100
    xml = ET.parse(svg)
    root = xml.getroot()
    assert root.tag.endswith("svg")
    svg_pt = tuple((float(root.attrib[k].removesuffix("pt")) for k in ("width", "height")))
    assert len(list(root.iter())) > 20
    pdf_text = "".join(page.extract_text().split())
    svg_labels = [
        "".join("".join(e.itertext()).split()) for e in root.iter() if e.tag.endswith("}text")
    ]
    assert all((t in pdf_text for t in svg_labels))
    svg_pdf_error = max((abs(a - b) for a, b in zip(svg_pt, pdf_pt)))
    png_pdf_error = max((abs(px / 600 * 72 - pt) for px, pt in zip(pixel_size, pdf_pt)))
    assert max(svg_pdf_error, png_pdf_error) < 2.0
    assert max(svg_pdf_error, png_pdf_error) / min(pdf_pt) < 0.005
    return {
        "png_pixels": list(pixel_size),
        "png_dpi": list(dpi),
        "pdf_pages": len(reader.pages),
        "pdf_size_points": list(pdf_pt),
        "svg_size_points": list(svg_pt),
        "png_verify_and_decode": "pass",
        "pdf_stream_validation": "pass",
        "svg_xml_validation": "pass",
        "pdf_svg_text_label_match": "pass",
        "svg_text_labels_checked": len(svg_labels),
        "cross_format_size_check": "pass; backend tight-bbox tolerance 2 pt and 0.5%",
        "max_svg_pdf_dimension_difference_points": svg_pdf_error,
        "max_png_pdf_dimension_difference_points": png_pdf_error,
    }


def save(fig, stem, title, caption, data, sources):
    staging_parent = NOTE / "figure_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    assert staging_parent.stat().st_dev == OUT.stat().st_dev
    with tempfile.TemporaryDirectory(prefix=stem + "-stage-", dir=staging_parent) as tmp:
        stage = Path(tmp)
        exports = [stage / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")]
        for ext, p in zip(("png", "pdf", "svg"), exports):
            fig.savefig(p, format=ext, dpi=600, bbox_inches="tight", pad_inches=0.13)
            sync_file(p)
        preview = stage / "preview.png"
        fig.savefig(preview, format="png", dpi=160, bbox_inches="tight", pad_inches=0.13)
        sync_file(preview)
        checks = validate_exports(exports + [preview])
        csv = stage / "source.csv"
        data.to_csv(csv, index=False)
        sync_file(csv)
        assert len(pd.read_csv(csv)) == len(data)
        for p in exports:
            os.replace(p, OUT / p.name)
        os.replace(preview, OUT / "preview" / f"{stem}.png")
        os.replace(csv, OUT / "data" / f"{stem}.csv")
    plt.close(fig)
    checks["sha256"] = {ext: sha(OUT / f"{stem}.{ext}") for ext in ("png", "pdf", "svg")}
    CHECKS.setdefault("export_integrity", {})[stem] = checks
    MANIFEST.append(
        {
            "figure": stem,
            "title": title,
            "rows": len(data),
            "sources": [source(p) for p in sources],
            "aggregate_csv_sha256": sha(OUT / "data" / f"{stem}.csv"),
        }
    )
    CAPTIONS.append(f"## {stem}\n\n**{title}.** {caption}\n")


def verify_archived_sources():
    provenance = json.loads((ORIG / "figures_final/source_provenance.json").read_text())
    checks = []
    for oldpath, expected in provenance["aggregate_source_sha256"].items():
        p = OLD / Path(oldpath).name
        checks.append(
            {
                "file": str(p.relative_to(ROOT)),
                "expected": expected,
                "actual": sha(p),
                "match": expected == sha(p),
            }
        )
    CHECKS["archived_source_sha256"] = checks
    assert all((x["match"] for x in checks)), "Archived source CSV hash mismatch"


def workflow():
    sample = pd.read_csv(TAB / "sample_reconstruction.csv").set_index("sample")
    overlap_table = pd.read_csv(
        ROOT / "revision/results/overlap_sample_proportions.csv"
    ).set_index("population")
    restricted = overlap_table.loc["restricted"]
    counts = tuple(
        (
            int(value)
            for value in restricted[["development_n", "evaluation_n", "shared_n", "disjoint_n"]]
        )
    )
    if counts != (1565, 1520, 641, 879):
        raise ValueError("Unexpected restricted cohort counts")
    if (
        int(sample.loc["development", "n"]) != counts[0]
        or int(sample.loc["evaluation", "n"]) != counts[1]
    ):
        raise ValueError("Sample reconstruction and overlap aggregates disagree")
    fig, ax = plt.subplots(figsize=(8.3, 7.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.8)
    ax.axis("off")
    nodes = []

    def box(key, x, y, w, h, title, body, fill="#F5F7F8"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.06",
                fc=fill,
                ec="#A8B1B8",
                lw=0.8,
            )
        )
        ax.text(x + w / 2, y + h - 0.19, title, ha="center", va="top", weight="bold", fontsize=11)
        ax.text(
            x + w / 2, y + h - 0.53, body, ha="center", va="top", fontsize=9.5, linespacing=1.4
        )
        nodes.append({"node": key, "title": title, "body": body.replace("\n", "; ")})

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=GRAY, lw=1.1
            )
        )

    box(
        "waves",
        0.25,
        8.52,
        9.5,
        1.02,
        "Three annual KISDI survey waves",
        "2023 baseline → 2024 outcome / baseline → 2025 outcome\nBaseline privacy hesitation and next-wave linkage define the restricted cohorts",
    )
    box(
        "development",
        0.25,
        6.74,
        4.45,
        1.18,
        "Development · 2023 → 2024",
        "1,565 respondents; 880 events\nH + seven candidate information blocks",
        "#EEF4F8",
    )
    box(
        "evaluation",
        5.3,
        6.74,
        4.45,
        1.18,
        "Temporal evaluation · 2024 → 2025",
        "1,520 respondents; 853 events\n641 shared; 879 person-disjoint",
        "#FAF1EB",
    )
    arrow(2.48, 8.52, 2.48, 7.92)
    arrow(7.52, 8.52, 7.52, 7.92)
    box(
        "selection",
        0.25,
        4.62,
        4.45,
        1.5,
        "Select using development data",
        "Ridge / HGB; item budgets 0–30\nFive-fold tuning and portfolio selection\nOriginal nested CV: 2 × 5 outer folds",
        "#EEF4F8",
    )
    box(
        "frozen",
        5.3,
        4.62,
        4.45,
        1.5,
        "Freeze, refit, and evaluate",
        "Fit selected configurations on all development data\nAssess whole and person-disjoint samples\nKeep the full-information reference explicit",
        "#FAF1EB",
    )
    arrow(2.48, 6.74, 2.48, 6.12)
    arrow(7.52, 6.74, 7.52, 6.12)
    arrow(4.7, 5.38, 5.3, 5.38)
    box(
        "contrasts",
        0.25,
        2.66,
        9.5,
        1.36,
        "Two paired comparisons answer different questions",
        "Incremental value: selected model versus P0 (H available) or NONE (H unavailable)\nPreservation: selected model versus the full-information model\nPositive model-minus-reference log loss indicates worse prediction",
    )
    arrow(7.52, 4.62, 7.52, 4.02)
    box(
        "uncertainty",
        0.25,
        0.12,
        4.45,
        1.97,
        "Uncertainty and measurement checks",
        "Paired fixed-fit bootstrap; full-process reselection\nCalibration and within-period holdouts\nAdjacent-service response diagnostics\nPost hoc capital-region recoding",
    )
    box(
        "sensitivity",
        5.3,
        0.12,
        4.45,
        1.97,
        "Added sensitivity analyses",
        "Wider Ridge/HGB searches; random forest; extra trees\nXGBoost, LightGBM, and CatBoost\nFull linked panel; weights; H+D already held\nBlock contributions, total and per item",
    )
    arrow(2.48, 2.66, 2.48, 2.09)
    arrow(7.52, 2.66, 7.52, 2.09)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(
        fig,
        "Round2_Figure_01_Workflow",
        "Research design and validation workflow",
        "Restricted development and temporal cohorts are reconstructed from the 2023–2025 source workbooks. H denotes prior self-reported service use. H availability does not force its inclusion; the available-H search permits H-present and H-absent portfolios as well as P0. P0 is an unpenalized H-only benchmark, whereas NONE predicts development prevalence. Original nested cross-validation is retained from the submitted analysis. Temporal models are newly refitted using the frozen development specifications. Added analyses use the same survey waves and are retrospective sensitivity analyses, not a new external validation cohort. The 879 person-disjoint evaluation participants are a subset of the 1,520 temporal respondents.",
        pd.DataFrame(nodes),
        [
            TAB / "sample_reconstruction.csv",
            ROOT / "revision/results/overlap_sample_proportions.csv",
            ROOT / "revision/audit/refit_reproduction_report.json",
            ORIG / "01_required_prior_analysis/revision_20260907/budget_policy_protocol.md",
        ],
    )


def frontier():
    nested = pd.read_csv(OLD / "nested_budget_summary.csv")
    old = pd.read_csv(OLD / "temporal_budget_policy_metrics.csv").query("evaluation_domain=='all'")
    new = pd.read_csv(TAB / "selected_budget_policy_results.csv").query("domain=='whole'")
    q = nested.merge(
        new[["policy_id", "source_model", "log_loss", "benchmark", "benchmark_log_loss"]],
        on="policy_id",
        validate="one_to_one",
    )
    agreement = old[["policy_id", "log_loss"]].merge(
        new[["policy_id", "log_loss"]], on="policy_id", suffixes=("_old", "_new")
    )
    CHECKS["frontier_original_vs_refit_max_abs_error"] = float(
        abs(agreement.log_loss_old - agreement.log_loss_new).max()
    )
    assert len(q) == 124 and CHECKS["frontier_original_vs_refit_max_abs_error"] < 1e-12
    predictions = np.load(ROOT / "private_data/refit_all_predictions.npz")
    ev = pd.read_csv(ROOT / "private_data/evaluation.csv", usecols=["outcome_any_use"])
    y = ev.outcome_any_use.to_numpy()
    keys = sorted(q.source_model.unique())
    p = np.column_stack([predictions[k] for k in keys])
    p = np.clip(p, 1e-15, 1 - 1e-15)
    losses = -(y[:, None] * np.log(p) + (1 - y[:, None]) * np.log1p(-p))
    rng = np.random.default_rng(2026102002)
    boot = []
    for _ in range(50):
        w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=100)
        boot.append(w @ losses / len(y))
    bounds = np.quantile(np.vstack(boot), [0.025, 0.975], axis=0)
    lookup = {k: (losses[:, i].mean(), bounds[0, i], bounds[1, i]) for i, k in enumerate(keys)}
    q["temporal_lower95"] = [lookup[k][1] for k in q.source_model]
    q["temporal_upper95"] = [lookup[k][2] for k in q.source_model]
    assert max((abs(lookup[r.source_model][0] - r.log_loss) for r in q.itertuples())) < 1e-12
    q["temporal_interval"] = (
        "pointwise percentile 95%; fixed predictions; 5000 participant resamples"
    )
    q["nested_range"] = (
        "minimum–maximum of two original nested-CV repetition losses; not a confidence interval"
    )
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 7.0), sharex=True, sharey=True)
    for i, a in enumerate(["ridge", "hgb"]):
        for j, h in enumerate(["history_available", "history_unavailable"]):
            s = q[(q.algorithm == a) & (q.availability == h)].sort_values("budget")
            ax = axs[i, j]
            ax.fill_between(
                s.budget,
                s.temporal_lower95,
                s.temporal_upper95,
                color=BLUE,
                alpha=0.13,
                step="post",
                lw=0,
            )
            ax.step(s.budget, s.log_loss, where="post", color=BLUE, lw=1.8)
            ax.fill_between(
                s.budget,
                s.nested_log_loss_repeat_min,
                s.nested_log_loss_repeat_max,
                color=ORANGE,
                alpha=0.22,
                lw=0,
            )
            ax.plot(
                s.budget,
                s.nested_log_loss_mean,
                color=ORANGE,
                lw=1.8,
                marker="s",
                markersize=2.5,
                markevery=[0, 1, 4, 15, 30],
            )
            ax.axhline(s.benchmark_log_loss.iloc[0], color=GRAY, ls="--", lw=1.1)
            ax.set_title(
                ("Ridge" if a == "ridge" else "HGB")
                + (" · H available" if h == "history_available" else " · H unavailable")
            )
            ax.set_xlim(0, 30)
            ax.set_ylim(0.61, max(0.76, float(q.temporal_upper95.max()) + 0.008))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(0.02))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(axis="y")
            if j == 0:
                ax.set_ylabel("Log loss (lower is better)")
            if i == 1:
                ax.set_xlabel("Additional questionnaire-item budget")
    handles = [
        Line2D(
            [0], [0], color=ORANGE, lw=1.8, marker="s", markersize=4, label="Original nested CV"
        ),
        Line2D([0], [0], color=BLUE, lw=1.8, label="Temporal evaluation"),
        Line2D([0], [0], color=GRAY, lw=1.1, ls="--", label="Temporal P0 / NONE"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.99), fontsize=10
    )
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.085, right=0.99, hspace=0.27, wspace=0.12)
    save(
        fig,
        "Round2_Figure_02_Development_Temporal",
        "Development cross-validation and temporal log loss",
        "Orange curves retain the original mean log loss from two repetitions of five-fold outer cross-validation, with five-fold tuning and portfolio selection inside each outer training set (development n=1,565). The narrow orange bands span the two repetition losses and are descriptive ranges, not confidence intervals. Blue curves evaluate the development-selected frozen policies after refitting on all development data (temporal n=1,520). Blue bands are pointwise 95% percentile intervals from 5,000 evaluation-participant bootstrap resamples, holding fitted predictions fixed (seed 2026102002); they are not simultaneous frontier intervals or intervals for a development–temporal difference. The dashed line is the temporal P0 or NONE benchmark. Original temporal losses were reproduced from the raw-data refits to numerical precision. Changes across periods combine time, survey wording, sample composition, and different model training sizes. Selection OOF scores, which reuse tuning folds and can be optimistic, are retained in the accompanying CSV but are not the orange curve.",
        q,
        [
            OLD / "nested_budget_summary.csv",
            OLD / "temporal_budget_policy_metrics.csv",
            TAB / "selected_budget_policy_results.csv",
            ROOT / "revision/audit/refit_reproduction_report.json",
        ],
    )
    CHECKS["frontier_temporal_interval_reps"] = 5000
    CHECKS["frontier_temporal_interval_seed"] = 2026102002


def importance():
    all_rows = pd.read_csv(TAB / "attribution_per_item.csv")
    q = all_rows[all_rows.domain.eq("person_disjoint")].copy()
    fig, axs = plt.subplots(2, 2, figsize=(9.8, 7.6), sharex="col")
    for i, h in enumerate([True, False]):
        for j, per_item in enumerate([False, True]):
            ax = axs[i, j]
            val = "loss_reduction_per_item" if per_item else "loss_reduction"
            lo = "per_item_simultaneous95_lower" if per_item else "simultaneous95_lower"
            hi = "per_item_simultaneous95_upper" if per_item else "simultaneous95_upper"
            for a, offset, color, marker in [
                ("ridge", -0.14, BLUE, "o"),
                ("hgb", 0.14, ORANGE, "s"),
            ]:
                z = q[(q.algorithm == a) & (q.history_fixed == h)].set_index("block").loc[BLOCKS]
                v = z[val].to_numpy()
                l = z[lo].to_numpy()
                u = z[hi].to_numpy()
                assert np.all(l <= v) and np.all(v <= u)
                ax.errorbar(
                    v,
                    np.arange(7) + offset,
                    xerr=np.vstack([v - l, u - v]),
                    fmt=marker,
                    color=color,
                    lw=1.25,
                    capsize=2.5,
                    markersize=4.3,
                )
            ax.axvline(0, color=GRAY, lw=1, ls="--")
            ax.grid(axis="x")
            ax.set_yticks(range(7), [f"{LABELS[b]} ({COST[b]})" for b in BLOCKS])
            ax.set_ylim(6.6, -0.6)
            ax.set_title(
                ("Total allocation" if not per_item else "Per-item allocation")
                + (" · H present" if h else " · H absent")
            )
            if i == 1:
                ax.set_xlabel(
                    "Log-loss reduction per item" if per_item else "Total log-loss reduction"
                )
            ax.xaxis.set_major_locator(MultipleLocator(0.02 if per_item else 0.04))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            lo_all = all_rows[lo].min()
            hi_all = all_rows[hi].max()
            span = hi_all - lo_all
            ax.set_xlim(lo_all - 0.04 * span, hi_all + 0.04 * span)
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=1.2, label="Ridge"),
        Line2D([0], [0], color=ORANGE, marker="s", lw=1.2, label="HGB"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, fontsize=10.5
    )
    fig.subplots_adjust(top=0.91, bottom=0.075, left=0.2, right=0.99, wspace=0.82, hspace=0.25)
    save(
        fig,
        "Round2_Figure_03_Person_Disjoint_Importance",
        "Person-disjoint block contributions, total and per item",
        "Exact seven-block Shapley allocations of held-out log-loss reduction in the 879 person-disjoint temporal respondents are shown for ridge and HGB, with H fixed present (top) or absent (bottom). Positive values mean better prediction. Left: total block allocation. Right: allocation and interval divided by the original item count, shown in parentheses. Both rows share the same scale within each column; the total and per-item columns have different units and axis ranges. Bars are two-sided simultaneous 95% maximum-absolute centered-bootstrap intervals across all 56 block × algorithm × history × evaluation-group contrasts, not a smaller family restricted to this figure. The 5,000 resamples condition on fitted predictions and jointly resample the whole temporal cohort, with each replicate’s person-disjoint mean based on its resampled nonoverlapping members. The H-present coalition reference is the learner-specific H-only model, not P0. Per-item normalization is a descriptive efficiency measure, not a cost-weighted Shapley game or evidence about a shortened scale. These are block-level predictive allocations, not causal effects or individual-feature SHAP values.",
        q,
        [
            TAB / "attribution_per_item.csv",
            ROOT / "revision/audit/attribution_audit.json",
            ROOT / "revision/code/run_attribution.py",
        ],
    )


def verify_development_contribution():
    dev = pd.read_csv(
        OLD / "block_attribution_development_descriptive.csv", float_precision="round_trip"
    )
    old_temporal = pd.read_csv(
        OLD / "block_attribution_evaluation.csv", float_precision="round_trip"
    )
    now = pd.read_csv(TAB / "attribution_per_item.csv", float_precision="round_trip")
    subset = pd.read_csv(
        ORIG / "01_required_prior_analysis/analysis_outputs/development_all_subsets.csv",
        float_precision="round_trip",
    )
    rows = []
    for a in ["ridge", "hgb"]:
        lookup = subset[subset.algorithm == a].set_index("subset").oof_log_loss
        for h in [True, False]:
            ll = []
            for m in range(128):
                bs = (["H"] if h else []) + [b for j, b in enumerate(BLOCKS) if m & 1 << j]
                ll.append(float(lookup["_".join(bs) if bs else "NONE"]))
            for j, b in enumerate(BLOCKS):
                value = sum(
                    (
                        math.factorial(m.bit_count())
                        * math.factorial(6 - m.bit_count())
                        / math.factorial(7)
                        * (ll[m] - ll[m | 1 << j])
                        for m in range(128)
                        if not m & 1 << j
                    )
                )
                archival = float(
                    dev[
                        (dev.algorithm == a) & (dev.has_history == h) & (dev.block == b)
                    ].shapley_log_loss_reduction.iloc[0]
                )
                assert abs(value - archival) < 1e-12
                for domain in ["whole", "person_disjoint"]:
                    r = now[
                        (now.algorithm == a)
                        & (now.history_fixed == h)
                        & (now.block == b)
                        & (now.domain == domain)
                    ].iloc[0]
                    previous = old_temporal[
                        (old_temporal.algorithm == a)
                        & (old_temporal.has_history == h)
                        & (old_temporal.block == b)
                        & (old_temporal.domain == domain)
                    ].shapley_log_loss_reduction.iloc[0]
                    assert abs(previous - r.loss_reduction) < 1e-12
                    rows.append(
                        {
                            "algorithm": a,
                            "history_fixed": h,
                            "block": b,
                            "domain": domain,
                            "items": COST[b],
                            "development_oof_archived": archival,
                            "development_recomputed_from_subset_means": value,
                            "temporal_recomputed_raw_refits": r.loss_reduction,
                            "temporal_minus_development": r.loss_reduction - archival,
                            "temporal_simultaneous95_lower": r.simultaneous95_lower,
                            "temporal_simultaneous95_upper": r.simultaneous95_upper,
                            "point_sign_reversal": value * r.loss_reduction < 0,
                            "development_status": "descriptive; tuning folds reused; no development CI",
                            "reference": a + "::H" if h else a + "::NONE",
                        }
                    )
    z = pd.DataFrame(rows)
    atomic_text(TAB / "round2_development_temporal_block_attribution.csv", z.to_csv(index=False))
    CHECKS["development_attribution_max_abs_reproduction_error"] = float(
        abs(z.development_oof_archived - z.development_recomputed_from_subset_means).max()
    )
    CHECKS["temporal_attribution_original_reproduction"] = (
        "All 56 estimates match archived attribution within 1e-12."
    )
    r = z[
        (z.algorithm == "ridge") & z.history_fixed & (z.block == "D") & (z.domain == "whole")
    ].iloc[0]
    ds = pd.read_csv(OLD / "block_attribution_development_descriptive.csv", dtype=str)
    ts = pd.read_csv(TAB / "attribution_per_item.csv", dtype=str)
    original_decimal = ds[
        (ds.algorithm == "ridge") & (ds.has_history == "True") & (ds.block == "D")
    ].shapley_log_loss_reduction.iloc[0]
    temporal_decimal = ts[
        (ts.algorithm == "ridge")
        & (ts.history_fixed == "True")
        & (ts.block == "D")
        & (ts.domain == "whole")
    ].loss_reduction.iloc[0]
    difference_decimal = str(Decimal(temporal_decimal) - Decimal(original_decimal))
    return f"## Verification of the demographic reversal\n\nThe reviewer’s +0.0176 versus −0.0416 compares the same seven-block exact Shapley definition under ridge with H fixed present. For a block j, the allocation averages LL(S) − LL(S ∪ {{j}}) over all 64 coalitions excluding j with weight |S|!(6−|S|)!/7!. H is present in every coalition, so the empty optional-block reference is ridge::H, not the unpenalized P0 benchmark. This is the total four-item D allocation, not LL(H)−LL(H+D) and not a per-item value.\n\n- Original development OOF allocation: **{original_decimal}**.\n- Independently recomputed from all 128 archived subset mean OOF losses: **{r.development_recomputed_from_subset_means:+.17f}**.\n- Temporal allocation from the fresh raw-data refits: **{temporal_decimal}**.\n- Temporal minus development point allocation: **{difference_decimal}**.\n- Temporal simultaneous 95% interval: **[{r.temporal_simultaneous95_lower:+.8f}, {r.temporal_simultaneous95_upper:+.8f}]**.\n\nThe algebraic allocation, block costs, encoding specification and coalition set are compatible across phases. The estimates are not equivalent validation quantities: development OOF predictions reused the folds that selected each subset’s hyperparameters and therefore can be optimistic; temporal predictions use models fitted on the full development cohort. No development interval or paired interval for the difference is available here. Report a descriptive point-estimate sign reversal, not a statistically tested change, measurement-invariant transfer effect, or causal concept drift. All 28 original development allocations were independently reconstructed from archived subset-level means; all 56 temporal allocations match the original aggregate estimates within 1e−12. The new temporal intervals use 5,000 resamples and therefore need not equal the archived 2,000-resample endpoints.\n\nFull verification table: `revision/tables/round2_development_temporal_block_attribution.csv`.\n"


def main():
    parser = argparse.ArgumentParser(description="Render aggregate research figures.")
    parser.add_argument("--workflow-only", action="store_true")
    args = parser.parse_args()
    for p in [OUT, OUT / "data", OUT / "preview", NOTE]:
        p.mkdir(parents=True, exist_ok=True)
    if args.workflow_only:
        workflow()
        atomic_text(OUT / "workflow_figure_manifest.json", json.dumps(MANIFEST, indent=2))
        atomic_text(
            ROOT / "revision/audit/workflow_figure_validation.json", json.dumps(CHECKS, indent=2)
        )
        atomic_text(OUT / "Round2_Figure_01_Workflow.txt", "\n".join(CAPTIONS))
        print(
            json.dumps(
                {"figures": [entry["figure"] for entry in MANIFEST], "checks": CHECKS}, indent=2
            )
        )
        return
    verify_archived_sources()
    reversal = verify_development_contribution()
    workflow()
    frontier()
    importance()
    note = "# Round 2 figure evidence and provenance\n\n"
    note += "These figures address Reviewer 1’s request to retain the workflow and development-versus-temporal comparison, show feature importance across evaluation datasets, and display uncertainty consistently; they also support Reviewer 2’s demographic-reversal and per-item-attribution requests. They use the original submitted aggregates and the executed raw-data reconstruction. No model fitting is rerun.\n\n"
    note += "## Suggested document mapping\n\nMain Figure 1: Round2_Figure_01_Workflow. Main Figure 2: Round2_Figure_02_Development_Temporal. Supplementary Figure S4: Round2_Figure_03_Person_Disjoint_Importance. Four panels show total and per-item allocations, with H present and absent; ridge and HGB use offset markers. Both history conditions share the same axis range within each normalization. Main Figure 7 is generated separately by plot_submission_block_heatmap.py and combines development, whole-cohort temporal, and person-disjoint temporal per-item allocations; Table S21 provides temporal intervals. All have 600 dpi PNG, vector PDF/SVG, an aggregate source CSV, and a preview.\n\n"
    note += '## Trace of the current Figure 5\n\nThe Step 2 document builder maps manuscript Figure 5 to `JTAER_Figures/Figure_03`. Its `contribution()` function in `revision/code/plot_revision_figures.py` filters `domain == "whole"`; it displays only the whole-sample per-item allocation. Although its intervals belong to the joint 56-contrast family, that figure does not display person-disjoint estimates or total block allocations. The new four-panel supplementary figure supplies the missing person-disjoint domain and both normalizations without changing that original file. Its bounds retain exactly the same 56-contrast family as the whole-sample figure, rather than recalibrating to a smaller displayed family.\n\n'
    note += reversal + "\n" + "\n".join(CAPTIONS)
    note += "\n## Numerical and visual provenance\n\nThe five original CSV sources match the SHA-256 values retained in the original figure provenance record. The 124 temporal frontier losses match the fresh raw-data refit results within 1e−12. Recomputed development Shapley allocations use the algebraic linearity of the original exact allocation; they are a reconstruction from stored fitted-model OOF means, not a new fit or nested development attribution. Figure source CSVs contain aggregate outputs only. Bootstrap prediction arrays and participant identifiers remain private.\n\nFull-process reselection uncertainty is reported separately from the frontier bootstrap. The frontier bootstrap uses 5,000 fixed-prediction evaluation resamples, seed 2026102002, pointwise absolute-loss intervals.\n"
    atomic_text(NOTE / "figures.md", note)
    atomic_text(OUT / "round2_figure_manifest.json", json.dumps(MANIFEST, indent=2))
    atomic_text(
        ROOT / "revision/audit/round2_figure_validation.json", json.dumps(CHECKS, indent=2)
    )
    print(json.dumps({"figures": [m["figure"] for m in MANIFEST], "checks": CHECKS}, indent=2))


if __name__ == "__main__":
    main()
