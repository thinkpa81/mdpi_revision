from pathlib import Path
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "revision" / "tables"
RESULTS = ROOT / "revision" / "results"
FIGURES = ROOT / "JTAER_Figures"
BLOCKS = ("D", "B", "M", "P", "C", "A", "K")
DOMAINS = ("development_oof", "whole", "disjoint")
PANELS = (("ridge", True), ("ridge", False), ("hgb", True), ("hgb", False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_source():
    attribution = pd.read_csv(TABLES / "round2_development_temporal_block_attribution.csv")
    reference = pd.read_csv(TABLES / "attribution_per_item.csv")
    reference = reference.set_index(
        ["algorithm", "history_fixed", "block", "domain"], verify_integrity=True
    )
    rows = []
    for (algorithm, history, block), group in attribution.groupby(
        ["algorithm", "history_fixed", "block"], sort=False
    ):
        if len(group) != 2 or np.ptp(group.development_recomputed_from_subset_means) >= 1e-12:
            raise ValueError(
                f"Inconsistent development allocation: {algorithm}, {history}, {block}"
            )
        first = group.iloc[0]
        rows.append(
            {
                "algorithm": algorithm,
                "history_fixed": bool(history),
                "block": block,
                "items": int(first["items"]),
                "domain": "development_oof",
                "loss_reduction_per_item": float(
                    first.development_recomputed_from_subset_means / first["items"]
                ),
                "simultaneous_lower_per_item": np.nan,
                "simultaneous_upper_per_item": np.nan,
                "scope": "descriptive development OOF contribution; tuning folds reused",
                "reference": first.reference,
            }
        )
        for row in group.itertuples(index=False):
            expected = reference.loc[algorithm, history, block, row.domain]
            values = np.array(
                [
                    row.temporal_recomputed_raw_refits / row.items,
                    row.temporal_simultaneous95_lower / row.items,
                    row.temporal_simultaneous95_upper / row.items,
                ]
            )
            targets = expected[
                [
                    "loss_reduction_per_item",
                    "per_item_simultaneous95_lower",
                    "per_item_simultaneous95_upper",
                ]
            ].to_numpy(dtype=float)
            if not np.allclose(values, targets, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Temporal allocation or interval mismatch: {algorithm}, {history}, {block}, {row.domain}"
                )
            rows.append(
                {
                    "algorithm": algorithm,
                    "history_fixed": bool(history),
                    "block": block,
                    "items": int(row.items),
                    "domain": "disjoint" if row.domain == "person_disjoint" else "whole",
                    "loss_reduction_per_item": float(values[0]),
                    "simultaneous_lower_per_item": float(values[1]),
                    "simultaneous_upper_per_item": float(values[2]),
                    "scope": "temporal contribution conditional on fitted subset models",
                    "reference": row.reference,
                }
            )
    source = pd.DataFrame(rows)
    if (
        len(source) != 84
        or source.duplicated(["algorithm", "history_fixed", "block", "domain"]).any()
    ):
        raise ValueError("Expected 84 unique block-by-model-by-domain cells")
    source["interval_excludes_zero"] = source.simultaneous_lower_per_item.gt(
        0
    ) | source.simultaneous_upper_per_item.lt(0)
    source["outlined"] = source.domain.ne("development_oof") & source.interval_excludes_zero
    return source


def render(source):
    limit = float(np.ceil(source.loss_reduction_per_item.abs().max() * 1000) / 1000)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.7), constrained_layout=True)
    labels = ["Development\nOOF", "Temporal\nwhole", "Temporal\ndisjoint"]
    outline_count = 0
    for index, ((algorithm, history), axis) in enumerate(zip(PANELS, axes.flat)):
        group = source[source.algorithm.eq(algorithm) & source.history_fixed.eq(history)]
        lookup = group.set_index(["block", "domain"], verify_integrity=True)
        matrix = group.pivot(
            index="block", columns="domain", values="loss_reduction_per_item"
        ).loc[list(BLOCKS), list(DOMAINS)]
        image = axis.imshow(
            matrix.to_numpy(),
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            aspect="auto",
        )
        axis.set_xticks(range(3), labels, fontsize=8.5)
        axis.set_yticks(
            range(7),
            [
                f"{block} ({int(group[group.block.eq(block)]['items'].iloc[0])})"
                for block in BLOCKS
            ],
            fontsize=9,
        )
        axis.tick_params(axis="both", length=0, pad=5)
        learner = "Ridge" if algorithm == "ridge" else "HGB"
        history_label = "fixed" if history else "unavailable"
        axis.set_title(f"{chr(65 + index)}  {learner} | H {history_label}", loc="left", pad=10)
        for y, block in enumerate(BLOCKS):
            for x, domain in enumerate(DOMAINS):
                row = lookup.loc[block, domain]
                value = float(row.loss_reduction_per_item)
                color = "white" if abs(value) > 0.55 * limit else "#111111"
                axis.text(
                    x, y, f"{value:+.4f}", ha="center", va="center", color=color, fontsize=8.6
                )
                if row.outlined:
                    axis.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="#101010",
                            linewidth=1.65,
                            clip_on=False,
                            zorder=4,
                        )
                    )
                    outline_count += 1
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar = figure.colorbar(
        image, ax=axes, location="bottom", shrink=0.88, pad=0.04, fraction=0.06, aspect=40
    )
    colorbar.set_label(
        "Log-loss reduction per questionnaire item (positive = improvement)", labelpad=6
    )
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "Black outline: temporal simultaneous 95% interval excludes zero", fontsize=9, y=1.035
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "Submission_Figure_07_Block_Heatmap"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            stem.with_suffix("." + suffix), dpi=600, facecolor="white", bbox_inches="tight"
        )
    (FIGURES / "preview").mkdir(exist_ok=True)
    figure.savefig(
        FIGURES / "preview" / (stem.name + ".png"), dpi=180, facecolor="white", bbox_inches="tight"
    )
    plt.close(figure)
    return (stem, limit, outline_count)


def main():
    source = prepare_source()
    RESULTS.mkdir(parents=True, exist_ok=True)
    source.to_csv(RESULTS / "submission_block_heatmap.csv", index=False)
    stem, limit, outline_count = render(source)
    caption = "Figure 7. Block contributions per questionnaire item across development and temporal evaluation. Each panel uses the same zero-centered color scale; positive values indicate a reduction in log loss. Rows show D demographics (4 items), B expected benefits (1), M data management (1), P safeguards (11), C consent strategy (1), A generative-AI experience (1), and K digital competence (11). Development values are descriptive out-of-fold Shapley contributions calculated from the original subset losses; the tuning folds were reused, so these estimates have no independent development-test interpretation and no intervals are shown. Temporal columns show the restricted whole cohort (n=1,520) and its person-disjoint subset (n=879), conditional on the fitted subset models. Outlined cells mark temporal allocations whose 56-contrast simultaneous 95% intervals exclude zero; all seven indicate higher loss. No positive temporal allocation has an interval excluding zero. History-fixed contributions use the fitted H-only model as their reference, which differs from the P0 lookup benchmark used in the collection-policy comparisons. Temporal simultaneous bounds and full-precision values are provided in the supplementary attribution tables and the figure source CSV."
    stem.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
    selected = source[source.outlined]
    audit = {
        "status": "passed",
        "panels": 4,
        "blocks_per_panel": 7,
        "domains": list(DOMAINS),
        "cells": 84,
        "temporal_values_and_bounds_match_per_item_source": True,
        "global_symmetric_color_limit": limit,
        "interval_family_size": 56,
        "outlined_cells": int(outline_count),
        "outlined_cells_higher_loss": int(selected.simultaneous_upper_per_item.lt(0).sum()),
        "outlined_cells_lower_loss": int(selected.simultaneous_lower_per_item.gt(0).sum()),
        "outline_detail": selected.to_dict(orient="records"),
        "source_csv": "revision/results/submission_block_heatmap.csv",
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                TABLES / "round2_development_temporal_block_attribution.csv",
                TABLES / "attribution_per_item.csv",
            ]
        },
        "script_sha256": digest(Path(__file__)),
        "image_files": {
            stem.with_suffix("." + suffix).name: digest(stem.with_suffix("." + suffix))
            for suffix in ("png", "pdf", "svg")
        },
        "visual_qa": "pending",
    }
    (RESULTS / "submission_block_heatmap_validation.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "created",
                "cells": len(source),
                "outlined_cells": outline_count,
                "figure": str(stem),
            }
        )
    )


if __name__ == "__main__":
    main()
