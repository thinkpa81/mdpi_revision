from pathlib import Path
import subprocess, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
rows = []
manifest = []
for y in (2023, 2024, 2025):
    p = ROOT / "private_raw" / f"{y}_questionnaire.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(p), "-"]).decode("utf-8")
    pages = text.split("\x0c")
    q = [(i + 1, t) for i, t in enumerate(pages) if "Q1." in t]
    assert len(q) == 1
    page, t = q[0]
    stem = next((x.strip() for x in t.splitlines() if x.strip().startswith("Q1.")))
    row = next((x.strip() for x in t.splitlines() if x.strip().startswith("8) 소비")))
    assert "소비(쇼핑 어드바이저, 온라인 쇼핑 등)" in row
    rows.append(
        {
            "year": y,
            "aspect": "Q1 question stem",
            "pdf_page": page,
            "printed_page": 3,
            "text_ko": stem,
            "interpretation_en": "2025 adds an explicit frequency expression (how often); 2023 and 2024 use how much.",
        }
    )
    rows.append(
        {
            "year": y,
            "aspect": "Consumer-service category",
            "pdf_page": page,
            "printed_page": 3,
            "text_ko": row[: row.find("①")].strip(),
            "interpretation_en": "The consumer-service category label is unchanged.",
        }
    )
    additional = [(i + 1, t) for i, t in enumerate(pages) if "8) 지능형(인공지능) 소비서비스" in t]
    rows.append(
        {
            "year": y,
            "aspect": "Detailed consumption-service explanation",
            "pdf_page": additional[0][0] if additional else None,
            "printed_page": 9 if additional else None,
            "text_ko": (
                "개인 맞춤형 상품 추천, 가격 비교, 할인 정보, 쇼핑 타이밍, 제품 사용 방법; 네이버플러스 스토어 및 컬리 장바구니 추천 예시"
                if additional
                else "별도 상세 설명 확인되지 않음"
            ),
            "interpretation_en": (
                "Additional explanation and named examples appear in the 2025 questionnaire. Their administration and causal effects are not identifiable from response records."
                if additional
                else "No corresponding separate extended consumption-service explanation was found."
            ),
        }
    )
    manifest.append(
        {
            "year": y,
            "file": p.name,
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "pdf_pages": len(pages) - 1,
        }
    )
pd.DataFrame(rows).to_csv(
    T / "questionnaire_comparability_audit.csv", index=False, encoding="utf-8-sig"
)
p = T / "paired_outcome_change_diagnostics.csv"
d = pd.read_csv(p)
rng = np.random.default_rng(20261020)
for i, r in d.iterrows():
    n = int(r.paired_n)
    plus = int(r.no_to_yes)
    minus = int(r.yes_to_no)
    draws = rng.multinomial(n, [plus / n, minus / n, 1 - (plus + minus) / n], size=10000)
    z = (draws[:, 0] - draws[:, 1]) / n
    d.loc[i, "change_percentile95_lower"] = np.quantile(z, 0.025)
    d.loc[i, "change_percentile95_upper"] = np.quantile(z, 0.975)
order = np.argsort(d.mcnemar_exact_two_sided_p.to_numpy())
adj = np.empty(len(d))
cum = 0
for rank, ix in enumerate(order):
    cum = max(cum, min(1, float(d.loc[ix, "mcnemar_exact_two_sided_p"]) * (len(d) - rank)))
    adj[ix] = cum
d["mcnemar_holm_p_three_thresholds"] = adj
d.to_csv(p, index=False, encoding="utf-8-sig")
(A / "questionnaire_source_manifest.json").write_text(
    json.dumps(
        {
            "sources": manifest,
            "paired_bootstrap_replicates": 10000,
            "paired_bootstrap_seed": 20261020,
            "measurement_identifiability": "The available single-item, single-instrument-per-wave design does not identify measurement invariance, wording effects, genuine behavior change, or their separate contributions. Response-shift tests and wording audits characterize the threat; they do not remove it.",
        },
        indent=2,
        ensure_ascii=False,
    )
)
print(
    d[
        [
            "outcome",
            "change_2025_minus_2024",
            "change_percentile95_lower",
            "change_percentile95_upper",
            "mcnemar_holm_p_three_thresholds",
        ]
    ].to_string(index=False)
)
