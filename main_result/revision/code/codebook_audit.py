from pathlib import Path
import json, re, hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "revision/tables"
A = ROOT / "revision/audit"
rows = []
manifest = []
aliases = {
    2023: {"region": "지역", "age_group": "연령대", "gender": "SQ1", "education": "DQ2_1"},
    2024: {"region": "Area1", "age_group": "Ager_group", "gender": "SQ1", "education": "DQ2_1"},
    2025: {"region": "Area1", "age_group": "Ager_group", "gender": "SQ1", "education": "DQ2_1"},
}
rawnames = {"region": "Area1", "age_group": "Age_group", "gender": "SQ1", "education": "DQ2_1"}
out = []
for y in (2023, 2024, 2025):
    p = ROOT / "private_raw" / f"{y}_codebook.xlsx"
    xl = pd.ExcelFile(p)
    manifest.append(
        {
            "year": y,
            "file": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
    )
    for sn in xl.sheet_names:
        if sn == "Variable Lables" or sn == "코드북 가이드":
            continue
        d = pd.read_excel(p, sheet_name=sn, header=None)
        if d.shape[1] < 5:
            continue
        d[1] = d[1].ffill()
        d[2] = d[2].ffill()
        for var, alias in aliases[y].items():
            for idx, r in d[d[1].eq(alias)].iterrows():
                code = pd.to_numeric(r[3], errors="coerce")
                if pd.notna(code):
                    rows.append(
                        {
                            "year": y,
                            "variable": var,
                            "raw_variable": rawnames[var],
                            "codebook_variable": alias,
                            "code": int(code),
                            "value_label_ko": str(r[4]).strip(),
                            "sheet": sn,
                            "excel_row": idx + 1,
                        }
                    )
        for idx, r in d[d[1].astype(str).str.lower().eq("q1_8")].iterrows():
            code = pd.to_numeric(r[3], errors="coerce")
            if pd.notna(code):
                out.append(
                    {
                        "year": y,
                        "outcome_variable": str(r[1]),
                        "question_label_ko": str(r[2]),
                        "code": int(code),
                        "value_label_ko": str(r[4]).strip(),
                        "sheet": sn,
                        "excel_row": idx + 1,
                    }
                )
df = pd.DataFrame(rows)
df.to_csv(T / "demographic_codebook_mapping.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(out).to_csv(T / "outcome_codebook_wording.csv", index=False, encoding="utf-8-sig")
checks = []
for var in aliases[2023]:
    ref = df[(df.year == 2023) & (df.variable == var)].set_index("code").value_label_ko.to_dict()
    for y in (2024, 2025):
        other = (
            df[(df.year == y) & (df.variable == var)].set_index("code").value_label_ko.to_dict()
        )
        checks.append(
            {
                "variable": var,
                "reference_year": 2023,
                "comparison_year": y,
                "code_and_label_identity": ref == other,
                "n_categories_2023": len(ref),
                "n_categories_comparison": len(other),
                "notes": (
                    "Age_group in Numeric sheets; Ager_group is the codebook alias in 2024 and 2025. Code 7 is allowed by historical harmonization but absent from published demographic age labels."
                    if var == "age_group"
                    else ""
                ),
            }
        )
pd.DataFrame(checks).to_csv(
    T / "demographic_coding_identity_check.csv", index=False, encoding="utf-8-sig"
)
(A / "codebook_source_manifest.json").write_text(
    json.dumps(
        {
            "sources": manifest,
            "coding_checks": checks,
            "scope": "Coding identity concerns labels, not stability of respondents or conditional predictive relationships.",
        },
        indent=2,
        ensure_ascii=False,
    )
)
print(pd.DataFrame(checks).to_string(index=False))
print(pd.DataFrame(out).to_string(index=False))
embedded = []
for y in (2023, 2024, 2025):
    d = pd.read_excel(ROOT / "private_raw" / f"{y}_data.xlsx", sheet_name="Codebook")
    d.iloc[:, 0] = d.iloc[:, 0].ffill()
    for var, rawvar in rawnames.items():
        for idx, r in d[d.iloc[:, 0].eq(rawvar)].iterrows():
            code = pd.to_numeric(r.iloc[1], errors="coerce")
            if pd.notna(code):
                embedded.append(
                    {
                        "year": y,
                        "variable": var,
                        "raw_variable": rawvar,
                        "code": int(code),
                        "value_label_ko": str(r.iloc[2]).strip(),
                        "source_sheet": "Codebook",
                        "excel_row": idx + 2,
                    }
                )
e = pd.DataFrame(embedded)
e.to_csv(T / "demographic_raw_embedded_codebook_mapping.csv", index=False, encoding="utf-8-sig")
checks = []
for var in rawnames:
    ref = e[(e.year == 2023) & (e.variable == var)].set_index("code").value_label_ko.to_dict()
    for y in (2024, 2025):
        comp = e[(e.year == y) & (e.variable == var)].set_index("code").value_label_ko.to_dict()
        checks.append(
            {
                "variable": var,
                "reference_year": 2023,
                "comparison_year": y,
                "code_and_label_identity": ref == comp,
                "n_categories_2023": len(ref),
                "n_categories_comparison": len(comp),
                "authoritative_source": "Embedded Codebook sheet in each raw XLSX",
                "note": (
                    "Standalone codebooks omit category7; embedded codebooks consistently define7 as70s. No age7 observations in linked development/evaluation predictors."
                    if var == "age_group"
                    else ""
                ),
            }
        )
pd.DataFrame(checks).to_csv(
    T / "demographic_coding_identity_check.csv", index=False, encoding="utf-8-sig"
)
(A / "codebook_source_manifest.json").write_text(
    json.dumps(
        {
            "sources": manifest,
            "coding_checks": checks,
            "scope": "Embedded raw XLSX Codebook sheets verify 2 gender,7 age,8 education,and17 region categories with identical code meanings. Standalone codebooks omit age7 but embedded sources label it 70s. Label identity does not establish stability of conditional predictive relationships.",
        },
        indent=2,
        ensure_ascii=False,
    )
)
print("EMBEDDED CODING COMPLETE", all((x["code_and_label_identity"] for x in checks)))
