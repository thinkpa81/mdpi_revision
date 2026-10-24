# Figures and aggregate plot data

**Questionnaire Item Budgets and Incremental Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Temporal Evidence from a Korean Panel**

This directory contains the 7 main figures and 4 supplementary figures used in the final 24 October 2026 documents. Each figure is provided as a 600 dpi PNG, vector PDF, vector SVG, and aggregate source CSV. All 11 PNGs are byte-identical to their embedded document images. The document filenames and SHA-256 hashes in the manifest identify the clean files from the latest editorial revision; the documents themselves are not distributed here. The document and response checks are summarized in [EDITORIAL_ALIGNMENT_CHECK_20261024.json](../EDITORIAL_ALIGNMENT_CHECK_20261024.json).

- `Main/`: Figures 1–7.
- `Supplementary/`: Figures S1–S4.
- `data/`: aggregate plot data and the workflow text for Figure 1.
- `FIGURE_CROSSWALK.csv`: publication numbers, analytical source names, released data paths, and final figure captions.
- `FIGURE_MANIFEST.json`: final-document identities, image properties, captions, and figure/data checksums.
- `SHA256SUMS.json`: checksums of all other files in this directory.

The `analytical_source_csv` column describes the corresponding path relative to an analytical workspace rooted at `main_result/`. Some such files are generated during reproduction. The `data_csv` column identifies the available publication-numbered CSV relative to this directory. Analytical plotting scripts write to `main_result/JTAER_Figures/`; this top-level directory holds the checked publication exports.

Figure 7 contains 84 development and temporal block-contribution cells. Development values are descriptive; the temporal intervals use the stated simultaneous family. Figure 4 uses 1,000 complete-procedure bootstrap repetitions per setting. The latest editorial alignment updates document identities and verification metadata. All 11 captions, plot data, and image bytes are unchanged. An earlier release removed the redundant standalone editorial caption file.

