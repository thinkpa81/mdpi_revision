# Figure 7 panel lettering

Panel names were changed from A–D to (a)–(d) to distinguish them from the questionnaire-block labels. The numerical source CSV is byte-identical to the archived 84-cell source at commit `af06ecda65c0f5768cbe2b3449bb46ebf229edca`. The image dimensions, colors, values, axes and significance outlines are unchanged. A pixel comparison confirms that every changed pixel lies within the two panel-title bands.

From the addendum root, run:

```bash
python code/regenerate_figure7.py
python code/verify_figure7.py
```

The regeneration script reads the aggregate CSV inputs in `figures/data/` and writes PNG at 600 dpi, PDF, SVG, a preview, a source CSV and validation metadata. It does not access participant data or refit models. The verification script also uses `figures/source_figure7.png`, extracted unchanged from the previous manuscript, to check the pixel-level scope of the relabeling.

The seven outlined temporal cells remain six for D and one for A. The sole outlined A cell is HGB with H fixed in the person-disjoint subset.

Runtime: Matplotlib 3.10.8, pandas 2.2.3 and NumPy 2.3.5. Figure inspection confirmed readable panel labels and no clipping or overlaps.
