# Replication Package

**Public repository:** https://github.com/1138034910-a11y/RPS-

**Paper:** When renewable mandates meet heating lock-in: evidence from China's wind curtailment rebound

**Authors:** Haoshuang Cheng (corresponding), Leiming Li — China University of Petroleum (East China)

**Contact:** Haoshuang Cheng, Z25080015@s.upc.edu.cn

Data and code supporting the findings of the study. The panel is assembled from publicly available official sources (National Energy Administration, China Electricity Council, and NDRC/NEA policy documents).

## Structure

```
replication_package/
├── data/
│   ├── master_panel_v8d.csv            # Main annual provincial panel (31 provinces, 2015–2026)
│   └── monthly_utilization_panel.csv   # Monthly provincial wind/solar utilization panel (2024-01–2025-12)
├── code/                               # Analysis pipeline (see below)
└── results/                            # Key estimation outputs for comparison
```

## Code overview

| Script | Purpose |
|---|---|
| `redesign_analysis_v8d.py` | Main results (baseline regressions, interaction estimates) |
| `robustness_v8d.py` | Robustness battery (leave-one-out, randomization inference, etc.) |
| `gapfill_analysis_v8d.py` | Gap-filling analysis and pre-trend diagnostics |
| `did_analysis_v8d.py` | Difference-in-differences design |
| `review_response_v9.py` | Additional analyses requested in internal review |
| `monthly_heating_mechanism.py` | Monthly heating-season mechanism (Supporting Information S4) |
| `make_figures_v1.py` / `make_updates_v8d.py` / `make_updates_v9.py` | Figures |
| `make_tables_v1.py` | Tables |
| `replicate_t2c_fixest.R` | Independent R/`fixest` replication of the main results (SI S2) |

## Requirements

- Python 3.x with `pandas`, `numpy`, `statsmodels`, `matplotlib`
- R (≥ 4.x) with `fixest` for the R replication script

## Reproduction notes

1. Place `master_panel_v8d.csv` in the working directory (scripts reference it by relative path from the project root).
2. Run `redesign_analysis_v8d.py` for the main results, then the robustness/gap-fill/DiD scripts.
3. Compare outputs against the CSVs in `results/`.
4. Run `replicate_t2c_fixest.R` for the independent R replication.

## Citation

If you use this data or code, please cite the paper once published. Until then, please contact the corresponding author.
