# E2 剂量倒置与普通年份分配检验报告

- 生成: 2026-09-30 | 脚本: 01_analysis_current/e2_assignment_dose.py
- 自检锚点: bench β2=0.2295 (锚点 0.229) -> 通过

## 判定汇总

- E2d 剔2019: β2=0.193, p=0.0001 -> 通过(落在[0.18,0.30]且显著)
- E2e 残差化: β2=0.277, p=0.0134 -> 通过(分配内生性封住)

## 1. 相关检验(含 p 值)

| 检验 | 变量对 | r | p | n |
|---|---|---|---|---|
| E2a_pooled_ordinary_2019_2023|lag_curt_level | dw_dm × L1_wind_curtailment_pct_dm | +0.037 | 0.6600 | 145 |
| E2a_year2019|lag_curt_level | dweight × L1_wind_curtailment_pct | +0.196 | 0.3077 | 29 |
| E2a_year2020|lag_curt_level | dweight × L1_wind_curtailment_pct | -0.474 | 0.0094 | 29 |
| E2a_year2021|lag_curt_level | dweight × L1_wind_curtailment_pct | +0.228 | 0.2333 | 29 |
| E2a_year2022|lag_curt_level | dweight × L1_wind_curtailment_pct | +0.362 | 0.0536 | 29 |
| E2a_year2023|lag_curt_level | dweight × L1_wind_curtailment_pct | +0.020 | 0.9199 | 29 |
| E2a_pooled_ordinary_2019_2023|lag_curt_change | dw_dm × L1_dcurt_dm | +0.084 | 0.3680 | 116 |
| E2a_year2019|lag_curt_change | dweight × L1_dcurt | +nan | nan | 0 |
| E2a_year2020|lag_curt_change | dweight × L1_dcurt | +0.115 | 0.5537 | 29 |
| E2a_year2021|lag_curt_change | dweight × L1_dcurt | +0.136 | 0.4834 | 29 |
| E2a_year2022|lag_curt_change | dweight × L1_dcurt | -0.050 | 0.7964 | 29 |
| E2a_year2023|lag_curt_change | dweight × L1_dcurt | +0.001 | 0.9942 | 29 |
| E2a_pooled_ordinary_2019_2023|lag_nonhydro_rate | dw_dm × L1_actual_nonhydro_rate_dm | +0.166 | 0.0497 | 141 |
| E2a_year2019|lag_nonhydro_rate | dweight × L1_actual_nonhydro_rate | +0.341 | 0.0755 | 28 |
| E2a_year2020|lag_nonhydro_rate | dweight × L1_actual_nonhydro_rate | +0.311 | 0.1073 | 28 |
| E2a_year2021|lag_nonhydro_rate | dweight × L1_actual_nonhydro_rate | +0.019 | 0.9216 | 28 |
| E2a_year2022|lag_nonhydro_rate | dweight × L1_actual_nonhydro_rate | +0.242 | 0.2148 | 28 |
| E2a_year2023|lag_nonhydro_rate | dweight × L1_actual_nonhydro_rate | +0.132 | 0.4949 | 29 |
| E2b_pooled_2019_2024|lag_slack | dw_dm × L1_slack_dm | +0.466 | 0.0000 | 170 |
| E2b_pooled_ordinary|lag_slack | dw_dm × L1_slack_dm | +0.243 | 0.0037 | 141 |

## 2. 回归结果

| 检验 | 系数 | β | se | p(cluster) | p(wildboot) | n |
|---|---|---|---|---|---|---|
| E2a_ols_ordinary | L1_wind_curtailment_pct | -0.0559 | 0.0478 | 0.2415 |  | 113 |
| E2a_ols_ordinary | L1_dcurt | 0.0517 | 0.0543 | 0.3411 |  | 113 |
| E2a_ols_ordinary | L1_actual_nonhydro_rate | 0.0322 | 0.0149 | 0.0311 |  | 113 |
| E2b_ols_allyears | L1_slack | 0.3292 | 0.0896 | 0.0002 |  | 170 |
| T2c_bench_fullsample(anchor) | dw_heat | 0.2295 | 0.0609 | 0.0002 | 0.0040 | 174 |
| T2c_drop2019 | dw_heat | 0.1927 | 0.0497 | 0.0001 | 0.0010 | 145 |
| T2c_residualized_dweight | dw_heat | 0.2771 | 0.1120 | 0.0134 | 0.0600 | 170 |

## 3. 文档化与警告

- E2a_year2019|lag_curt_change: 有效观测不足(n=0), 跳过
- E2c nonhydro_weight_binding 年度覆盖(省数): {2018: 29, 2019: 29, 2020: 29, 2021: 29, 2022: 29, 2023: 29, 2024: 29}
- E2c dweight_2019 = w2019 - w2018: mean=1.155, min=0.000, max=4.000; 2018基期29省全非空=True
- 自检锚点: bench β2=0.2295 vs 0.229 -> 通过
- E2e 残差化回归 R2=0.490 (Δweight 被滞后变量解释的比例)