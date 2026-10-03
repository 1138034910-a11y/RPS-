# E1 尺度效应 vs 机制 赛马报告

- 生成: 2026-09-30, 脚本 01_analysis_current/e1_baseline_horserace.py
- 预注册判定: **A_存活** -- 赛马后 β2_heat = 0.280 (保留 122%), p=0.0000
- 自检: T2c bench β2_heat = 0.229 (锚点 0.229) -> 自检通过

## 1. 相关矩阵(29 省, 含 p 值)

| 变量对 | r | p | n |
|---|---|---|---|
| heat_rigidity × base_level | +0.251 | 0.1885 | 29 |
| heat_rigidity × base_2018 | +0.235 | 0.2206 | 29 |
| heat_rigidity × base_vol | +0.051 | 0.7945 | 29 |
| heat_rigidity × pretrend_slope | -0.178 | 0.3565 | 29 |
| delta_target × heat_rigidity | +0.327 | 0.0834 | 29 |
| delta_target × base_level | +0.332 | 0.0786 | 29 |
| delta_target × pretrend_slope | -0.256 | 0.1803 | 29 |
| base_level × base_vol | +0.756 | 0.0000 | 29 |

## 2. 赛马结果(被解释变量: wind_curtailment_pct)

| 设计 | 设定 | 系数项 | β | se | p(cluster) | p(wildboot) | n |
|---|---|---|---|---|---|---|---|
| T2c | bench_heat | dw_heat | 0.2295*** | 0.0609 | 0.0002 | 0.0040 | 174 |
| T2c | scale_only | dw_base | 0.0013 | 0.0925 | 0.9889 | 0.9860 | 174 |
| T2c | race | dw_heat | 0.2799*** | 0.0689 | 0.0000 | 0.0010 | 174 |
| T2c | race | dw_base | -0.1199 | 0.0741 | 0.1056 | 0.1530 | 174 |
| T2c | race_plus_vol | dw_heat | 0.2713*** | 0.0786 | 0.0006 | 0.0040 | 174 |
| T2c | race_plus_vol | dw_base | -0.0963 | 0.0951 | 0.3115 | 0.5730 | 174 |
| T2c | race_plus_vol | dw_basevol | -0.0302 | 0.0487 | 0.5353 | 0.5530 | 174 |
| T2c | race_base2018 | dw_heat | 0.2381*** | 0.0589 | 0.0001 | 0.0020 | 174 |
| T2c | race_base2018 | dw_base2018 | -0.0244 | 0.0212 | 0.2496 | 0.2320 | 174 |
| T2c | race_plus_pretrend | dw_heat | 0.2797*** | 0.0553 | 0.0000 | 0.0030 | 174 |
| T2c | race_plus_pretrend | dw_base | -0.1537* | 0.0842 | 0.0678 | 0.4900 | 174 |
| T2c | race_plus_pretrend | dw_pretrend | -0.0656 | 0.0587 | 0.2637 | 0.8010 | 174 |
| T2c | race_plus_atcap | dw_heat | 0.2765*** | 0.0782 | 0.0004 | 0.0030 | 174 |
| T2c | race_plus_atcap | dw_base | -0.1010 | 0.1348 | 0.4536 | 0.6180 | 174 |
| T2c | race_plus_atcap | dw_atcap | -0.0477 | 0.1921 | 0.8038 |  | 174 |
| T2c | race_baseXyearFE | dw_heat | 0.2895*** | 0.0653 | 0.0000 | 0.0020 | 174 |
| T2c | race_baseXyearFE | dw_base | -0.3362*** | 0.0820 | 0.0000 | 0.0120 | 174 |
| T2c | heating_only_bench | dw_heat | 0.2316*** | 0.0435 | 0.0000 | 0.0060 | 150 |
| T2c | heating_only_race | dw_heat | 0.2433*** | 0.0475 | 0.0000 | 0.0020 | 150 |
| T2c | heating_only_race | dw_base | -0.0239 | 0.0365 | 0.5121 | 0.5320 | 150 |
| T2c | log_bench | dw_heat | 0.0486*** | 0.0132 | 0.0002 | 0.0070 | 174 |
| T2c | log_race | dw_heat | 0.0628*** | 0.0115 | 0.0000 | 0.0020 | 174 |
| T2c | log_race | dw_base | -0.0338*** | 0.0109 | 0.0020 | 0.0150 | 174 |
| T2c | orth_resid_only | dw_heatres | 0.2496*** | 0.0765 | 0.0011 | 0.0020 | 174 |
| T2c | orth_race | dw_heatres | 0.2567*** | 0.0758 | 0.0007 | 0.0070 | 174 |
| T2c | orth_race | dw_base | -0.0395 | 0.0742 | 0.5947 | 0.6530 | 174 |
| T1 | bench_heat | t_heat | 0.0947** | 0.0438 | 0.0307 | 0.0630 | 203 |
| T1 | scale_only | t_base | 0.0590 | 0.0801 | 0.4618 | 0.5290 | 203 |
| T1 | race | t_heat | 0.0857 | 0.0540 | 0.1124 | 0.1380 | 203 |
| T1 | race | t_base | 0.0231 | 0.0874 | 0.7918 | 0.8520 | 203 |
| T1 | race_baseXpost | t_heat | 0.1412** | 0.0555 | 0.0109 | 0.0300 | 203 |
| T1 | race_baseXpost | t_base | -0.4120** | 0.2022 | 0.0417 | 0.0920 | 203 |
| T1 | race_base2018 | t_heat | 0.0936** | 0.0459 | 0.0416 | 0.0950 | 203 |
| T1 | race_base2018 | t_base2018 | -0.1517*** | 0.0464 | 0.0011 | 0.3340 | 203 |
| T1 | race_full | t_heat | 0.1054* | 0.0578 | 0.0683 | 0.1320 | 203 |
| T1 | race_full | t_base | -0.2566* | 0.1501 | 0.0874 | 0.1100 | 203 |
| T1 | race_full | t_basevol | -0.1120* | 0.0662 | 0.0908 | 0.5030 | 203 |

## 3. 诊断与警告

- [T2c|race] VIF(dw_heat) = 1.17
- [T2c|race] VIF(dw_base) = 1.17
- [T2c|race_plus_vol] VIF(dw_heat) = 1.33
- [T2c|race_plus_vol] VIF(dw_base) = 2.35
- [T2c|race_plus_vol] VIF(dw_basevol) = 2.00
- [T2c|race_base2018] VIF(dw_heat) = 1.08
- [T2c|race_base2018] VIF(dw_base2018) = 1.08
- [T2c|race_plus_pretrend] VIF(dw_heat) = 1.17
- [T2c|race_plus_pretrend] VIF(dw_base) = 1.41
- [T2c|race_plus_pretrend] VIF(dw_pretrend) = 1.23
- [T2c|race_plus_atcap] VIF(dw_heat) = 1.35
- [T2c|race_plus_atcap] VIF(dw_base) = 5.42
- [T2c|race_plus_atcap] VIF(dw_atcap) = 4.76
- [T2c|race_baseXyearFE] VIF(dw_heat) = 1.17
- [T2c|race_baseXyearFE] VIF(dw_base) = 1.17
- [T2c|heating_only_race] VIF(dw_heat) = 1.30
- [T2c|heating_only_race] VIF(dw_base) = 1.30
- [T2c|log_race] VIF(dw_heat) = 1.17
- [T2c|log_race] VIF(dw_base) = 1.17
- [T2c|orth_race] VIF(dw_heatres) = 1.02
- [T2c|orth_race] VIF(dw_base) = 1.02
- [T1|race] VIF(t_heat) = 1.23
- [T1|race] VIF(t_base) = 1.23
- [T1|race_baseXpost] VIF(t_heat) = 1.23
- [T1|race_baseXpost] VIF(t_base) = 1.23
- [T1|race_base2018] VIF(t_heat) = 1.08
- [T1|race_base2018] VIF(t_base2018) = 1.08
- [T1|race_full] VIF(t_heat) = 1.32
- [T1|race_full] VIF(t_base) = 2.50
- [T1|race_full] VIF(t_basevol) = 2.07

## 4. 预注册判定标准(跑前写明)

- 存活(A): 赛马列 β2_heat 量级保留 ≥70% 且 p<0.05
- 半活(B): 40-70% 或边缘显著 -> 降级表述, 主打供热省子样本+月度季节签名
- 死(C): base 显著且 β2_heat <40% 或不显著 -> 标题级结论不成立
- 共线陷阱: corr(heat, base_level)>0.8 时赛马无分辨力, 以子样本/正交化为准