# E5 顺手件报告

- 生成: 2026-09-30 | 脚本: 01_analysis_current/e5_odds_and_ends.py

## 结果

| 检验 | 统计量 | p |
|---|---|---|
| E5a_eventstudy_jointF (wind_curtailment_pct) | 0.9025 | 0.4933 |
| E5a_eventstudy_jointF (solar_curtailment_pct) | 1.5462 | 0.2078 |
| E5b_wildboot_9999 (wind_curtailment_pct) | 0.2295 | 0.0017 |
| E5c_ME_low (wind_curtailment_pct) | -0.3951 | 0.0124 |
| E5c_ME_mid (wind_curtailment_pct) | -0.2267 | 0.0741 |
| E5c_ME_high (wind_curtailment_pct) | 0.1248 | 0.1274 |
| E5d_T1_with_storage5xPost (wind_curtailment_pct) | 0.0954 | 0.0324 |

## 备注

- E5b 锚点: beta2=0.2295 (期望≈0.229), cluster p=0.0002, wild9999 p=0.0017
- tercile 分组: {'low': ['Chongqing', 'Fujian', 'Guangdong', 'Guangxi', 'Guizhou', 'Hainan', 'Hunan', 'Qinghai', 'Sichuan', 'Yunnan'], 'mid': ['Anhui', 'Beijing', 'Gansu', 'Inner Mongolia', 'Jiangxi', 'Ningxia', 'Shaanxi', 'Shanghai', 'Zhejiang'], 'high': ['Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Jiangsu', 'Jilin', 'Liaoning', 'Shandong', 'Shanxi', 'Tianjin']}
- E5c 线性隐含 ME @ Hunan (z=-1.06): -0.375 pp/pp —— 与实际反弹方向对比见正文讨论
- E5c 线性隐含 ME @ Guangxi (z=-1.28): -0.427 pp/pp —— 与实际反弹方向对比见正文讨论
- E5c 线性隐含 ME @ Qinghai (z=-1.52): -0.481 pp/pp —— 与实际反弹方向对比见正文讨论
- E5d corr(lock-in, 储能前五省哑变量) = +0.218 (p=0.2570); T1 加储能×Post 后 beta2=0.0954 (p=0.0324)
- E5e 对账: T1 n=203 (29省x7年=203), T2c n=174 (29x6=174, 2018差分损耗), G1b n=199 (=203-4, 缺 actual_nonhydro_rate 的省年: Inner Mongolia 2018, Inner Mongolia 2019, Inner Mongolia 2020, Inner Mongolia 2021)