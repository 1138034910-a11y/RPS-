# -*- coding: utf-8 -*-
"""
e4_counterfactual_redo.py
E4: 反事实 1.69 TWh 重做(弹性区间 + 不确定性传播 + 双基准 + 分母口径 + CO2 换算)
================================================================================
回应审稿意见(见 改投准备/review_eiar/DECISION.md):
  R1-缺点4: 用 T2c 弹性评估一次性冲击, 与 T1 解释差 2.4 倍; 无 CI; "4%"分母口径不清
  R4-C4: 三重沙基——2023 节奏零成本假设(与逐年系数矛盾) + 点估计无不确定性传播 + 分母口径切换
  R3-M3b / R4-m6: 缺 CO2 环境口径(EIA Review 读者需要)
  R1⑤(SETA): 省级明细全表 + 不确定性

换算口径严格复用 make_updates_v9.py(不得新造口径):
  avoided_twh_i = max(ME_i,0) x E_i / 100 x wind_generation_i(1e8 kWh) / 10
  ME_i = β1 + β2 x z_i; E_i = max(0, ΔT_2024 - ΔT_2023)(2023节奏基准)

预注册输出(不判定生死, 只要求口径诚实):
  - 弹性区间: T2c(0.229) 与 T1(0.095) 两套 (β1,β2) 各算一次, 并列报告
  - CI: 从 (β1,β2) 聚类协方差 MVN 抽样 10,000 次, 传播到总量 TWh
  - 双基准: 2023 节奏基准 vs 零增量基准(E_i = ΔT_2024)
  - 双分母: 面板口径(29省自算全国弃风) vs 官方口径(42.6 TWh, 含新疆)
  - CO2: 排放因子区间 [0.55, 0.90] tCO2/MWh(全国平均~0.55, 北方电网更高)
  - 阈值敏感性: ME>0 截断(原版) vs 全样本不截断 vs 0.64 阈值
自检锚点: 基准变体总量应 ≈1.69 TWh(容差 ±0.05)。
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os, warnings

warnings.filterwarnings('ignore')
BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
OUTDIR = os.path.join(BASE, '01_analysis_current')
np.random.seed(20260724)
CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']
NDRAW = 10000
EF_LO, EF_HI = 0.55, 0.90      # tCO2/MWh
OFFICIAL_TWH = 42.6            # 官方口径全国弃风(含新疆, 996.8TWh发电量/95.9%利用率推出)

# ---------------- 数据 ----------------
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df[~df['province_en'].isin(['Tibet', 'Xinjiang'])].copy()
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)
assert 'wind_generation' in df.columns, '缺 wind_generation 列(1e8 kWh)'

# lock-in(2023 基年; 论文口径: 供热容量/火电装机, 同名单位相除, 均值0.460, sd 0.303)
p23 = df[df['year'] == 2023].set_index('province_en')
heat = (p23['yb_heat_supply_capacity_mw'].fillna(0) /
        p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna()).rename('heat_rigidity')

# ---------------- 拟合两套弹性 ----------------
def fit_t2c():
    d = df[df['year'].between(2018, 2024)].copy()
    d['dweight'] = d.groupby('province_en')['nonhydro_weight_binding'].diff()
    d = d[d['year'].between(2019, 2024)].copy()
    col = d['province_en'].map(heat)
    d['heat_z'] = (col - col.mean()) / col.std()
    d['dw_heat'] = d['dweight'] * d['heat_z']
    d = d.dropna(subset=['wind_curtailment_pct', 'dweight', 'dw_heat'] + CTRL_B)
    m = smf.ols('wind_curtailment_pct ~ dw_heat + dweight + ' + ' + '.join(CTRL_B) +
                ' + C(province_en) + C(year)', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['province_en']})
    # 样本 z 化矩(用于把省级 rigidity 转到 z 尺度)
    return m, col.mean(), col.std(), d

def fit_t1():
    d = df[df['year'].between(2018, 2024)].copy()
    col = d['province_en'].map(heat)
    d['heat_z'] = (col - col.mean()) / col.std()
    d['post'] = (d['year'] >= 2024).astype(int)
    d['dt_post'] = d['delta_target_i'] * d['post']
    d['t_heat'] = d['dt_post'] * d['heat_z']
    d = d.dropna(subset=['wind_curtailment_pct', 'dt_post', 't_heat'] + CTRL_B)
    m = smf.ols('wind_curtailment_pct ~ t_heat + dt_post + ' + ' + '.join(CTRL_B) +
                ' + C(province_en) + C(year)', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['province_en']})
    return m, col.mean(), col.std(), d

m2, zm2, zs2, d2 = fit_t2c()
m1, zm1, zs1, d1 = fit_t1()
B1_T2C, B2_T2C = m2.params['dweight'], m2.params['dw_heat']
B1_T1, B2_T1 = m1.params['dt_post'], m1.params['t_heat']
print(f'T2c: beta1={B1_T2C:.4f}, beta2={B2_T2C:.4f} (锚点: -0.133 / 0.229)')
print(f'T1 : beta1={B1_T1:.4f}, beta2={B2_T1:.4f} (锚点: +0.210 / 0.095)')

# ---------------- 省级明细 ----------------
d22 = df[df['year'] == 2022].set_index('province_en')
d24 = df[df['year'] == 2024].set_index('province_en')
sim = pd.DataFrame(index=heat.index.dropna())
sim['rigidity'] = heat
sim['dT_2024'] = d24['delta_target_i']
sim['dT_2023'] = p23['nonhydro_weight_binding'] - d22['nonhydro_weight_binding']
sim['E_2023pace'] = (sim['dT_2024'] - sim['dT_2023']).clip(lower=0)
sim['E_zero'] = sim['dT_2024'].clip(lower=0)
sim['wind_gen_1e8kwh'] = d24['wind_generation']
sim = sim.dropna(subset=['dT_2024', 'E_2023pace', 'wind_gen_1e8kwh'])

def cf_total(beta1, beta2, zmean, zsd, E_col, clip_mode='pos', threshold=None):
    z = (sim['rigidity'] - zmean) / zsd
    me = beta1 + beta2 * z
    if clip_mode == 'pos':
        me_use = me.clip(lower=0)
    elif clip_mode == 'threshold':
        me_use = me.where(sim['rigidity'] >= threshold, 0.0)
    else:
        me_use = me
    avoided_pp = me_use * sim[E_col]
    twh = avoided_pp / 100.0 * sim['wind_gen_1e8kwh'] / 10.0
    return twh, me

# 主变体(与 v9 一致的口径, 用于锚点自检)
twh_main, me_main = cf_total(B1_T2C, B2_T2C, zm2, zs2, 'E_2023pace', 'pos')
tot_main = twh_main.sum()
anchor_ok = abs(tot_main - 1.69) < 0.05
print(f'锚点自检: 主变体总量={tot_main:.3f} TWh vs 1.69 -> {"通过" if anchor_ok else "⚠️不通过"}')

# 省级明细表(主变体)
sim['z'] = (sim['rigidity'] - zm2) / zs2
sim['ME'] = me_main
sim['avoided_pp'] = (me_main.clip(lower=0) * sim['E_2023pace'])
sim['avoided_twh'] = twh_main
sim['value_mio_cny'] = sim['avoided_twh'] * 1e9 * 0.35 / 1e6
sim_out = sim[['rigidity', 'z', 'dT_2024', 'dT_2023', 'E_2023pace', 'ME',
               'avoided_pp', 'avoided_twh', 'value_mio_cny']].round(4)
sim_out.to_csv(os.path.join(OUTDIR, 'e4_province_detail.csv'), encoding='utf-8-sig')

# ---------------- 变体矩阵 ----------------
VARIANTS = []
def add_variant(name, beta1, beta2, zm, zs, E, clip, thr=None):
    twh, _ = cf_total(beta1, beta2, zm, zs, E, clip, thr)
    VARIANTS.append(dict(variant=name, total_twh=twh.sum()))

add_variant('T2c | 2023节奏基准 | ME>0截断(原版)', B1_T2C, B2_T2C, zm2, zs2, 'E_2023pace', 'pos')
add_variant('T2c | 零增量基准 | ME>0截断', B1_T2C, B2_T2C, zm2, zs2, 'E_zero', 'pos')
add_variant('T1  | 2023节奏基准 | ME>0截断', B1_T1, B2_T1, zm1, zs1, 'E_2023pace', 'pos')
add_variant('T1  | 零增量基准 | ME>0截断', B1_T1, B2_T1, zm1, zs1, 'E_zero', 'pos')
add_variant('T2c | 2023节奏基准 | 不截断(全省)', B1_T2C, B2_T2C, zm2, zs2, 'E_2023pace', 'none')
add_variant('T2c | 2023节奏基准 | 阈值0.64', B1_T2C, B2_T2C, zm2, zs2, 'E_2023pace', 'threshold', 0.64)
add_variant('T2c | 2023节奏基准 | 阈值0.80', B1_T2C, B2_T2C, zm2, zs2, 'E_2023pace', 'threshold', 0.80)

# ---------------- 不确定性传播(MVN 抽样, 主变体) ----------------
cov = m2.cov_params().loc[['dweight', 'dw_heat'], ['dweight', 'dw_heat']].values
mu = np.array([B1_T2C, B2_T2C])
draws = np.random.multivariate_normal(mu, cov, NDRAW)
tot_draws = np.array([cf_total(b[0], b[1], zm2, zs2, 'E_2023pace', 'pos')[0].sum()
                      for b in draws])
ci_lo, ci_hi = np.percentile(tot_draws, [2.5, 97.5])
p_zero = float((tot_draws <= 0).mean())

# ---------------- 分母与 CO2 ----------------
# 面板口径全国弃风: curtailed = gen x rate/(1-rate) 对29省求和(2024)
d24s = d24.loc[sim.index]
rate = d24s['wind_curtailment_pct'] / 100.0
curt_panel_twh = (d24s['wind_generation'] / 10.0 * rate / (1 - rate)).sum()
share_panel = tot_main / curt_panel_twh * 100
share_official = tot_main / OFFICIAL_TWH * 100
co2_lo, co2_hi = tot_main * EF_LO, tot_main * EF_HI   # TWh x tCO2/MWh = MtCO2

VARIANTS.append(dict(variant='[诊断] 面板口径全国弃风(29省, TWh)', total_twh=curt_panel_twh))
pd.DataFrame(VARIANTS).to_csv(os.path.join(OUTDIR, 'e4_variants.csv'),
                              index=False, encoding='utf-8-sig')

# ---------------- 报告 ----------------
lines = ['# E4 反事实重做报告', '',
         '- 生成: 2026-09-30 | 脚本: 01_analysis_current/e4_counterfactual_redo.py',
         '- 换算口径严格复用 make_updates_v9.py', '',
         '## 自检与锚点', '',
         f'- 弹性锚点: T2c (β1,β2)=({B1_T2C:.3f}, {B2_T2C:.3f}) [−0.133/0.229]; T1 (β1,β2)=({B1_T1:.3f}, {B2_T1:.3f}) [+0.210/0.095]',
         f'- 主变体总量: **{tot_main:.3f} TWh** vs 旧版 1.69 -> {"通过" if anchor_ok else "⚠️不通过"}', '',
         '## 变体矩阵(总量 TWh)', '',
         '| 变体 | 总量(TWh) |', '|---|---|']
for v in VARIANTS:
    lines.append(f"| {v['variant']} | {v['total_twh']:.3f} |")
lines += ['', '## 不确定性(主变体, MVN 10,000 次抽样)', '',
          f'- 点估计: {tot_main:.3f} TWh',
          f'- 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] TWh',
          f'- P(总量 ≤ 0): {p_zero:.3f}', '',
          '## 分母口径', '',
          f'- 面板口径(29省自算): 全国弃风 {curt_panel_twh:.1f} TWh -> 主变体占 {share_panel:.1f}%',
          f'- 官方口径(含新疆): {OFFICIAL_TWH} TWh -> 主变体占 {share_official:.1f}%', '',
          '## CO2 环境口径(主变体)', '',
          f'- 排放因子 [{EF_LO}, {EF_HI}] tCO2/MWh -> 避免排放 [{co2_lo:.2f}, {co2_hi:.2f}] MtCO2',
          f'- 人民币口径(0.35 CNY/kWh 煤电标杆, 上限): {sim["value_mio_cny"].sum():.0f} 百万元', '',
          '## 口径说明(写作时必须带上的三句)', '',
          '1. 弹性区间两端含义不同: T2c=稳态年增量弹性, T1=一次性冲击弹性; 反事实问的是后者, T2c 为上界参考。',
          '2. 零增量基准隐含"2023年速度也有成本", 与逐年系数一致(2023 交互为正), 故比 2023 节奏基准大。',
          '3. 所有变体均为偏均衡、线性外推、无供给响应的示意性测算(illustrative only)。']
with open(os.path.join(OUTDIR, 'e4_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
safe = '\n'.join(lines[:20]).encode('gbk', 'replace').decode('gbk')
print(safe)
print('\n完成 -> e4_variants.csv / e4_province_detail.csv / e4_report.md')
