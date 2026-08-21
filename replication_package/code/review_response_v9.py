# -*- coding: utf-8 -*-
"""
review_response_v9.py
终审意见补做实验 (数据: master_panel_v8d.csv)
  W1a: T2c 剔除2024 (2019-2023) 交互估计 — 识别来源分解
  W1b: 逐年交互系数 (Δweight×rigidity 按年份) — 效应时间轮廓
  W1c: 2024年单独交互 vs T1点估计 的系数相等检验 (Wald)
  W3a: T1 的RI改为置换heat_rigidity标签 (与T2c对称, 500次)
  W4c: 2024增幅 与 冲击前弃风水平/趋势 相关性
推断口径与 redesign_analysis_v8d.py 一致 (TWFE, 省聚类SE)。
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os, warnings
warnings.filterwarnings('ignore')
BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
np.random.seed(20260726)
CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']

df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'))
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)
p23 = df[df['year'] == 2023].set_index('province_en')
hr = (p23['yb_heat_supply_capacity_mw'].fillna(0) * 10 / p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())

d = df[(df['year'].between(2018, 2024)) & (~df['province_en'].isin(['Tibet', 'Xinjiang']))].copy()
d['dweight'] = d.groupby('province_en')['nonhydro_weight_binding'].diff()
col = d['province_en'].map(hr)
d['heat_z'] = (col - col.mean()) / col.std()

def twfe(dd, outcome, rhs_target, rhs_others, label):
    dd = dd.dropna(subset=[outcome, rhs_target] + rhs_others + CTRL_B)
    f = f"{outcome} ~ {rhs_target} + " + ' + '.join(rhs_others + CTRL_B) + ' + C(province_en) + C(year)'
    m = smf.ols(f, data=dd).fit(cov_type='cluster', cov_kwds={'groups': dd['province_en']})
    print(f'{label}: n={len(dd)}, beta={m.params[rhs_target]:+.4f}, se={m.bse[rhs_target]:.4f}, p={m.pvalues[rhs_target]:.4f}')
    return m

t2 = d[d['year'].between(2019, 2024)].copy()
t2['dw_weak'] = t2['dweight'] * t2['heat_z']

print('== W1a: T2c 剔除2024 (2019-2023) ==')
t2n = t2[t2['year'] <= 2023].copy()
m_w1a = twfe(t2n, 'wind_curtailment_pct', 'dw_weak', ['dweight'], 'T2c excl-2024')

print('\n== W1b: 逐年交互系数 ==')
for y in range(2019, 2025):
    t2[f'int_{y}'] = t2['dweight'] * t2['heat_z'] * (t2['year'] == y)
rhs_int = [f'int_{y}' for y in range(2019, 2025)]
dd = t2.dropna(subset=['wind_curtailment_pct', 'dweight'] + rhs_int + CTRL_B)
f = 'wind_curtailment_pct ~ dweight + ' + ' + '.join(rhs_int + CTRL_B) + ' + C(province_en) + C(year)'
m_w1b = smf.ols(f, data=dd).fit(cov_type='cluster', cov_kwds={'groups': dd['province_en']})
yearly = {}
for y in range(2019, 2025):
    b, se, p = m_w1b.params[f'int_{y}'], m_w1b.bse[f'int_{y}'], m_w1b.pvalues[f'int_{y}']
    yearly[y] = (b, se, p)
    print(f'  {y}: {b:+.4f} (se {se:.4f}, p={p:.4f})')
pd.DataFrame([(y, *v) for y, v in yearly.items()], columns=['year', 'beta', 'se', 'p']).to_csv(
    os.path.join(BASE, 'review_yearly_interactions.csv'), index=False)

print('\n== W1c: 2024单独交互 vs 非2024交互 的系数相等Wald检验 ==')
t2['int_2024'] = t2['dweight'] * t2['heat_z'] * (t2['year'] == 2024)
t2['int_other'] = t2['dweight'] * t2['heat_z'] * (t2['year'] < 2024)
dd2 = t2.dropna(subset=['wind_curtailment_pct', 'int_2024', 'int_other'] + CTRL_B)
f2 = 'wind_curtailment_pct ~ dweight + int_2024 + int_other + ' + ' + '.join(CTRL_B) + ' + C(province_en) + C(year)'
m_w1c = smf.ols(f2, data=dd2).fit(cov_type='cluster', cov_kwds={'groups': dd2['province_en']})
b24, b_o = m_w1c.params['int_2024'], m_w1c.params['int_other']
wt = m_w1c.wald_test('int_2024 = int_other')
print(f'  int_2024 = {b24:+.4f} (p={m_w1c.pvalues["int_2024"]:.4f})')
print(f'  int_other = {b_o:+.4f} (p={m_w1c.pvalues["int_other"]:.4f})')
print(f'  Wald (equal): {wt}')

print('\n== W3a: T1 RI 置换rigidity标签 (500次) ==')
d18 = d.copy()
d18['post'] = (d18['year'] >= 2024).astype(int)
d18['dt_post'] = d18['delta_target_i'] * d18['post']
d18['treat_weak'] = d18['dt_post'] * d18['heat_z']
need = ['wind_curtailment_pct', 'treat_weak', 'dt_post'] + CTRL_B
d18f = d18.dropna(subset=need)
m_t1 = twfe(d18f, 'wind_curtailment_pct', 'treat_weak', ['dt_post'], 'T1 actual')
beta_actual = m_t1.params['treat_weak']
provs = d18f['province_en'].unique()
rng = np.random.default_rng(20260726)
cnt = 0
perms = 500
for i in range(perms):
    perm = rng.permutation(provs)
    hmap = dict(zip(provs, d18f.groupby('province_en')['heat_z'].first().loc[perm].values))
    d18f['heat_p'] = d18f['province_en'].map(hmap)
    d18f['tw_p'] = d18f['dt_post'] * d18f['heat_p']
    try:
        mp = smf.ols('wind_curtailment_pct ~ tw_p + dt_post + ' + ' + '.join(CTRL_B) +
                     ' + C(province_en) + C(year)', data=d18f).fit()
        if abs(mp.params['tw_p']) >= abs(beta_actual):
            cnt += 1
    except Exception:
        cnt += 1
ri_p = (cnt + 1) / (perms + 1)
print(f'  RI-T1 (permute rigidity labels): beta_actual={beta_actual:+.4f}, RI p={ri_p:.4f}')

print('\n== W4c: 2024增幅 与 冲击前弃风水平/趋势 相关 ==')
rows = []
for p, g in d.groupby('province_en'):
    g = g.sort_values('year')
    pre = g[g['year'] <= 2022]
    lvl = pre['wind_curtailment_pct'].mean()
    slope = np.polyfit(pre['year'], pre['wind_curtailment_pct'], 1)[0]
    rows.append(dict(province_en=p, pre_level=lvl, pre_slope=slope,
                     delta_target=g['delta_target_i'].iloc[0]))
w4 = pd.DataFrame(rows).dropna()
print(f"  corr(ΔTarget, 2018-2022弃风均值) = {w4['delta_target'].corr(w4['pre_level']):.3f}")
print(f"  corr(ΔTarget, 2018-2022弃风趋势) = {w4['delta_target'].corr(w4['pre_slope']):.3f}")
w4.to_csv(os.path.join(BASE, 'review_w4c_preshock_corr.csv'), index=False)

with open(os.path.join(BASE, 'review_response_v9_results.txt'), 'w') as f:
    f.write(f'W1a_T2c_excl2024: beta={m_w1a.params["dw_weak"]:.4f}, p={m_w1a.pvalues["dw_weak"]:.4f}, n={len(t2n.dropna(subset=["wind_curtailment_pct","dw_weak","dweight"]+CTRL_B))}\n')
    f.write(f'W1b_yearly: {yearly}\n')
    f.write(f'W1c: int_2024={b24:.4f} (p={m_w1c.pvalues["int_2024"]:.4f}), int_other={b_o:.4f} (p={m_w1c.pvalues["int_other"]:.4f}), Wald={float(wt.statistic):.3f} (p={float(wt.pvalue):.4f})\n')
    f.write(f'W3a_RI_T1_permRigidity: beta={beta_actual:.4f}, RI p={ri_p:.4f}\n')
    f.write(f'W4c: corr(dT,pre_level)={w4["delta_target"].corr(w4["pre_level"]):.3f}, corr(dT,pre_slope)={w4["delta_target"].corr(w4["pre_slope"]):.3f}\n')
print('\nsaved: review_yearly_interactions.csv, review_w4c_preshock_corr.csv, review_response_v9_results.txt')
