# -*- coding: utf-8 -*-
"""
monthly_heating_mechanism.py
供暖季机制验证: 用月度分省利用率数据检验 "以热定电挤压风电" 的季节性机制
数据: monthly_data/monthly_utilization_panel.csv (2024-01~2025-12, 24个月, 33行/月)
机制预测:
  M1 描述: 高rigidity省份弃电率供暖季更高, 季节差随rigidity递增
  M2 主检验: curt ~ HeatSeason×rigidity_z + 省FE + 年月FE, 风电β2>0
  M3 安慰剂: 夏季(6-8月)×rigidity 应≈0或负
  M4 逐月剖面: rigidity×月份哑变量 12个系数, 应冬峰夏谷
注: 蒙东/蒙西合并为Inner Mongolia(简单平均, 无装机权重, 附注)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
np.random.seed(20260725)
RESULTS = []

# ---------------- 数据装配 ----------------
m = pd.read_csv(os.path.join(BASE, 'monthly_data', 'monthly_utilization_panel.csv'))
m = m[m['province_en'] != 'National'].copy()
# 蒙东/蒙西合并为 Inner Mongolia (简单平均)
im = m[m['province_en'].isin(['Inner Mongolia East', 'Inner Mongolia West'])]
im = im.groupby(['year', 'month'], as_index=False)[['wind_util_pct', 'solar_util_pct',
        'wind_util_ytd_pct', 'solar_util_ytd_pct']].mean()
im['province_en'] = 'Inner Mongolia'
m = m[~m['province_en'].isin(['Inner Mongolia East', 'Inner Mongolia West'])]
m = pd.concat([m, im], ignore_index=True)
m['wind_curt'] = 100 - m['wind_util_pct']
m['solar_curt'] = 100 - m['solar_util_pct']
m['ym'] = m['year'] * 100 + m['month']
# 与主分析一致: 剔除 Tibet/Xinjiang
m = m[~m['province_en'].isin(['Tibet', 'Xinjiang'])].copy()
HEAT = {11, 12, 1, 2, 3}
m['heat_season'] = m['month'].isin(HEAT).astype(int)
m['summer'] = m['month'].isin({6, 7, 8}).astype(int)

# heat_rigidity (2023基期, 与主分析同口径)
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'))
p23 = df[df['year'] == 2023].set_index('province_en')
hr = (p23['yb_heat_supply_capacity_mw'].fillna(0) / p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())
m['rigidity'] = m['province_en'].map(hr)
m = m.dropna(subset=['rigidity'])
m['rigidity_z'] = (m['rigidity'] - m['rigidity'].mean()) / m['rigidity'].std()
m['heatXrig'] = m['heat_season'] * m['rigidity_z']
m['sumXrig'] = m['summer'] * m['rigidity_z']
print(f'月度面板: {len(m)} obs, {m["province_en"].nunique()}省, {m["ym"].nunique()}个月')

def run_twfe(d, outcome, target, rhs, tag):
    dd = d.dropna(subset=[outcome, target] + rhs)
    f = f'{outcome} ~ ' + ' + '.join([target] + rhs) + ' + C(province_en) + C(ym)'
    mod = smf.ols(f, data=dd).fit(cov_type='cluster', cov_kwds={'groups': dd['province_en']})
    b, se, p = mod.params[target], mod.bse[target], mod.pvalues[target]
    RESULTS.append(dict(analysis=tag, outcome=outcome, coef=target, beta=b, se=se, p=p,
                        n_obs=len(dd), n_prov=dd['province_en'].nunique()))
    return b, se, p

# ---------------- M1 描述: 高低rigidity组季节均值 ----------------
med = m['rigidity'].median()
m['grp'] = np.where(m['rigidity'] >= med, 'High heating lock-in', 'Low heating lock-in')
desc = m.groupby(['grp', 'heat_season'])[['wind_curt', 'solar_curt']].mean().round(3)
print('\n== M1 分组季节均值 ==')
print(desc.to_string())

# ---------------- M2 主检验: 供暖季×rigidity ----------------
print('\n== M2 供暖季×rigidity (TWFE, 省+年月FE) ==')
for out in ['wind_curt', 'solar_curt']:
    b, se, p = run_twfe(m, out, 'heatXrig', ['heat_season'], 'M2_heatseasonXrigidity')
    print(f'{out}: beta2={b:+.3f} (se {se:.3f}, p={p:.4f})')

# ---------------- M3 安慰剂: 夏季×rigidity ----------------
print('\n== M3 安慰剂: 夏季×rigidity ==')
for out in ['wind_curt', 'solar_curt']:
    b, se, p = run_twfe(m, out, 'sumXrig', ['summer'], 'M3_summerXrigidity')
    print(f'{out}: beta={b:+.3f} (se {se:.3f}, p={p:.4f})')

# ---------------- M4 逐月剖面: rigidity×月份 ----------------
print('\n== M4 rigidity×月份剖面 (风电) ==')
coefs = {}
for mo in range(1, 13):
    m[f'mo{mo}Xrig'] = (m['month'] == mo).astype(int) * m['rigidity_z']
dd = m.dropna(subset=['wind_curt', 'rigidity_z'])
rhs = ' + '.join([f'mo{mo}Xrig' for mo in range(1, 13)])
mod = smf.ols(f'wind_curt ~ {rhs} + C(province_en) + C(ym)', data=dd).fit(
    cov_type='cluster', cov_kwds={'groups': dd['province_en']})
for mo in range(1, 13):
    coefs[mo] = (mod.params[f'mo{mo}Xrig'], mod.bse[f'mo{mo}Xrig'])
    print(f'  {mo:02d}月: {mod.params[f"mo{mo}Xrig"]:+.3f} (se {mod.bse[f"mo{mo}Xrig"]:.3f})')
    RESULTS.append(dict(analysis='M4_month_profile', outcome='wind_curt', coef=f'm{mo}',
                        beta=mod.params[f'mo{mo}Xrig'], se=mod.bse[f'mo{mo}Xrig'],
                        p=mod.pvalues[f'mo{mo}Xrig'], n_obs=len(dd), n_prov=29))

# ---------------- 图1: 分组季节曲线 ----------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mm = m.copy()
mm['cal'] = pd.to_datetime(dict(year=mm['year'], month=mm['month'], day=1))
for ax, var, lab in [(axes[0], 'wind_curt', 'Wind curtailment (%)'),
                     (axes[1], 'solar_curt', 'Solar curtailment (%)')]:
    for grp, c in [('High heating lock-in', '#D55E00'), ('Low heating lock-in', '#0072B2')]:
        s = mm[mm['grp'] == grp].groupby('cal')[var].mean()
        ax.plot(s.index, s.values, color=c, lw=2, marker='o', ms=3.5, label=lab.replace(' curtailment (%)','') + ', ' + grp.lower())
    for x0, x1 in [('2024-01-01', '2024-03-31'), ('2024-11-01', '2025-03-31'), ('2025-11-01', '2025-12-31')]:
        ax.axvspan(pd.Timestamp(x0), pd.Timestamp(x1), color='gray', alpha=0.12)
    ax.set_ylabel(lab); ax.legend(frameon=False, fontsize=9)
    ax.grid(axis='y', ls=':', color='lightgray')
axes[0].set_title('(a) Wind curtailment by heating lock-in group')
axes[1].set_title('(b) Solar curtailment by heating lock-in group')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'monthly_data', 'figM1_seasonal_groups.png'), dpi=300)
plt.savefig(os.path.join(BASE, 'monthly_data', 'figM1_seasonal_groups.pdf'))
plt.close()

# ---------------- 图2: 逐月剖面 ----------------
fig, ax = plt.subplots(figsize=(8, 5))
xs = list(range(1, 13))
bs = [coefs[x][0] for x in xs]
ses = [coefs[x][1] for x in xs]
ax.axhline(0, color='gray', lw=0.8)
ax.errorbar(xs, bs, yerr=[1.645*s for s in ses], fmt='o-', color='#D55E00', capsize=4, lw=2)
for x0, x1 in [(0.5, 3.5), (10.5, 12.5)]:
    ax.axvspan(x0, x1, color='gray', alpha=0.12)
ax.set_xticks(xs); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_xlabel('Month'); ax.set_ylabel('Coefficient on month × heating lock-in (z)')
ax.grid(axis='y', ls=':', color='lightgray')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'monthly_data', 'figM4_month_profile.png'), dpi=300)
plt.savefig(os.path.join(BASE, 'monthly_data', 'figM4_month_profile.pdf'))
plt.close()

# ---------------- 输出 ----------------
res = pd.DataFrame(RESULTS)
res.to_csv(os.path.join(BASE, 'monthly_data', 'monthly_mechanism_results.csv'), index=False)
print('\nsaved: monthly_data/monthly_mechanism_results.csv, figM1_seasonal_groups.png/pdf, figM4_month_profile.png/pdf')
