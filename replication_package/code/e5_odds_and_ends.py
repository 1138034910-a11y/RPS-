# -*- coding: utf-8 -*-
"""
e5_odds_and_ends.py
E5: 顺手件打包(最小充分矩阵之五)
=================================
回应(见 改投准备/review_eiar/DECISION.md):
  E5a R1-缺点5: 事件研究 pre-2024 系数联合 F 检验(风电+光伏), 补 Fig.3a 的统计量
  E5b R1-缺点12: 主结果 wild bootstrap 升 9,999 次(小 p 值分辨率)
  E5c R4-M6: 分段/三组 ME schedule(线性交互对低锁定端隐含荒谬负预测);
      并算湖南/广西/青海三个低锁定高响应离群点的线性隐含 ME, 供正文讨论
  E5d R1-缺点10: 储能混杂定量化——NEA 储能前五省(内蒙/山东/江苏/宁夏; 新疆不在样本)
      哑变量×Post2024 进 T1; corr(储能代理, lock-in)
  E5e R3-m7: n 对账(199/203/174 差异来源)
锚点自检: T2c β2≈0.229; T1 β2≈0.095。
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats as sstats
import os, warnings

warnings.filterwarnings('ignore')
BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
OUTDIR = os.path.join(BASE, '01_analysis_current')
np.random.seed(20260724)
CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']
ROWS, NOTES = [], []

df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df[~df['province_en'].isin(['Tibet', 'Xinjiang'])].copy()
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)

p23 = df[df['year'] == 2023].set_index('province_en')
heat = (p23['yb_heat_supply_capacity_mw'].fillna(0) /
        p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna()).rename('heat_rigidity')

def zmap_on(d, s):
    col = d['province_en'].map(s)
    return ((s - col.mean()) / col.std()).to_dict()

# ================= E5a 事件研究联合 F 检验 =================
print('== E5a 事件研究联合F ==')
d18 = df[df['year'].between(2018, 2024)].copy()
years_es = [2018, 2019, 2020, 2021, 2022, 2024]   # 基期 2023
for y in years_es:
    d18[f'treat_{y}'] = d18['delta_target_i'] * (d18['year'] == y).astype(int)
terms = [f'treat_{y}' for y in years_es]
for outcome in ['wind_curtailment_pct', 'solar_curtailment_pct']:
    d = d18.dropna(subset=[outcome, 'delta_target_i'])
    m = smf.ols(f"{outcome} ~ {' + '.join(terms)} + C(province_en) + C(year)", data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['province_en']})
    pre = [f'treat_{y}' for y in range(2018, 2023)]
    hyp = ' , '.join(f'{t} = 0' for t in pre)
    F = m.f_test(hyp)
    ROWS.append(dict(test='E5a_eventstudy_jointF', outcome=outcome,
                     stat=float(F.fvalue), p=float(F.pvalue), df_denom=float(F.df_denom)))
    print(f'{outcome}: pre-2024 联合F={float(F.fvalue):.3f}, p={float(F.pvalue):.4f}')

# ================= E5b 主结果 wild bootstrap 9999 =================
print('== E5b wild boot 9999 ==')
d2 = df[df['year'].between(2018, 2024)].copy()
d2['dweight'] = d2.groupby('province_en')['nonhydro_weight_binding'].diff()
d2 = d2[d2['year'].between(2019, 2024)].copy()
d2['heat_z'] = d2['province_en'].map(zmap_on(d2, heat))
d2['dw_heat'] = d2['dweight'] * d2['heat_z']
d2 = d2.dropna(subset=['wind_curtailment_pct', 'dweight', 'dw_heat'] + CTRL_B)
m_main = smf.ols('wind_curtailment_pct ~ dw_heat + dweight + ' + ' + '.join(CTRL_B) +
                 ' + C(province_en) + C(year)', data=d2).fit(
    cov_type='cluster', cov_kwds={'groups': d2['province_en']})
b2, se2 = m_main.params['dw_heat'], m_main.bse['dw_heat']

# wild bootstrap(与管线同实现)
provs = d2['province_en'].values
uniq = np.unique(provs)
years = d2['year'].values
M = [np.ones(len(d2))]
for g in uniq[1:]:
    M.append((provs == g).astype(float))
for yy in np.unique(years)[1:]:
    M.append((years == yy).astype(float))
M = np.column_stack(M)
Ccols = ['dweight'] + CTRL_B
Cmat = d2[Ccols].values.astype(float)
tvec = d2['dw_heat'].values.astype(float)
y = d2['wind_curtailment_pct'].values.astype(float)
X = np.column_stack([tvec, Cmat])
N, K = len(d2), M.shape[1] + X.shape[1]
G = len(uniq)

def absorb(v):
    b, *_ = np.linalg.lstsq(M, v, rcond=None)
    return v - M @ b
X_t = np.column_stack([absorb(X[:, j]) for j in range(X.shape[1])])
XtX_inv = np.linalg.pinv(X_t.T @ X_t)

def cluster_t(ytilde):
    beta = XtX_inv @ (X_t.T @ ytilde)
    e = ytilde - X_t @ beta
    meat = np.zeros((X_t.shape[1], X_t.shape[1]))
    for g in uniq:
        idx = provs == g
        u = X_t[idx].T @ e[idx]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    V *= (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta[0] / se[0] if se[0] > 0 else np.inf

t_obs = abs(b2 / se2)
MR = np.column_stack([M, Cmat])
br, *_ = np.linalg.lstsq(MR, y, rcond=None)
fitted_r, resid_r = MR @ br, y - MR @ br
cnt = 0
NB = 9999
for _ in range(NB):
    w = np.random.choice([-1.0, 1.0], size=G)
    wmap = dict(zip(uniq, w))
    y_star = fitted_r + resid_r * np.array([wmap[g] for g in provs])
    tb = cluster_t(absorb(y_star))
    if not np.isfinite(tb) or abs(tb) >= t_obs:
        cnt += 1
p9999 = (cnt + 1) / (NB + 1)
ROWS.append(dict(test='E5b_wildboot_9999', outcome='wind_curtailment_pct',
                 stat=b2, p=p9999, df_denom=np.nan))
NOTES.append(f'E5b 锚点: beta2={b2:.4f} (期望≈0.229), cluster p={m_main.pvalues["dw_heat"]:.4f}, wild9999 p={p9999:.4f}')

# ================= E5c 三组 ME schedule =================
print('== E5c 分段 ME ==')
d2c = df[df['year'].between(2018, 2024)].copy()
d2c['dweight'] = d2c.groupby('province_en')['nonhydro_weight_binding'].diff()
d2c = d2c[d2c['year'].between(2019, 2024)].copy()
terc = pd.qcut(heat.dropna(), 3, labels=['low', 'mid', 'high'])
NOTES.append('tercile 分组: ' + str({k: v for k, v in terc.groupby(terc).groups.items()
                                     for k, v in [(k, sorted(v))]}))
d2c['t_mid'] = d2c['province_en'].map((terc == 'mid').astype(float))
d2c['t_high'] = d2c['province_en'].map((terc == 'high').astype(float))
d2c['dw_mid'] = d2c['dweight'] * d2c['t_mid']
d2c['dw_high'] = d2c['dweight'] * d2c['t_high']
d2c = d2c.dropna(subset=['wind_curtailment_pct', 'dweight', 'dw_mid', 'dw_high'] + CTRL_B)
m3 = smf.ols('wind_curtailment_pct ~ dweight + dw_mid + dw_high + ' + ' + '.join(CTRL_B) +
             ' + C(province_en) + C(year)', data=d2c).fit(
    cov_type='cluster', cov_kwds={'groups': d2c['province_en']})
me_low = m3.params['dweight']
me_mid = m3.params['dweight'] + m3.params['dw_mid']
me_high = m3.params['dweight'] + m3.params['dw_high']
t_mid = m3.t_test('dweight + dw_mid = 0')
t_high = m3.t_test('dweight + dw_high = 0')
ROWS.append(dict(test='E5c_ME_low', outcome='wind_curtailment_pct', stat=me_low,
                 p=float(m3.pvalues['dweight']), df_denom=np.nan))
ROWS.append(dict(test='E5c_ME_mid', outcome='wind_curtailment_pct', stat=me_mid,
                 p=float(t_mid.pvalue), df_denom=np.nan))
ROWS.append(dict(test='E5c_ME_high', outcome='wind_curtailment_pct', stat=me_high,
                 p=float(t_high.pvalue), df_denom=np.nan))
print(f'ME: low={me_low:.3f}(p={float(m3.pvalues["dweight"]):.3f}), '
      f'mid={me_mid:.3f}(p={float(t_mid.pvalue):.3f}), high={me_high:.3f}(p={float(t_high.pvalue):.3f})')
# 线性隐含 ME 在三个离群省
zm, zs = d2['province_en'].map(heat).mean(), None
colmap = d2.drop_duplicates('province_en').set_index('province_en')['heat_z']
for prov in ['Hunan', 'Guangxi', 'Qinghai']:
    z = colmap.get(prov, np.nan)
    me_imp = m_main.params['dweight'] + m_main.params['dw_heat'] * z
    NOTES.append(f'E5c 线性隐含 ME @ {prov} (z={z:.2f}): {me_imp:+.3f} pp/pp —— 与实际反弹方向对比见正文讨论')

# ================= E5d 储能控制 =================
print('== E5d 储能 ==')
STORAGE5 = ['Inner Mongolia', 'Shandong', 'Jiangsu', 'Ningxia']  # 新疆不在样本
d1 = df[df['year'].between(2018, 2024)].copy()
d1['post'] = (d1['year'] >= 2024).astype(int)
d1['dt_post'] = d1['delta_target_i'] * d1['post']
d1['heat_z'] = d1['province_en'].map(zmap_on(d1, heat))
d1['t_heat'] = d1['dt_post'] * d1['heat_z']
d1['storage5_post'] = d1['province_en'].isin(STORAGE5).astype(int) * d1['post']
d1s = d1.dropna(subset=['wind_curtailment_pct', 'dt_post', 't_heat'] + CTRL_B)
m_s = smf.ols('wind_curtailment_pct ~ t_heat + dt_post + storage5_post + ' +
              ' + '.join(CTRL_B) + ' + C(province_en) + C(year)', data=d1s).fit(
    cov_type='cluster', cov_kwds={'groups': d1s['province_en']})
ROWS.append(dict(test='E5d_T1_with_storage5xPost', outcome='wind_curtailment_pct',
                 stat=m_s.params['t_heat'], p=float(m_s.pvalues['t_heat']), df_denom=np.nan))
sub = pd.concat([heat, pd.Series(heat.index.isin(STORAGE5).astype(float), index=heat.index)],
                axis=1).dropna()
r_s, p_s = sstats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
NOTES.append(f'E5d corr(lock-in, 储能前五省哑变量) = {r_s:+.3f} (p={p_s:.4f}); '
             f'T1 加储能×Post 后 beta2={m_s.params["t_heat"]:.4f} (p={m_s.pvalues["t_heat"]:.4f})')

# ================= E5e n 对账 =================
print('== E5e n 对账 ==')
n_t1 = len(d18.dropna(subset=['wind_curtailment_pct', 'delta_target_i'] + CTRL_B))
n_t2c = len(d2)
d_g1b = d18.dropna(subset=['actual_nonhydro_rate', 'delta_target_i'] + CTRL_B)
n_g1b = len(d_g1b)
miss = d18[d18['actual_nonhydro_rate'].isna()][['province_en', 'year']]
NOTES.append(f'E5e 对账: T1 n={n_t1} (29省x7年=203), T2c n={n_t2c} (29x6=174, 2018差分损耗), '
             f'G1b n={n_g1b} (=203-{203 - n_g1b}, 缺 actual_nonhydro_rate 的省年: '
             + ', '.join(f"{r.province_en} {r.year}" for r in miss.itertuples()) + ')')

# ================= 输出 =================
pd.DataFrame(ROWS).to_csv(os.path.join(OUTDIR, 'e5_results.csv'), index=False,
                          encoding='utf-8-sig')
lines = ['# E5 顺手件报告', '',
         '- 生成: 2026-09-30 | 脚本: 01_analysis_current/e5_odds_and_ends.py', '',
         '## 结果', '',
         '| 检验 | 统计量 | p |', '|---|---|---|']
for r in ROWS:
    lines.append(f"| {r['test']} ({r['outcome']}) | {r['stat']:.4f} | {r['p']:.4f} |")
lines += ['', '## 备注', ''] + ['- ' + str(n) for n in NOTES]
with open(os.path.join(OUTDIR, 'e5_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(NOTES))
print('\n完成 -> e5_results.csv / e5_report.md')
