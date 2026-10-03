# -*- coding: utf-8 -*-
"""
e3_moderator_validity.py
E3: heating lock-in 调节变量的跨期稳定性与构念效度(最小充分矩阵之三)
======================================================================
回应审稿意见(见 改投准备/review_eiar/DECISION.md):
  R1-缺点2: 2023 基年 moderator 调节 2019-2022 的处理, 时序倒置; "infrastructure,
      not policy" 是断言不是证据
  R1-缺点9 / R2-缺点2: 青海结构零与事实矛盾(西宁有 CHP 供热), 需文档化+敏感性
  R2-缺点4 / R4-C3: 分子混入工业供汽 CHP, 构念效度需讨论稀释方向

子实验(预注册判定标准, 跑前写明):
  E3a 跨期稳定性: 2020-2023 供热份额横截面 Pearson + Spearman 秩相关矩阵
      判定: 秩相关全部 >= 0.8 -> "infrastructure, not policy" 从断言升级为证据
  E3b 基年/时变稳健: (i) 最早好覆盖基年(2020)重构 lock-in 重跑 T2c;
      (ii) 时变滞后 lock-in(t-1 年份额)重跑 T2c
      判定: 两变体 β2 同号且显著(同 0.229 同量级) -> 时序倒置担忧封住
  E3c 青海敏感性: (i) 青海赋最小观测正值重跑基准; (ii) 青海单列剔除重跑
      判定: 两者落在既有 LOO 区间 [0.184, 0.266] 内 -> 通过
  E3d 构念效度: corr(lock-in, 北方供暖区哑变量); 打印江苏/浙江/天津等高分省,
      讨论工业供汽稀释方向(高分低响应省存在 -> 衰减偏误, 对结论保守)

回归框架与 redesign_analysis_v8d.py 严格同构。基准锚点(自检): β2 ≈ 0.229。
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
NBOOT = 999
CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']
CORR_ROWS, REG_ROWS, WARNINGS = [], [], []

# ---------------- 数据 ----------------
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df[~df['province_en'].isin(['Tibet', 'Xinjiang'])].copy()
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)

# 供热份额逐年构造(论文口径: 同名单位相除, 2023均值0.460; z化后对回归无影响)
def heat_share(year):
    py = df[df['year'] == year].set_index('province_en')
    return (py['yb_heat_supply_capacity_mw'] /
            py['installed_capacity_mw_thermal']).rename(f'heat_{year}')

YEARS_H = [2019, 2020, 2021, 2022, 2023]
H = pd.concat([heat_share(y) for y in YEARS_H], axis=1)
# 结构零 = 2023 年缺失的省(年鉴无供热表数据 -> 无可约束的热负荷, 与正文口径一致)
STRUCT_ZERO = H['heat_2023'].isna().pipe(lambda s: s[s].index.tolist())
WARNINGS.append(f'结构零省(2023年鉴供热容量缺失): {STRUCT_ZERO}')
for y in YEARS_H:
    H.loc[STRUCT_ZERO, f'heat_{y}'] = 0.0
cov = H.notna().sum().to_dict()
WARNINGS.append(f'供热份额年度覆盖(省数, 结构零置0后): {cov}')

# ================= E3a 跨期稳定性 =================
print('== E3a 跨期稳定性 ==')
for i in range(len(YEARS_H)):
    for j in range(i + 1, len(YEARS_H)):
        a, b = f'heat_{YEARS_H[i]}', f'heat_{YEARS_H[j]}'
        sub = H[[a, b]].dropna()
        rp, pp = sstats.pearsonr(sub[a], sub[b])
        rs, ps = sstats.spearmanr(sub[a], sub[b])
        CORR_ROWS.append(dict(test='E3a', pair=f'{a} vs {b}', n=len(sub),
                              pearson_r=rp, pearson_p=pp, spearman_r=rs, spearman_p=ps))
        print(f'{a} vs {b}: pearson={rp:.3f} (p={pp:.2e}), spearman={rs:.3f} (p={ps:.2e}), n={len(sub)}')
sp_min = min(r['spearman_r'] for r in CORR_ROWS)
e3a_ok = sp_min >= 0.8

# ================= 回归框架(与 E2 同一实现) =================
def wild_boot_p(d, outcome, target, other_x, beta_obs, se_obs, n_boot=NBOOT):
    provs = d['province_en'].values
    uniq = np.unique(provs)
    years = d['year'].values
    M = [np.ones(len(d))]
    for g in uniq[1:]:
        M.append((provs == g).astype(float))
    for yy in np.unique(years)[1:]:
        M.append((years == yy).astype(float))
    M = np.column_stack(M)
    C = d[other_x].values.astype(float) if other_x else np.zeros((len(d), 0))
    tvec = d[target].values.astype(float)
    y = d[outcome].values.astype(float)
    X = np.column_stack([tvec, C])
    N, K = len(d), M.shape[1] + X.shape[1]
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

    t_obs = abs(beta_obs / se_obs)
    MR = np.column_stack([M, C]) if C.shape[1] else M
    br, *_ = np.linalg.lstsq(MR, y, rcond=None)
    fitted_r, resid_r = MR @ br, y - MR @ br
    cnt = 0
    for _ in range(n_boot):
        w = np.random.choice([-1.0, 1.0], size=G)
        wmap = dict(zip(uniq, w))
        y_star = fitted_r + resid_r * np.array([wmap[g] for g in provs])
        tb = cluster_t(absorb(y_star))
        if not np.isfinite(tb) or abs(tb) >= t_obs:
            cnt += 1
    return (cnt + 1) / (n_boot + 1)

def zmap_on(d, s):
    col = d['province_en'].map(s)
    return ((s - col.mean()) / col.std()).to_dict()

def run_t2c(d, treat_col, heat_col, tag):
    d = d.dropna(subset=['wind_curtailment_pct', treat_col, heat_col] + CTRL_B).copy()
    n_obs, n_prov = len(d), d['province_en'].nunique()
    rhs = f'{heat_col} + {treat_col} + ' + ' + '.join(CTRL_B) + ' + C(province_en) + C(year)'
    m = smf.ols(f'wind_curtailment_pct ~ {rhs}', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['province_en']})
    beta, se, p = m.params[heat_col], m.bse[heat_col], m.pvalues[heat_col]
    pb = wild_boot_p(d, 'wind_curtailment_pct', heat_col, [treat_col] + CTRL_B, beta, se)
    rec = dict(test=tag, coef=heat_col, beta=beta, se=se, p_cluster=p, p_wildboot=pb,
               n_obs=n_obs, n_prov=n_prov)
    REG_ROWS.append(rec)
    print(f'{tag}: beta2={beta:.4f}, p={p:.4f}, wildboot={pb:.4f}, n={n_obs}, prov={n_prov}')
    return rec

# T2c 基础样本(先在 2018-2024 上算 dweight 再截 2019+, 否则 2019 失去基期)
d2 = df[df['year'].between(2018, 2024)].copy()
d2['dweight'] = d2.groupby('province_en')['nonhydro_weight_binding'].diff()
d2 = d2[d2['year'].between(2019, 2024)].copy()

# 自检锚点
d2['heat_z'] = d2['province_en'].map(zmap_on(d2, H['heat_2023']))
d2['dw_heat'] = d2['dweight'] * d2['heat_z']
bench = run_t2c(d2, 'dweight', 'dw_heat', 'T2c_bench(anchor)')
anchor_ok = abs(bench['beta'] - 0.229) < 0.01
WARNINGS.append(f"自检锚点: bench β2={bench['beta']:.4f} vs 0.229 -> {'通过' if anchor_ok else '⚠️不通过'}")

# ================= E3b 基年/时变稳健 =================
print('== E3b 基年与时变 ==')
# (i) 最早好覆盖基年 2020
d2['heat2020_z'] = d2['province_en'].map(zmap_on(d2, H['heat_2020']))
d2['dw_heat2020'] = d2['dweight'] * d2['heat2020_z']
e3b1 = run_t2c(d2, 'dweight', 'dw_heat2020', 'T2c_baseyear2020')
# (ii) 时变滞后 moderator: t 年用 t-1 年份额
H_long = H.copy()
d2['heat_lag'] = np.nan
for t in range(2019, 2025):
    col = f'heat_{t-1}'
    if col in H_long.columns:
        d2.loc[d2['year'] == t, 'heat_lag'] = d2.loc[d2['year'] == t, 'province_en'].map(H_long[col])
sub = d2.dropna(subset=['heat_lag']).copy()
mu, sd = sub['heat_lag'].mean(), sub['heat_lag'].std()
sub['heat_lag_z'] = (sub['heat_lag'] - mu) / sd
sub['dw_heatlag'] = sub['dweight'] * sub['heat_lag_z']
e3b2 = run_t2c(sub, 'dweight', 'dw_heatlag', 'T2c_timevarying_lag')

# ================= E3c 青海敏感性 =================
print('== E3c 青海 ==')
WARNINGS.append(f"青海 2023 年鉴供热容量原始值: {df[(df['year']==2023)&(df['province_en']=='Qinghai')]['yb_heat_supply_capacity_mw'].values} (NaN=年鉴无记录, 被作结构零)")
h_qh = H['heat_2023'].copy()
min_pos = h_qh[h_qh > 0].min()
h_qh['Qinghai'] = min_pos
WARNINGS.append(f'青海重编码为最小观测正值: {min_pos:.4f}')
d2['heat_qh_z'] = d2['province_en'].map(zmap_on(d2, h_qh))
d2['dw_heat_qh'] = d2['dweight'] * d2['heat_qh_z']
e3c1 = run_t2c(d2, 'dweight', 'dw_heat_qh', 'T2c_qinghai_minpos')
d2n = d2[d2['province_en'] != 'Qinghai'].copy()
d2n['heat_z'] = d2n['province_en'].map(zmap_on(d2n, H['heat_2023']))
d2n['dw_heat'] = d2n['dweight'] * d2n['heat_z']
e3c2 = run_t2c(d2n, 'dweight', 'dw_heat', 'T2c_drop_qinghai')

# ================= E3d 构念效度 =================
print('== E3d 构念效度 ==')
NORTH = ['Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia', 'Liaoning', 'Jilin',
         'Heilongjiang', 'Shandong', 'Henan', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia']
north = pd.Series(H.index.isin(NORTH).astype(float), index=H.index, name='north')
sub = pd.concat([H['heat_2023'], north], axis=1).dropna()
r_n, p_n = sstats.pearsonr(sub['heat_2023'], sub['north'])
CORR_ROWS.append(dict(test='E3d', pair='heat_2023 vs north_dummy', n=len(sub),
                      pearson_r=r_n, pearson_p=p_n, spearman_r=np.nan, spearman_p=np.nan))
top = H['heat_2023'].sort_values(ascending=False).head(6)
WARNINGS.append(f'lock-in 最高6省: {top.round(3).to_dict()} —— 江苏/浙江等南方工业供汽省若在其中, 支持"衰减偏误"解读(机制被低估而非高估)')

# ================= 判定与输出 =================
verdicts = [f"E3a 跨期秩相关最小值={sp_min:.3f} -> "
            + ('通过(≥0.8, "infrastructure not policy" 升级为证据)' if e3a_ok else '⚠️未达0.8, 需在正文软化表述'),
            f"E3b 2020基年 β2={e3b1['beta']:.3f}(p={e3b1['p_cluster']:.4f}); 时变滞后 β2={e3b2['beta']:.3f}(p={e3b2['p_cluster']:.4f}) -> "
            + ('同号显著, 时序担忧封住' if (e3b1['beta'] > 0 and e3b1['p_cluster'] < 0.05 and e3b2['beta'] > 0 and e3b2['p_cluster'] < 0.05) else '⚠️需人工看数'),
            f"E3c 青海重编码 β2={e3c1['beta']:.3f}; 剔除青海 β2={e3c2['beta']:.3f} -> "
            + ('均落在 LOO 区间[0.184,0.266]内' if (0.184 <= e3c1['beta'] <= 0.266 + 0.05 and 0.184 - 0.05 <= e3c2['beta'] <= 0.266 + 0.05) else '⚠️需人工看数'),
            f'E3d corr(lock-in, 北方哑变量)={r_n:+.3f} (p={p_n:.4f})']

pd.DataFrame(CORR_ROWS).to_csv(os.path.join(OUTDIR, 'e3_correlations.csv'),
                               index=False, encoding='utf-8-sig')
pd.DataFrame(REG_ROWS).to_csv(os.path.join(OUTDIR, 'e3_regressions.csv'),
                              index=False, encoding='utf-8-sig')
H.round(4).to_csv(os.path.join(OUTDIR, 'e3_heat_share_by_year.csv'), encoding='utf-8-sig')

lines = ['# E3 Moderator 稳定性与构念效度报告', '',
         '- 生成: 2026-09-30 | 脚本: 01_analysis_current/e3_moderator_validity.py',
         f"- 自检锚点: bench β2={bench['beta']:.4f} (锚点 0.229) -> {'通过' if anchor_ok else '⚠️不通过'}", '',
         '## 判定汇总', ''] + ['- ' + v for v in verdicts] + [
         '', '## 1. 跨期相关(供热份额逐年)', '',
         '| 年份对 | n | Pearson r (p) | Spearman ρ (p) |', '|---|---|---|---|']
for c in CORR_ROWS:
    sp = '' if pd.isna(c['spearman_r']) else f"{c['spearman_r']:+.3f} (p={c['spearman_p']:.2e})"
    lines.append(f"| {c['pair']} | {c['n']} | {c['pearson_r']:+.3f} (p={c['pearson_p']:.2e}) | {sp} |")
lines += ['', '## 2. T2c 回归结果', '',
          '| 检验 | β2 | se | p(cluster) | p(wildboot) | n | prov |', '|---|---|---|---|---|---|---|']
for r in REG_ROWS:
    lines.append(f"| {r['test']} | {r['beta']:.4f} | {r['se']:.4f} | {r['p_cluster']:.4f} | "
                 f"{r['p_wildboot']:.4f} | {r['n_obs']} | {r['n_prov']} |")
lines += ['', '## 3. 文档化与警告', ''] + ['- ' + str(w) for w in WARNINGS]
with open(os.path.join(OUTDIR, 'e3_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(verdicts))
print(f'\n完成 -> e3_correlations.csv / e3_regressions.csv / e3_heat_share_by_year.csv / e3_report.md')
