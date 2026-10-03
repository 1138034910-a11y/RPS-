# -*- coding: utf-8 -*-
"""
e2_assignment_dose.py
E2: 剂量倒置与普通年份分配检验(最小充分矩阵之二)
==================================================
回应审稿意见(见 改投准备/review_eiar/DECISION.md):
  R4-C1 (CRITICAL): 逐年交互 2019(+0.476)/2023(+0.521) > 2024(+0.242), 与"软年份无物理
      压力"的一阶结果矛盾——剂量倒置必须正面处理
  R1-缺点1: 普通年份增量若"按可达成性校准", 就是按预期消纳能力分配——分配内生性
  R1-缺点6: 2019 年增量构造(基期)从未写明, 而 2019 交互恰恰显著
  R2-缺点8: "约束被缓冲稀释" vs "约束本就未binding" 两种叙述并存, 需判别事实

子实验(预注册判定标准, 跑前写明):
  E2a 普通年份(2019-2023)分配检验: Δweight ~ 滞后弃风水平/滞后弃风变化/滞后消纳率
      判定: 普通年份增量与滞后弃风显著相关 -> 需 E2e 残差化兜底
  E2b 缓冲说 vs 跟随说判别: Δweight ~ 滞后可达余量(前期实际消纳率-前期目标)
      判定: 正相关=跟随说(按能力分配), 不相关=权重有独立推力
  E2c 2019 增量构造文档化: 打印 2018/2019 权重原值, 确认基期真实存在
      判定: 2018 值全 29 省非空 -> 构造透明(但其来源需在论文SI写明)
  E2d 剔 2019 年 T2c 稳健性(2020-2024)
      判定: β2 落在 [0.18, 0.30] 且 p<0.05 -> 通过
  E2e 残差化处理变体: Δweight 对滞后变量回归取残差(="超预期增量"), 残差×lock-in 重跑 T2c
      判定: β2 同号且显著 -> 分配内生性封住; 若死掉 -> 按决策树情形C纪律, 不硬救

回归框架与 redesign_analysis_v8d.py 严格同构(样本/控制/聚类/wild bootstrap 999)。
基准锚点(自检): T2c 全样本 β2_heat ≈ 0.229。
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
d18 = df[df['year'].between(2018, 2024)].copy()
d18['dweight'] = d18.groupby('province_en')['nonhydro_weight_binding'].diff()

# 滞后变量(省内滞后一年)
for v in ['wind_curtailment_pct', 'actual_nonhydro_rate', 'wind_cap_growth_pct',
          'consumption_growth_pct']:
    d18['L1_' + v] = d18.groupby('province_en')[v].shift(1)
# 滞后弃风变化: 先算省内逐年 diff, 再滞后(2019 年需 2017 值, 数据从2018起 -> 2019 为空, 守卫处理)
d18['dcurt'] = d18.groupby('province_en')['wind_curtailment_pct'].diff()
d18['L1_dcurt'] = d18.groupby('province_en')['dcurt'].shift(1)
# 滞后可达余量 = 前期实际消纳率 - 前期约束目标(余量越大, 当年目标越容易达成)
d18['slack'] = d18['actual_nonhydro_rate'] - d18['nonhydro_weight_binding']
d18['L1_slack'] = d18.groupby('province_en')['slack'].shift(1)

# lock-in(2023 基年, 论文口径: 同名单位相除, 均值0.460)
p23 = df[df['year'] == 2023].set_index('province_en')
heat = (p23['yb_heat_supply_capacity_mw'].fillna(0) /
        p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna()).rename('heat_rigidity')

def zmap_on(d, s):
    """z 标准化: 矩取自分析样本的映射列(与管线一致), 返回省级映射 dict"""
    col = d['province_en'].map(s)
    return ((s - col.mean()) / col.std()).to_dict()

# ---------------- 相关/回归工具 ----------------
def corr_row(a, b, d, tag):
    sub = d[[a, b]].dropna()
    if len(sub) < 3:
        WARNINGS.append(f'{tag}: 有效观测不足(n={len(sub)}), 跳过')
        CORR_ROWS.append(dict(test=tag, var_a=a, var_b=b, n=len(sub), r=np.nan, p=np.nan))
        return np.nan, np.nan
    r, p = sstats.pearsonr(sub[a], sub[b])
    CORR_ROWS.append(dict(test=tag, var_a=a, var_b=b, n=len(sub), r=r, p=p))
    return r, p

def demean_year(d, v):
    return d[v] - d.groupby('year')[v].transform('mean')

# ================= E2a 普通年份分配检验 =================
print('== E2a 普通年份分配检验 ==')
ordinary = d18[d18['year'].between(2019, 2023)].copy()
# 年内去均值后的合并相关(消除年度共同冲击)
ordinary['dw_dm'] = demean_year(ordinary, 'dweight')
for lv, lab in [('L1_wind_curtailment_pct', 'lag_curt_level'),
                ('L1_dcurt', 'lag_curt_change'),
                ('L1_actual_nonhydro_rate', 'lag_nonhydro_rate')]:
    ordinary[lv + '_dm'] = demean_year(ordinary, lv)
    corr_row('dw_dm', lv + '_dm', ordinary, f'E2a_pooled_ordinary_2019_2023|{lab}')
    for y in range(2019, 2024):
        dy = ordinary[ordinary['year'] == y]
        corr_row('dweight', lv, dy, f'E2a_year{y}|{lab}')
# 聚类 OLS 版(合并, 年份FE)
dreg = ordinary.dropna(subset=['dweight', 'L1_wind_curtailment_pct',
                               'L1_actual_nonhydro_rate', 'L1_dcurt'])
m_a = smf.ols('dweight ~ L1_wind_curtailment_pct + L1_dcurt + L1_actual_nonhydro_rate + C(year)',
              data=dreg).fit(cov_type='cluster', cov_kwds={'groups': dreg['province_en']})
for v in ['L1_wind_curtailment_pct', 'L1_dcurt', 'L1_actual_nonhydro_rate']:
    REG_ROWS.append(dict(test='E2a_ols_ordinary', coef=v, beta=m_a.params[v],
                         se=m_a.bse[v], p=m_a.pvalues[v], n=len(dreg)))

# ================= E2b 缓冲说 vs 跟随说 =================
print('== E2b 判别事实 ==')
all_t = d18[d18['year'].between(2019, 2024)].copy()
all_t['dw_dm'] = demean_year(all_t, 'dweight')
all_t['L1_slack_dm'] = demean_year(all_t, 'L1_slack')
corr_row('dw_dm', 'L1_slack_dm', all_t, 'E2b_pooled_2019_2024|lag_slack')
ordinary['L1_slack_dm'] = demean_year(ordinary, 'L1_slack')
corr_row('dw_dm', 'L1_slack_dm', ordinary, 'E2b_pooled_ordinary|lag_slack')
dreg2 = all_t.dropna(subset=['dweight', 'L1_slack'])
m_b = smf.ols('dweight ~ L1_slack + C(year)', data=dreg2).fit(
    cov_type='cluster', cov_kwds={'groups': dreg2['province_en']})
REG_ROWS.append(dict(test='E2b_ols_allyears', coef='L1_slack', beta=m_b.params['L1_slack'],
                     se=m_b.bse['L1_slack'], p=m_b.pvalues['L1_slack'], n=len(dreg2)))

# ================= E2c 2019 增量构造文档化 =================
print('== E2c 2019 增量构造 ==')
w_wide = d18.pivot_table(index='province_en', columns='year',
                         values='nonhydro_weight_binding')
cov = w_wide.notna().sum().to_dict()
dw2019 = (w_wide[2019] - w_wide[2018]).describe()
WARNINGS.append(f'E2c nonhydro_weight_binding 年度覆盖(省数): {cov}')
WARNINGS.append(f'E2c dweight_2019 = w2019 - w2018: mean={dw2019["mean"]:.3f}, '
                f'min={dw2019["min"]:.3f}, max={dw2019["max"]:.3f}; '
                f'2018基期29省全非空={w_wide[2018].notna().all()}')

# ================= 回归框架(T2c, 与管线同构) =================
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

def run_t2c(d, treat_col, tag, heat_col='dw_heat', bootstrap=True):
    """T2c: outcome ~ treat×heat + treat + CTRL_B + FE; 返回交互项记录"""
    d = d.dropna(subset=['wind_curtailment_pct', treat_col, heat_col] + CTRL_B).copy()
    n_obs, n_prov = len(d), d['province_en'].nunique()
    rhs = f'{heat_col} + {treat_col} + ' + ' + '.join(CTRL_B) + ' + C(province_en) + C(year)'
    m = smf.ols(f'wind_curtailment_pct ~ {rhs}', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['province_en']})
    beta, se, p = m.params[heat_col], m.bse[heat_col], m.pvalues[heat_col]
    pb = wild_boot_p(d, 'wind_curtailment_pct', heat_col,
                     [treat_col] + CTRL_B, beta, se) if bootstrap else np.nan
    rec = dict(test=tag, coef=heat_col, beta=beta, se=se, p_cluster=p, p_wildboot=pb,
               n_obs=n_obs, n_prov=n_prov)
    REG_ROWS.append(rec)
    print(f'{tag}: beta2={beta:.4f}, p={p:.4f}, wildboot={pb:.4f}, n={n_obs}')
    return rec

# T2c 样本与交互项
d2 = d18[d18['year'].between(2019, 2024)].copy()
hz = zmap_on(d2, heat)
d2['heat_z'] = d2['province_en'].map(hz)
d2['dw_heat'] = d2['dweight'] * d2['heat_z']

# 自检锚点: 全样本基准
bench = run_t2c(d2, 'dweight', 'T2c_bench_fullsample(anchor)')
anchor_ok = abs(bench['beta'] - 0.229) < 0.01
WARNINGS.append(f"自检锚点: bench β2={bench['beta']:.4f} vs 0.229 -> {'通过' if anchor_ok else '⚠️不通过'}")

# ================= E2d 剔 2019 =================
print('== E2d 剔2019 ==')
d2_no19 = d2[d2['year'] >= 2020].copy()
hz19 = zmap_on(d2_no19, heat)   # 子样本重新 z 化(与管线惯例一致: 矩取自分析样本)
d2_no19['heat_z'] = d2_no19['province_en'].map(hz19)
d2_no19['dw_heat'] = d2_no19['dweight'] * d2_no19['heat_z']
e2d = run_t2c(d2_no19, 'dweight', 'T2c_drop2019')

# ================= E2e 残差化处理变体 =================
print('== E2e 残差化处理 ==')
lag_vars = ['L1_wind_curtailment_pct', 'L1_actual_nonhydro_rate',
            'L1_wind_cap_growth_pct', 'L1_consumption_growth_pct']
dres = d2.dropna(subset=['dweight'] + lag_vars).copy()
m_res = smf.ols('dweight ~ ' + ' + '.join(lag_vars) + ' + C(year)', data=dres).fit()
d2['dweight_res'] = np.nan
d2.loc[dres.index, 'dweight_res'] = dres['dweight'] - m_res.fittedvalues
WARNINGS.append(f'E2e 残差化回归 R2={m_res.rsquared:.3f} (Δweight 被滞后变量解释的比例)')
d2e = d2.dropna(subset=['dweight_res']).copy()
hze = zmap_on(d2e, heat)
d2e['heat_z'] = d2e['province_en'].map(hze)
d2e['dw_heat'] = d2e['dweight_res'] * d2e['heat_z']
e2e = run_t2c(d2e, 'dweight_res', 'T2c_residualized_dweight')

# ================= 判定与输出 =================
verdicts = []
v_a = '见 e2_correlations'  # E2a 无单一阈值, 判断写入报告
e2d_ok = (0.18 <= e2d['beta'] <= 0.30) and (e2d['p_cluster'] < 0.05)
verdicts.append(f"E2d 剔2019: β2={e2d['beta']:.3f}, p={e2d['p_cluster']:.4f} -> "
                + ('通过(落在[0.18,0.30]且显著)' if e2d_ok else '⚠️未通过预注册标准'))
e2e_ok = (e2e['beta'] > 0) and (e2e['p_cluster'] < 0.05)
verdicts.append(f"E2e 残差化: β2={e2e['beta']:.3f}, p={e2e['p_cluster']:.4f} -> "
                + ('通过(分配内生性封住)' if e2e_ok else '⚠️未通过——按决策树情形C纪律处理, 不硬救'))

pd.DataFrame(CORR_ROWS).to_csv(os.path.join(OUTDIR, 'e2_correlations.csv'),
                               index=False, encoding='utf-8-sig')
pd.DataFrame(REG_ROWS).to_csv(os.path.join(OUTDIR, 'e2_regressions.csv'),
                              index=False, encoding='utf-8-sig')

lines = ['# E2 剂量倒置与普通年份分配检验报告', '',
         '- 生成: 2026-09-30 | 脚本: 01_analysis_current/e2_assignment_dose.py',
         f"- 自检锚点: bench β2={bench['beta']:.4f} (锚点 0.229) -> {'通过' if anchor_ok else '⚠️不通过'}", '',
         '## 判定汇总', ''] + ['- ' + v for v in verdicts] + [
         '', '## 1. 相关检验(含 p 值)', '',
         '| 检验 | 变量对 | r | p | n |', '|---|---|---|---|---|']
for c in CORR_ROWS:
    lines.append(f"| {c['test']} | {c['var_a']} × {c['var_b']} | {c['r']:+.3f} | {c['p']:.4f} | {c['n']} |")
lines += ['', '## 2. 回归结果', '',
          '| 检验 | 系数 | β | se | p(cluster) | p(wildboot) | n |', '|---|---|---|---|---|---|---|']
for r in REG_ROWS:
    pb = '' if pd.isna(r.get('p_wildboot')) else f"{r['p_wildboot']:.4f}"
    se = f"{r['se']:.4f}" if not pd.isna(r.get('se')) else ''
    lines.append(f"| {r['test']} | {r['coef']} | {r['beta']:.4f} | {se} | "
                 f"{r.get('p_cluster', r.get('p')):.4f} | {pb} | {r.get('n_obs', r.get('n'))} |")
lines += ['', '## 3. 文档化与警告', ''] + ['- ' + w for w in WARNINGS]
with open(os.path.join(OUTDIR, 'e2_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(verdicts))
print(f'\n完成: {len(CORR_ROWS)} 条相关 + {len(REG_ROWS)} 条回归 -> e2_correlations.csv / e2_regressions.csv / e2_report.md')
