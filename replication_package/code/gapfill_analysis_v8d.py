# -*- coding: utf-8 -*-
"""
gapfill_analysis_v8.py
实证缺口补全 (数据: master_panel_v8d.csv)
  G1 缺口1-第一阶: Δweight/ΔTarget 是否推高消纳率(预期边际) — 处理变量有效性验证
  G2 缺口2-混杂排除: heat_rigidity 是否与冲击前弃风趋势相关(交互项捡趋势的假说检验)
  G3 缺口3-结构零: 纯供暖省子样本(剔除4个无供暖省)重跑 T2c/T1 — 南北气候混淆的正面回应
  G4 缺口4-多重检验: Romano-Wolf 单步校正 (T2c弃风×4调节变量族, 联合wild bootstrap max-t)
  G5 缺口6-替代度量: 供热量/火电发电量 版 heat_rigidity 重跑 T2c
推断口径与 redesign_analysis_v8.py 完全一致: TWFE(省+年FE), 省聚类SE, wild cluster bootstrap
(FWL, Rademacher, 999次), 退化防护。
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os, warnings, json

warnings.filterwarnings('ignore')
BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
np.random.seed(20260724)
NBOOT = 999

RESULTS, DEGENERATE, WARNINGS = [], [], []

CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']
CTRL_FS = ['wind_cap_growth_pct', 'solar_cap_growth_pct']  # 第一阶: 去掉消费增速(与结果机械相关)

# ---------------- 数据准备 (与redesign一致, 换v8c) ----------------
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)

d18 = df[df['year'].between(2018, 2024)].copy()
d18 = d18[~d18['province_en'].isin(['Tibet', 'Xinjiang'])].copy()
d18['post'] = (d18['year'] >= 2024).astype(int)
d18['dt_post'] = d18['delta_target_i'] * d18['post']

p23 = df[df['year'] == 2023].set_index('province_en')
weak = pd.DataFrame(index=p23.index)
weak['weak_export'] = -p23['export_ratio_pct']
heat_mw = p23['yb_heat_supply_capacity_mw'] * 10.0
weak['heat_rigidity'] = (heat_mw.fillna(0) / p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())
weak['resource_vol'] = p23['wind_cf_cv']
weak['pipeline_pressure'] = (p23['yb_power_project_under_construction_cap10kw'] * 10.0 /
                             p23['installed_capacity_mw_total'])
# G5 替代度量: 供热量(GJ)/火电装机(万千瓦), 无供暖=结构零
# 注: electricity_generation_gwh_thermal 2023年全缺失, 分母改用火电装机(与主度量同量纲族, 供热"强度"口径)
weak['heat_rigidity_gj'] = (p23['yb_heat_supply_gj'].fillna(0) /
                            p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())
WEAK_VARS = ['weak_export', 'heat_rigidity', 'resource_vol', 'pipeline_pressure']

# 时变强度面板
d_full = df[df['year'].between(2018, 2024)].copy()
d_full = d_full[~d_full['province_en'].isin(['Tibet', 'Xinjiang'])].copy()
d_full['dweight'] = d_full.groupby('province_en')['nonhydro_weight_binding'].diff()
d_t2 = d_full[d_full['year'].between(2019, 2024)].copy()
for w in WEAK_VARS + ['heat_rigidity_gj']:
    col = d_t2['province_en'].map(weak[w])
    d_t2[w + '_z'] = (col - col.mean()) / col.std()
for w in WEAK_VARS:
    col = d18['province_en'].map(weak[w])
    d18[w + '_z'] = (col - col.mean()) / col.std()

# ---------------- 回归框架 (复制redesign口径) ----------------
def collinearity_precheck(d, controls, tag):
    kept, dropped = [], []
    for c in controls:
        if c not in d.columns or d[c].notna().sum() == 0:
            dropped.append((c, 'all-missing')); continue
        if d[c].std(skipna=True) < 1e-12:
            dropped.append((c, 'zero-variance')); continue
        kept.append(c)
    if kept:
        sub = d[kept].dropna()
        if len(sub) > 0:
            X = np.column_stack([np.ones(len(sub)), sub.values])
            while len(kept) > 1 and np.linalg.matrix_rank(X) < X.shape[1]:
                worst = kept[-1]; kept = kept[:-1]
                dropped.append((worst, 'perfect-collinearity'))
                sub = d[kept].dropna()
                X = np.column_stack([np.ones(len(sub)), sub.values]) if kept else np.ones((len(sub), 1))
    for c, why in dropped:
        WARNINGS.append(f"[{tag}] 控制变量 {c} 剔除: {why}")
    return kept

def is_degenerate(beta, se):
    return abs(beta) < 1e-6 and se < 1e-10

def wild_boot_p(d, outcome, target, other_x, beta_obs, se_obs, n_boot=NBOOT):
    provs = d['province_en'].values
    uniq = np.unique(provs)
    years = d['year'].values
    uy = np.unique(years)
    M = [np.ones(len(d))]
    for g in uniq[1:]:
        M.append((provs == g).astype(float))
    for yy in uy[1:]:
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
        ys_t = absorb(y_star)
        tb = cluster_t(ys_t)
        if not np.isfinite(tb) or abs(tb) >= t_obs:
            cnt += 1
    return (cnt + 1) / (n_boot + 1)

def run_reg(d, outcome, target, rhs_others, controls, eq, spec, weakabs='', form='',
            coef_label=None, bootstrap=True):
    d = d.copy()
    ctrls = collinearity_precheck(d, controls, f'{eq}|{spec}')
    need = [outcome, target, 'province_en', 'year'] + rhs_others + ctrls
    d = d.dropna(subset=[c for c in need if c in d.columns])
    n_obs, n_prov = len(d), d['province_en'].nunique()
    if d[target].var() < 1e-14:
        WARNINGS.append(f'[{eq}|{spec}] 跳过: {target} 样本内方差≈0 (n={n_obs})'); return None
    if n_prov < 5 or n_obs < 20:
        WARNINGS.append(f'[{eq}|{spec}] 跳过: 样本过小 n={n_obs}, prov={n_prov}'); return None
    rhs = ' + '.join([target] + rhs_others + ctrls) + ' + C(province_en) + C(year)'
    try:
        m = smf.ols(f'{outcome} ~ {rhs}', data=d).fit(
            cov_type='cluster', cov_kwds={'groups': d['province_en']})
    except Exception as e:
        WARNINGS.append(f'[{eq}|{spec}] 回归失败: {e}'); return None
    if target not in m.params.index:
        WARNINGS.append(f'[{eq}|{spec}] {target} 被FE吸收, 跳过'); return None
    beta, se, p = m.params[target], m.bse[target], m.pvalues[target]
    if is_degenerate(beta, se):
        DEGENERATE.append(f'[{eq}|{spec}] {outcome}~{target} beta={beta:.2e} se={se:.2e} 退化丢弃')
        return None
    p_boot = np.nan
    if bootstrap:
        try:
            p_boot = wild_boot_p(d, outcome, target, rhs_others + ctrls, beta, se)
        except Exception as e:
            WARNINGS.append(f'[{eq}|{spec}] bootstrap失败: {e}')
    rec = dict(eq=eq, spec=spec, outcome=outcome, weakabs=weakabs, form=form,
               coef=coef_label or target, beta=beta, se_cluster=se, p_cluster=p,
               p_wildboot=p_boot, n_obs=n_obs, n_prov=n_prov)
    RESULTS.append(rec)
    return rec

# ================= G1 第一阶: 目标冲击是否推高消纳率 =================
print('== G1 第一阶 ==')
# G1a 全面板: rate ~ dweight (2019-2024)
for rate in ['actual_nonhydro_rate', 'actual_total_rate']:
    run_reg(d_t2, rate, 'dweight', [], CTRL_FS, 'G1a', f'firststage_panel|{rate}',
            coef_label='beta_dweight')
# G1b 单冲击: rate ~ dt_post (2018-2024)
for rate in ['actual_nonhydro_rate', 'actual_total_rate']:
    run_reg(d18, rate, 'dt_post', [], CTRL_FS, 'G1b', f'firststage_shock|{rate}',
            coef_label='beta_dtPost')
# G1c 合规margin: 达标哑变量 met_nonhydro ~ dweight (LPM)
run_reg(d_t2, 'met_nonhydro', 'dweight', [], CTRL_FS, 'G1c', 'firststage_panel|met_nonhydro',
        coef_label='beta_dweight')

# ================= G2 heat_rigidity 与预处理趋势 =================
print('== G2 rigidity vs pretrend ==')
rows = []
for p, g in d18.groupby('province_en'):
    g = g.sort_values('year')
    pre = g[g['year'] <= 2022]
    slope_w = np.polyfit(pre['year'], pre['wind_curtailment_pct'], 1)[0] if len(pre) >= 3 else np.nan
    rows.append(dict(province_en=p, wind_pretrend_slope=slope_w,
                     heat_rigidity=weak.loc[p, 'heat_rigidity'] if p in weak.index else np.nan,
                     delta_target=g['delta_target_i'].iloc[0]))
g2 = pd.DataFrame(rows).dropna()
g2.to_csv(os.path.join(BASE, 'gapfill_g2_pretrend_v8d.csv'), index=False, encoding='utf-8-sig')
c_rig = g2['wind_pretrend_slope'].corr(g2['heat_rigidity'])
c_dt = g2['wind_pretrend_slope'].corr(g2['delta_target'])
c_rig_dt = g2['heat_rigidity'].corr(g2['delta_target'])
m_g2 = smf.ols('wind_pretrend_slope ~ heat_rigidity', data=g2).fit()
print(f'corr(弃风pre-trend, heat_rigidity) = {c_rig:.3f} (slope回归p={m_g2.pvalues["heat_rigidity"]:.3f})')
print(f'corr(弃风pre-trend, ΔTarget) = {c_dt:.3f}')
print(f'corr(heat_rigidity, ΔTarget) = {c_rig_dt:.3f}')
G2_STATS = dict(corr_pretrend_rigidity=float(c_rig), p_rigidity=float(m_g2.pvalues['heat_rigidity']),
                corr_pretrend_dtarget=float(c_dt), corr_rigidity_dtarget=float(c_rig_dt))

# ================= G3 纯供暖省子样本 =================
print('== G3 heating-only subsample ==')
heat_provs = weak[weak['heat_rigidity'] > 0].index.tolist()
print(f'供暖省数: {len(heat_provs)}')
d_t2h = d_t2[d_t2['province_en'].isin(heat_provs)].copy()
col = d_t2h['province_en'].map(weak['heat_rigidity'])
d_t2h['heat_z_sub'] = (col - col.mean()) / col.std()
d_t2h['dw_weak'] = d_t2h['dweight'] * d_t2h['heat_z_sub']
run_reg(d_t2h.dropna(subset=['heat_z_sub']), 'wind_curtailment_pct', 'dw_weak', ['dweight'],
        CTRL_B, 'G3a', 'T2c_heating_only|wind', weakabs='heat_rigidity', form='continuous',
        coef_label='beta2_triple')
d18h = d18[d18['province_en'].isin(heat_provs)].copy()
col = d18h['province_en'].map(weak['heat_rigidity'])
d18h['heat_z_sub'] = (col - col.mean()) / col.std()
d18h['treat_weak'] = d18h['dt_post'] * d18h['heat_z_sub']
run_reg(d18h.dropna(subset=['heat_z_sub']), 'wind_curtailment_pct', 'treat_weak', ['dt_post'],
        CTRL_B, 'G3b', 'T1_heating_only|wind', weakabs='heat_rigidity', form='continuous',
        coef_label='beta2_triple')

# ================= G4 Romano-Wolf (T2c弃风×4调节变量族) =================
print('== G4 Romano-Wolf ==')
def build_model(d, outcome, target, other_x):
    d = d.dropna(subset=[outcome, target] + other_x).copy()
    provs = d['province_en'].values
    uniq = np.unique(provs)
    years = d['year'].values
    uy = np.unique(years)
    M = [np.ones(len(d))]
    for g in uniq[1:]:
        M.append((provs == g).astype(float))
    for yy in uy[1:]:
        M.append((years == yy).astype(float))
    M = np.column_stack(M)
    C = d[other_x].values.astype(float) if other_x else np.zeros((len(d), 0))
    y = d[outcome].values.astype(float)
    X = np.column_stack([d[target].values.astype(float), C])
    def absorb(v, M=M):
        b, *_ = np.linalg.lstsq(M, v, rcond=None)
        return v - M @ b
    X_t = np.column_stack([absorb(X[:, j]) for j in range(X.shape[1])])
    XtX_inv = np.linalg.pinv(X_t.T @ X_t)
    N, K = len(d), M.shape[1] + X.shape[1]
    G = len(uniq)
    # 观测t
    beta = XtX_inv @ (X_t.T @ absorb(y))
    e = absorb(y) - X_t @ beta
    meat = np.zeros((X_t.shape[1], X_t.shape[1]))
    for g in uniq:
        idx = provs == g
        u = X_t[idx].T @ e[idx]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    V *= (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.maximum(np.diag(V), 0))
    t_obs = abs(beta[0] / se[0])
    # 受限(H0: beta_target=0)拟合
    MR = np.column_stack([M, C]) if C.shape[1] else M
    br, *_ = np.linalg.lstsq(MR, y, rcond=None)
    fitted_r, resid_r = MR @ br, y - MR @ br
    def t_boot(y_star):
        ys = absorb(y_star)
        b = XtX_inv @ (X_t.T @ ys)
        ee = ys - X_t @ b
        meat2 = np.zeros((X_t.shape[1], X_t.shape[1]))
        for g in uniq:
            idx = provs == g
            u = X_t[idx].T @ ee[idx]
            meat2 += np.outer(u, u)
        V2 = XtX_inv @ meat2 @ XtX_inv
        V2 *= (G / (G - 1)) * ((N - 1) / (N - K))
        se2 = np.sqrt(np.maximum(np.diag(V2), 0))
        return b[0] / se2[0] if se2[0] > 0 else np.inf
    return dict(provs=provs, uniq=uniq, fitted_r=fitted_r, resid_r=resid_r,
                t_obs=t_obs, t_boot=t_boot)

models = {}
for w in WEAK_VARS:
    dd = d_t2.dropna(subset=[w + '_z']).copy()
    dd['dw_weak'] = dd['dweight'] * dd[w + '_z']
    models[w] = build_model(dd, 'wind_curtailment_pct', 'dw_weak', ['dweight'] + CTRL_B)
t_max = np.zeros(NBOOT)
t_mat = np.zeros((NBOOT, len(WEAK_VARS)))
all_provs = np.unique(np.concatenate([m['uniq'] for m in models.values()]))
for b in range(NBOOT):
    wmap = dict(zip(all_provs, np.random.choice([-1.0, 1.0], size=len(all_provs))))
    for j, wv in enumerate(WEAK_VARS):
        mdl = models[wv]
        y_star = mdl['fitted_r'] + mdl['resid_r'] * np.array([wmap[g] for g in mdl['provs']])
        tb = mdl['t_boot'](y_star)
        t_mat[b, j] = tb if np.isfinite(tb) else np.inf
    t_max[b] = np.max(np.abs(t_mat[b, :]))
rw_rows = []
for j, wv in enumerate(WEAK_VARS):
    t_obs = models[wv]['t_obs']
    p_rw = (np.sum(t_max >= t_obs) + 1) / (NBOOT + 1)
    rw_rows.append(dict(eq='G4', spec='romano_wolf|T2c_wind', outcome='wind_curtailment_pct',
                        weakabs=wv, form='continuous', coef='beta2_triple', beta=np.nan,
                        se_cluster=np.nan, p_cluster=np.nan, p_wildboot=np.nan,
                        p_rw=p_rw, t_obs=t_obs, n_obs=145, n_prov=29))
    print(f'  {wv}: t_obs={t_obs:.2f}, Romano-Wolf p={p_rw:.4f}')
RESULTS.extend(rw_rows)

# ================= G5 替代度量 rigidity (供热量/火电发电量) =================
print('== G5 alt rigidity (GJ-based) ==')
dd = d_t2.dropna(subset=['heat_rigidity_gj_z']).copy()
dd['dw_weak'] = dd['dweight'] * dd['heat_rigidity_gj_z']
run_reg(dd, 'wind_curtailment_pct', 'dw_weak', ['dweight'], CTRL_B, 'G5',
        'T2c_alt_gj|wind', weakabs='heat_rigidity_gj', form='continuous',
        coef_label='beta2_triple')

# ================= 输出 =================
res = pd.DataFrame(RESULTS)
res.to_csv(os.path.join(BASE, 'gapfill_results_v8d.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(BASE, '_gapfill_meta_v8d.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(G2=G2_STATS, heat_provs=heat_provs, n_degenerate=len(DEGENERATE),
                   warnings=WARNINGS, degenerate=DEGENERATE), f, ensure_ascii=False, indent=1)
print(f'\n完成: {len(res)}条记录, {len(DEGENERATE)}个退化丢弃, {len(WARNINGS)}条警告')
print(res[['eq', 'spec', 'weakabs', 'coef', 'beta', 'se_cluster', 'p_cluster', 'p_wildboot',
           'n_obs', 'n_prov']].to_string())
