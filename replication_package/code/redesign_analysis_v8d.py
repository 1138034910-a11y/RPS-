# -*- coding: utf-8 -*-
"""
redesign_analysis_v8.py
RPS识别策略重做 (数据: master_panel_v8d.csv)
  任务一: 三重交互 ΔTarget×Post×WeakAbs (连续z + 中位数哑变量)
  任务二: 全面板时变强度 Δweight_it (静态 / lead-lag / 三重交互)
  任务三: 利用小时替换被解释变量 + 弃光率预处理趋势诊断
  任务四: 剂量-反应图
退化防护: 处理变量方差预检; |beta|<1e-6且se<1e-10剔除计数; 控制变量共线性预检。
推断: 省份聚类SE + 手动wild cluster bootstrap (FWL, Rademacher, 999次), 复用v8实现并推广到任意目标系数。
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
np.random.seed(20260724)
NBOOT = 999

RESULTS, DEGENERATE, WARNINGS = [], [], []

CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']

# ---------------- 数据准备 ----------------
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)

d18 = df[df['year'].between(2018, 2024)].copy()
d18 = d18[(d18['province_en'] != 'Tibet') & (d18['province_en'] != 'Xinjiang')].copy()
d18['post'] = (d18['year'] >= 2024).astype(int)
d18['dt_post'] = d18['delta_target_i'] * d18['post']

# ---------------- WeakAbs 指标 (2023年基期, 省份级时不变) ----------------
p23 = df[df['year'] == 2023].set_index('province_en')
weak = pd.DataFrame(index=p23.index)
# 1) 低外送 = 弱吸收
weak['weak_export'] = -p23['export_ratio_pct']
# 2) 供热刚性: 供热容量(万千瓦->MW) / 火电装机(MW); 无供热省份(年鉴空白)=结构零
heat_mw = p23['yb_heat_supply_capacity_mw'] * 10.0
weak['heat_rigidity'] = (heat_mw.fillna(0) / p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())
# 3) 资源波动: 风电利用小时/容量因子变异 (面板已有时不变量)
weak['resource_vol'] = p23['wind_cf_cv']
# 4) 在建压力: 在建容量(修正cap10kw序列, 万千瓦->MW) / 总装机(MW)
weak['pipeline_pressure'] = (p23['yb_power_project_under_construction_cap10kw'] * 10.0 /
                             p23['installed_capacity_mw_total'])
WEAK_VARS = ['weak_export', 'heat_rigidity', 'resource_vol', 'pipeline_pressure']
print('== WeakAbs 指标 (2023) 描述 ==')
print(weak[WEAK_VARS].describe().loc[['count', 'mean', 'std', 'min', 'max']].to_string())

for w in WEAK_VARS:
    col = d18['province_en'].map(weak[w])
    d18[w + '_z'] = (col - col.mean()) / col.std()
    med = col.median()
    d18[w + '_d'] = (col >= med).astype(float).where(col.notna())

# ---------------- 回归框架 ----------------
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
    """对 target 系数做 wild cluster bootstrap (受限模型 H0: beta_target=0)"""
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
    """TWFE(省份+年份FE), 省份聚类SE; target=关注系数列名"""
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

def triple(d, outcome, w, form, eq, spec, treat='dt_post', bootstrap=True):
    """三重交互: outcome ~ treat:weak + treat + controls + FE; 返回(β1记录, β2记录)"""
    dd = d.copy()
    if form == 'continuous':
        wcol = w + '_z'
    else:
        wcol = w + '_d'
    dd['treat_weak'] = dd[treat] * dd[wcol]
    dd2 = dd.dropna(subset=[wcol, treat])
    r1 = run_reg(dd2, outcome, treat, [], CTRL_B, eq, spec + f'|{w}|{form}',
                 weakabs=w, form=form, coef_label='beta1_main', bootstrap=bootstrap)
    r2 = run_reg(dd2, outcome, 'treat_weak', [treat], CTRL_B, eq, spec + f'|{w}|{form}',
                 weakabs=w, form=form, coef_label='beta2_triple', bootstrap=bootstrap)
    return r1, r2

# ================= 任务一: 三重交互 =================
print('== 任务一: 三重交互 ==')
for outcome in ['wind_curtailment_pct', 'solar_curtailment_pct']:
    for w in WEAK_VARS:
        for form in ['continuous', 'dummy']:
            triple(d18, outcome, w, form, 'T1', f'triple|{outcome}')

# ================= 任务二: 全面板时变强度 =================
print('== 任务二: 全面板时变强度 ==')
d_full = df[df['year'].between(2018, 2024)].copy()
d_full = d_full[(d_full['province_en'] != 'Tibet') & (d_full['province_en'] != 'Xinjiang')].copy()
d_full['dweight'] = d_full.groupby('province_en')['nonhydro_weight_binding'].diff()
d_full['dweight_l1'] = d_full.groupby('province_en')['dweight'].shift(1)
d_full['dweight_f1'] = d_full.groupby('province_en')['dweight'].shift(-1)
print('dweight 覆盖:', d_full.groupby('year')['dweight'].apply(lambda s: s.notna().sum()).to_dict())
d_t2 = d_full[d_full['year'].between(2019, 2024)].copy()

for outcome in ['wind_curtailment_pct', 'solar_curtailment_pct']:
    # T2a 静态
    run_reg(d_t2, outcome, 'dweight', [], CTRL_B, 'T2a', f'static|{outcome}')
    # T2b lead-lag
    for coef, lab in [('dweight_f1', 'lead1'), ('dweight', 'contemp'), ('dweight_l1', 'lag1')]:
        others = [c for c in ['dweight_f1', 'dweight', 'dweight_l1'] if c != coef]
        run_reg(d_t2, outcome, coef, others, CTRL_B, 'T2b', f'leadlag|{outcome}', coef_label=lab)

# T2c 全面板三重交互 (连续z; 对4个WeakAbs)
for w in WEAK_VARS:
    col = d_t2['province_en'].map(weak[w])
    d_t2[w + '_z'] = (col - col.mean()) / col.std()
for outcome in ['wind_curtailment_pct', 'solar_curtailment_pct']:
    for w in WEAK_VARS:
        dd = d_t2.dropna(subset=[w + '_z']).copy()
        dd['dw_weak'] = dd['dweight'] * dd[w + '_z']
        run_reg(dd, outcome, 'dw_weak', ['dweight'], CTRL_B, 'T2c',
                f'panel_triple|{outcome}', weakabs=w, form='continuous', coef_label='beta2_triple')

# ================= 任务三a/b: 利用小时 =================
print('== 任务三: 利用小时 + 弃光诊断 ==')
for outcome in ['wind_util_hours', 'solar_util_hours']:
    for w in WEAK_VARS:
        triple(d18, outcome, w, 'continuous', 'T3a', f'util_triple|{outcome}')
    run_reg(d_t2, outcome, 'dweight', [], CTRL_B, 'T3b', f'util_static|{outcome}')

# ================= 任务三c: 弃光率预处理趋势诊断 =================
# 各省2018-2022弃光率趋势
solar_diag = []
for p, g in d18.groupby('province_en'):
    g = g.sort_values('year')
    pre = g[g['year'] <= 2022]
    slope = np.polyfit(pre['year'], pre['solar_curtailment_pct'], 1)[0] if len(pre) >= 3 else np.nan
    chg1822 = pre['solar_curtailment_pct'].iloc[-1] - pre['solar_curtailment_pct'].iloc[0]
    solar_diag.append(dict(province_en=p, pretrend_slope=slope, chg_2018_2022=chg1822,
                           delta_target=g['delta_target_i'].iloc[0],
                           chg_2023_2024=(g[g.year == 2024]['solar_curtailment_pct'].sum()
                                          - g[g.year == 2023]['solar_curtailment_pct'].sum())))
sdg = pd.DataFrame(solar_diag).sort_values('chg_2018_2022', ascending=False)
sdg.to_csv(os.path.join(BASE, 'solar_pretrend_by_province_v8d.csv'), index=False, encoding='utf-8-sig')
print('弃光率2018-2022上升最多的省份:')
print(sdg.head(8).to_string())
# 分布式光伏: 面板仅2018-2020
dist = df[df['year'].between(2018, 2020)].pivot_table(
    index='province_en', columns='year', values='distributed_solar_capacity_mw')
dist_growth = ((dist[2020] / dist[2018]) - 1).rename('dist_solar_growth_18_20')
sdg2 = sdg.merge(dist_growth, on='province_en', how='left')
print('弃光上升省与分布式光伏增长(2018-2020)相关性:',
      sdg2[['chg_2018_2022', 'dist_solar_growth_18_20']].corr().iloc[0, 1])

# 诊断图
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
med_dt = d18['delta_target_i'].median()
for p, g in d18.groupby('province_en'):
    g = g.sort_values('year')
    hi = g['delta_target_i'].iloc[0] >= med_dt
    axes[0].plot(g['year'], g['solar_curtailment_pct'], color='red' if hi else 'steelblue',
                 alpha=0.25, lw=0.8)
for hi, c, lab in [(True, 'red', 'High delta_target (>=median)'), (False, 'steelblue', 'Low delta_target')]:
    sub = d18[d18['province_en'].map(
        d18.groupby('province_en')['delta_target_i'].first() >= med_dt) == hi]
    mp = sub.groupby('year')['solar_curtailment_pct'].mean()
    axes[0].plot(mp.index, mp.values, color=c, lw=2.5, marker='o', label=lab)
axes[0].axvline(2023.5, color='gray', ls='--', lw=1)
axes[0].set_title('Solar curtailment rate paths 2018-2024 (bold = group means)')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Solar curtailment (%)'); axes[0].legend(fontsize=8)
top = sdg.head(12).iloc[::-1]
axes[1].barh(top['province_en'], top['chg_2018_2022'], color='darkorange')
axes[1].set_title('Change in solar curtailment 2018->2022 (pp), top risers')
axes[1].set_xlabel('pp change')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'solar_pretrend_diagnosis_v8d.png'), dpi=300)
plt.close()

# 修正方案: 剔除2018-2022弃光率上升>1pp的省份, 重跑弃光基准+最强三重交互
risers = sdg[sdg['chg_2018_2022'] > 1.0]['province_en'].tolist()
print('剔除省份(弃光2018-2022上升>1pp):', risers)
d18_norise = d18[~d18['province_en'].isin(risers)].copy()
run_reg(d18_norise, 'solar_curtailment_pct', 'dt_post', [], CTRL_B, 'T3c',
        'solar_excl_risers|baseline', coef_label='beta1_main')

# ================= 选最强WeakAbs (弃风, 连续形式, β2的p最小) =================
res = pd.DataFrame(RESULTS)
t1w = res[(res['eq'] == 'T1') & (res['outcome'] == 'wind_curtailment_pct')
          & (res['form'] == 'continuous') & (res['coef'] == 'beta2_triple')]
best_w = t1w.sort_values('p_cluster').iloc[0]['weakabs'] if len(t1w) else 'heat_rigidity'
print('最强WeakAbs(弃风, β2 p最小):', best_w)
ddnr = d18_norise.copy()
ddnr['treat_weak'] = ddnr['dt_post'] * ddnr[best_w + '_z']
run_reg(ddnr.dropna(subset=[best_w + '_z']), 'solar_curtailment_pct', 'treat_weak',
        ['dt_post'], CTRL_B, 'T3c', f'solar_excl_risers|triple_{best_w}',
        weakabs=best_w, form='continuous', coef_label='beta2_triple')

# ================= 任务四: 剂量-反应图 =================
print('== 任务四: 剂量-反应图 ==')
chg = []
for p, g in d18.groupby('province_en'):
    g = g.sort_values('year')
    y23 = g.loc[g.year == 2023, 'wind_curtailment_pct']
    y24 = g.loc[g.year == 2024, 'wind_curtailment_pct']
    if len(y23) and len(y24):
        chg.append(dict(province_en=p, delta_target=g['delta_target_i'].iloc[0],
                        d_wind_curt=float(y24.iloc[0] - y23.iloc[0]),
                        weak=g[best_w + '_z'].iloc[0]))
dose = pd.DataFrame(chg).dropna()
med_w = dose['weak'].median()
dose['group'] = np.where(dose['weak'] >= med_w, 'Weak absorption (high)', 'Strong absorption (low)')
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'Weak absorption (high)': 'firebrick', 'Strong absorption (low)': 'steelblue'}
for grp, c in colors.items():
    s = dose[dose['group'] == grp]
    ax.scatter(s['delta_target'], s['d_wind_curt'], c=c, s=55, alpha=0.85,
               edgecolor='white', label=f'{grp} (n={len(s)})')
    z = np.polyfit(s['delta_target'], s['d_wind_curt'], 1)
    xs = np.linspace(s['delta_target'].min(), s['delta_target'].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=c, lw=2)
    # 组内斜率标注
    ax.annotate(f'slope={z[0]:.3f}', xy=(0.02 if c == 'firebrick' else 0.02, 0.95 if c == 'firebrick' else 0.88),
                xycoords='axes fraction', color=c, fontsize=10)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel('RPS non-hydro target acceleration 2024 vs 2023 (pp)')
ax.set_ylabel('Change in wind curtailment rate 2024 vs 2023 (pp)')
ax.set_title(f'Dose-response: target acceleration vs wind curtailment change\n(split by median {best_w})')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'dose_response_v8d.png'), dpi=300)
plt.close()
dose.to_csv(os.path.join(BASE, 'dose_response_data_v8d.csv'), index=False, encoding='utf-8-sig')

# ================= 输出 =================
res = pd.DataFrame(RESULTS)
res.to_csv(os.path.join(BASE, 'redesign_results_v8d.csv'), index=False, encoding='utf-8-sig')
print(f'\n完成: {len(res)}条系数记录, {len(DEGENERATE)}个退化丢弃, {len(WARNINGS)}条警告')
print('BEST_W =', best_w)
import json
meta = dict(best_w=best_w, risers=risers, n_degenerate=len(DEGENERATE),
            warnings=WARNINGS, degenerate=DEGENERATE)
with open(os.path.join(BASE, '_redesign_meta_v8d.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
