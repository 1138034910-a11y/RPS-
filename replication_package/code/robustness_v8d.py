# -*- coding: utf-8 -*-
"""
robustness_v8.py  —  RPS论文主结果稳健性三连 (睡前跑版)
数据: master_panel_v8d.csv (与 redesign_analysis_v8.py 相同的数据准备)

  洞1: T2c/T1 主设定加入 heat_z × 年份固定效应 (排除"北方坏年份"混淆)
  洞2: T2c leave-one-out (29省逐省剔除, 检验结果是否被个别省份撑起来)
  洞3: 随机化推断 (placebo assignment, 500次) + 2024现货试点省份控制

输出 (均为新文件, 不覆盖旧结果):
  robustness_v8d_results.csv   全部回归/检验明细
  robustness_v8d_report.md     自动生成的判读报告 (早上起来先看这个)

预计运行时间: 20-60 分钟 (主要在 leave-one-out 的 wild bootstrap)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os, time, datetime, warnings

warnings.filterwarnings('ignore')
BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
np.random.seed(20260724)
NBOOT_MAIN = 999   # 主设定 bootstrap 次数
NBOOT_LOO = 499    # leave-one-out bootstrap 次数 (省时间)
N_PERM = 500       # 随机化推断次数

CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']
RESULTS = []

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ================= 数据准备 (与 redesign_analysis_v8.py 完全一致) =================
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)

p23 = df[df['year'] == 2023].set_index('province_en')
heat_mw = p23['yb_heat_supply_capacity_mw'] * 10.0
heat_rigidity = (heat_mw.fillna(0) / p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())

def prep(d):
    """加入 post/dt_post/dweight/heat_z 及 heat_z×year 显式交互列"""
    d = d.copy()
    d['post'] = (d['year'] >= 2024).astype(int)
    d['dt_post'] = d['delta_target_i'] * d['post']
    d['dweight'] = d.groupby('province_en')['nonhydro_weight_binding'].diff()
    d['heat_z'] = d['province_en'].map(heat_rigidity)
    d['heat_z'] = (d['heat_z'] - d['heat_z'].mean()) / d['heat_z'].std()
    for yr in sorted(d['year'].unique()):
        d[f'heat_yr_{yr}'] = d['heat_z'] * (d['year'] == yr).astype(float)
    return d

d18 = df[df['year'].between(2018, 2024)].copy()
d18 = d18[(d18['province_en'] != 'Tibet') & (d18['province_en'] != 'Xinjiang')].copy()
d18 = prep(d18)
d18['treat_weak'] = d18['dt_post'] * d18['heat_z']          # T1 三重交互项

d_t2 = d18[d18['year'].between(2019, 2024)].copy()
d_t2['dw_weak'] = d_t2['dweight'] * d_t2['heat_z']          # T2c 三重交互项

# 现货市场首批试点省 (2017年首批8个: 广东/蒙西/浙江/山西/山东/福建/四川/甘肃)
SPOT_PILOT = ['Guangdong', 'Inner Mongolia', 'Zhejiang', 'Shanxi',
              'Shandong', 'Fujian', 'Sichuan', 'Gansu']
for d in (d18, d_t2):
    d['spot_pilot'] = d['province_en'].isin(SPOT_PILOT).astype(float)
    d['spot_post'] = d['spot_pilot'] * d['post']

# ================= 回归与推断框架 =================
def fit_target(d, outcome, target, rhs_others):
    """LSDV双向固定效应 + 省份聚类SE; 返回 (beta, se, p, n_obs, n_prov)"""
    cols = [outcome, target, 'province_en', 'year'] + rhs_others
    dd = d[cols].dropna()
    if len(dd) < 20 or dd['province_en'].nunique() < 5 or dd[target].var() < 1e-14:
        return None
    rhs = ' + '.join([target] + rhs_others) + ' + C(province_en) + C(year)'
    m = smf.ols(f'{outcome} ~ {rhs}', data=dd).fit(
        cov_type='cluster', cov_kwds={'groups': dd['province_en']})
    if target not in m.params.index:
        return None
    return (m.params[target], m.bse[target], m.pvalues[target],
            int(m.nobs), dd['province_en'].nunique(), dd)

def wild_boot_p(dd, outcome, target, other_x, beta_obs, se_obs, n_boot):
    """wild cluster bootstrap (FWL, Rademacher) — 与 redesign_analysis_v8.py 同一实现"""
    provs = dd['province_en'].values
    uniq = np.unique(provs)
    years = dd['year'].values
    uy = np.unique(years)
    M = [np.ones(len(dd))]
    for g in uniq[1:]:
        M.append((provs == g).astype(float))
    for yy in uy[1:]:
        M.append((years == yy).astype(float))
    M = np.column_stack(M)
    C = dd[other_x].values.astype(float) if other_x else np.zeros((len(dd), 0))
    tvec = dd[target].values.astype(float)
    y = dd[outcome].values.astype(float)
    X = np.column_stack([tvec, C])
    N, K = len(dd), M.shape[1] + X.shape[1]
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

def run_and_record(tag, d, outcome, target, rhs_others, n_boot=NBOOT_MAIN):
    """点估计 + 聚类p + wild bootstrap p, 记录到 RESULTS"""
    r = fit_target(d, outcome, target, rhs_others)
    if r is None:
        log(f'  {tag}: 拟合失败/样本不足, 跳过')
        return None
    beta, se, p, n_obs, n_prov, dd = r
    pb = wild_boot_p(dd, outcome, target, rhs_others, beta, se, n_boot)
    RESULTS.append(dict(part=tag, outcome=outcome, coef=target, beta=beta,
                        se_cluster=se, p_cluster=p, p_wildboot=pb,
                        n_obs=n_obs, n_prov=n_prov))
    log(f'  {tag}: beta={beta:+.4f}  p_cluster={p:.4f}  p_boot={pb:.4f}  (n={n_obs}, prov={n_prov})')
    return beta, se, p, pb

def fast_beta(d, outcome, target, rhs_others):
    """仅算点估计 (FWL numpy), 供随机化推断循环用"""
    cols = [outcome, target, 'province_en', 'year'] + rhs_others
    dd = d[cols].dropna()
    provs, years = dd['province_en'].values, dd['year'].values
    M = [np.ones(len(dd))]
    for g in np.unique(provs)[1:]:
        M.append((provs == g).astype(float))
    for yy in np.unique(years)[1:]:
        M.append((years == yy).astype(float))
    M = np.column_stack(M)
    X = np.column_stack([dd[target].values.astype(float)] +
                        [dd[c].values.astype(float) for c in rhs_others])
    y = dd[outcome].values.astype(float)
    def absorb(v):
        b, *_ = np.linalg.lstsq(M, v, rcond=None)
        return v - M @ b
    X_t = np.column_stack([absorb(X[:, j]) for j in range(X.shape[1])])
    y_t = absorb(y)
    beta, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
    return beta[0]

# ================= 洞1: 调节变量×年份FE =================
log('== 洞1: heat_z × 年份FE (排除高供热省份的差异性年度冲击) ==')
heat_yr_cols = [c for c in d18.columns if c.startswith('heat_yr_')]

run_and_record('H1_T2c_baseline', d_t2, 'wind_curtailment_pct', 'dw_weak',
               ['dweight'] + CTRL_B)
run_and_record('H1_T2c_heatXyear', d_t2, 'wind_curtailment_pct', 'dw_weak',
               ['dweight'] + CTRL_B + heat_yr_cols)
run_and_record('H1_T1_baseline', d18, 'wind_curtailment_pct', 'treat_weak',
               ['dt_post'] + CTRL_B)
run_and_record('H1_T1_heatXyear', d18, 'wind_curtailment_pct', 'treat_weak',
               ['dt_post'] + CTRL_B + heat_yr_cols)

# ================= 洞2: T2c leave-one-out =================
log('== 洞2: T2c leave-one-out (29省逐省剔除, 两种设定) ==')
loo_rows = []
provs = sorted(d_t2['province_en'].unique())
for i, prov in enumerate(provs, 1):
    d_loo = d_t2[d_t2['province_en'] != prov]
    for spec, extra in [('baseline', []), ('heatXyear', heat_yr_cols)]:
        r = fit_target(d_loo, 'wind_curtailment_pct', 'dw_weak',
                       ['dweight'] + CTRL_B + extra)
        if r is None:
            continue
        beta, se, p, n_obs, n_prov, dd = r
        pb = wild_boot_p(dd, 'wind_curtailment_pct', 'dw_weak',
                         ['dweight'] + CTRL_B + extra, beta, se, NBOOT_LOO)
        loo_rows.append(dict(left_out=prov, spec=spec, beta=beta, se_cluster=se,
                             p_cluster=p, p_wildboot=pb, n_obs=n_obs))
    log(f'  [{i}/{len(provs)}] 剔除 {prov} 完成')
loo_df = pd.DataFrame(loo_rows)

# ================= 洞3a: 随机化推断 =================
log('== 洞3a: 随机化推断 (placebo assignment, 500次) ==')
# RI-1 (T1): 把 delta_target_i 在省间随机重排, 重算三重交互系数
d_ri = d18.dropna(subset=['wind_curtailment_pct', 'treat_weak', 'dt_post',
                          'heat_z'] + CTRL_B).copy()
beta_actual_T1 = fast_beta(d_ri, 'wind_curtailment_pct', 'treat_weak', ['dt_post'] + CTRL_B)
prov_list = d_ri['province_en'].unique()
delta_map = d_ri.drop_duplicates('province_en').set_index('province_en')['delta_target_i']
cnt = 0
for b in range(N_PERM):
    perm = np.random.permutation(delta_map.values)
    d_ri['delta_p'] = d_ri['province_en'].map(dict(zip(delta_map.index, perm)))
    d_ri['dt_p'] = d_ri['delta_p'] * d_ri['post']
    d_ri['tw_p'] = d_ri['dt_p'] * d_ri['heat_z']
    bp = fast_beta(d_ri, 'wind_curtailment_pct', 'tw_p', ['dt_p'] + CTRL_B)
    if abs(bp) >= abs(beta_actual_T1):
        cnt += 1
    if (b + 1) % 100 == 0:
        log(f'  RI-T1 进度 {b+1}/{N_PERM}')
ri_p_T1 = (cnt + 1) / (N_PERM + 1)
RESULTS.append(dict(part='H3_RI_T1_permDeltaTarget', outcome='wind_curtailment_pct',
                    coef='beta2_triple', beta=beta_actual_T1, se_cluster=np.nan,
                    p_cluster=np.nan, p_wildboot=np.nan, n_obs=len(d_ri),
                    n_prov=len(prov_list), ri_p=ri_p_T1))
log(f'  RI-T1 (ΔTarget随机重排): beta_actual={beta_actual_T1:+.4f}, RI p={ri_p_T1:.4f}')

# RI-2 (T2c): 把 heat_rigidity 标签在省间随机重排 (placebo调节变量)
d_ri2 = d_t2.dropna(subset=['wind_curtailment_pct', 'dw_weak', 'dweight',
                            'heat_z'] + CTRL_B).copy()
beta_actual_T2c = fast_beta(d_ri2, 'wind_curtailment_pct', 'dw_weak', ['dweight'] + CTRL_B)
heat_map = d_ri2.drop_duplicates('province_en').set_index('province_en')['heat_z']
cnt = 0
for b in range(N_PERM):
    perm = np.random.permutation(heat_map.values)
    d_ri2['heat_p'] = d_ri2['province_en'].map(dict(zip(heat_map.index, perm)))
    d_ri2['dw_p'] = d_ri2['dweight'] * d_ri2['heat_p']
    bp = fast_beta(d_ri2, 'wind_curtailment_pct', 'dw_p', ['dweight'] + CTRL_B)
    if abs(bp) >= abs(beta_actual_T2c):
        cnt += 1
    if (b + 1) % 100 == 0:
        log(f'  RI-T2c 进度 {b+1}/{N_PERM}')
ri_p_T2c = (cnt + 1) / (N_PERM + 1)
RESULTS.append(dict(part='H3_RI_T2c_permHeatLabel', outcome='wind_curtailment_pct',
                    coef='beta2_triple', beta=beta_actual_T2c, se_cluster=np.nan,
                    p_cluster=np.nan, p_wildboot=np.nan, n_obs=len(d_ri2),
                    n_prov=len(prov_list), ri_p=ri_p_T2c))
log(f'  RI-T2c (heat标签随机重排): beta_actual={beta_actual_T2c:+.4f}, RI p={ri_p_T2c:.4f}')

# ================= 洞3b: 2024混杂政策控制 (现货试点) =================
log('== 洞3b: 控制现货市场试点省×post ==')
run_and_record('H3_T1_spot_ctrl', d18, 'wind_curtailment_pct', 'treat_weak',
               ['dt_post', 'spot_post'] + CTRL_B)

# ================= 汇总输出 =================
log('== 写出结果与报告 ==')
res_df = pd.DataFrame(RESULTS)
res_df.to_csv(os.path.join(BASE, 'robustness_v8d_results.csv'),
              index=False, encoding='utf-8-sig')
loo_df.to_csv(os.path.join(BASE, 'robustness_v8d_loo.csv'),
              index=False, encoding='utf-8-sig')

def get(tag):
    r = res_df[res_df['part'] == tag]
    return r.iloc[0] if len(r) else None

lines = ['# 稳健性三连报告 (robustness_v8)', '',
         f'生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M}', '',
         '## 洞1: heat_z × 年份FE', '',
         '| 设定 | beta2 | p_cluster | p_boot |', '|---|---|---|---|']
for tag, lab in [('H1_T2c_baseline', 'T2c 基准'), ('H1_T2c_heatXyear', 'T2c +heat×yearFE'),
                 ('H1_T1_baseline', 'T1 基准'), ('H1_T1_heatXyear', 'T1 +heat×yearFE')]:
    r = get(tag)
    if r is not None:
        lines.append(f"| {lab} | {r['beta']:+.4f} | {r['p_cluster']:.4f} | {r['p_wildboot']:.4f} |")

lines += ['', '**判读**: 加heat×yearFE后β2量级与显著性若基本不变 → 结果不是"北方坏年份"伪影; 若掉到不显著 → 主结果需降级为暗示性证据。', '',
          '## 洞2: T2c leave-one-out (29省)', '']
for spec in ['baseline', 'heatXyear']:
    s = loo_df[loo_df['spec'] == spec]
    if len(s):
        n05 = (s['p_cluster'] < 0.05).sum(); n10 = (s['p_cluster'] < 0.10).sum()
        lines += [f"**{spec}**: beta2范围 [{s['beta'].min():+.3f}, {s['beta'].max():+.3f}], "
                  f"p<0.05 的run数 {n05}/{len(s)}, p<0.10 的run数 {n10}/{len(s)}",
                  '',
                  s.nsmallest(3, 'beta')[['left_out', 'beta', 'p_cluster']].to_string(index=False),
                  ' (↑ 剔除后beta最小的3个省, 即对结果支撑最强的省)', '']
lines += ['**判读**: 若没有任何一个省剔除后β2掉到不显著区间且符号不变 → 稳健; 若剔黑龙江/吉林/辽宁/内蒙古中某一个就崩 → 论文必须交代单省驱动。', '',
          '## 洞3: 随机化推断 + 混杂政策', '',
          f"- RI-T1 (ΔTarget省间随机重排500次): beta_actual={beta_actual_T1:+.4f}, **RI p={ri_p_T1:.4f}**",
          f"- RI-T2c (heat标签随机重排500次): beta_actual={beta_actual_T2c:+.4f}, **RI p={ri_p_T2c:.4f}**",
          '- 判读: RI p < 0.05 → 真实分配下的系数处于随机分配分布的尾部, 不是凑巧。',
          '']
r = get('H3_T1_spot_ctrl')
if r is not None:
    lines.append(f"- T1控制现货试点省×post后: beta2={r['beta']:+.4f} (p_cluster={r['p_cluster']:.4f}, "
                 f"p_boot={r['p_wildboot']:.4f}) — 若与基准相近 → 2024现货改革不驱动主结果")
lines += ['', '## 明细', '', res_df.to_string(index=False)]

with open(os.path.join(BASE, 'robustness_v8d_report.md'), 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(lines))

log('全部完成! 早上看 robustness_v8d_report.md')
