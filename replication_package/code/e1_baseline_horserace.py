# -*- coding: utf-8 -*-
"""
e1_baseline_horserace.py
E1 立项检验: 尺度效应(scale artifact) vs 供热锁定机制(mechanism) 的赛马
=====================================================================
研究问题: 主结果 β2(Δweight × heat_rigidity)= 0.229 究竟是机制
  (heat-determined-electricity 的冬夜出力地板), 还是"高基线弃风省对任何冲击
  绝对波动都大"的尺度效应?

设计逻辑(资深研究者版, 预注册判定标准):
  对手变量(2023 基年, 与 heat_rigidity 同基年、同 z 标准化, 系数可直接比):
    base_level = 2023 年弃风率水平      <- "高基线省波动大"的主代理
    base_vol   = 2018-2023 弃风率省内标准差 <- 尺度效应的波动率代理
    atcap      = 2023 弃风率 >= 4% 哑变量  <- "贴在95%红线上被压制"的非线性代理
                                             (同时回应2024红线放松混杂, R2-缺点1)
  关键列: heat 与 base 的交互项同场赛马(同一回归)。
  附加识别:
    - base×yearFE 列: 允许高基线省有任意的逐年水平摆动后, 交互是否还在
    - 供热省子样本(剔4个结构零省): 基线变异天然收窄的子样本里赛马
    - log1p 被解释变量: 若效应纯属"与基线成比例", 取对数后 heat 交互应消失
    - 正交化: heat 对 base_level+base_vol 取残差后再交互(共线兜底)
  诊断: corr 矩阵(含 p 值) + 赛马列 VIF

预注册判定(写在跑数之前, 防事后挪动球门):
  存活(A): 赛马列 β2_heat 量级保留 >= 70% 且 cluster p < 0.05 -> 机制立住
  半活(B): 量级 40-70% 或边缘显著 -> 降级表述, 主打子样本+月度季节签名
  死(C):   base 交互显著且 β2_heat 量级 < 40% 或不再显著 -> 标题级结论不成立
  共线陷阱: 若 corr(heat, base_level) > 0.8, 赛马无分辨力, 以子样本/正交化为准

复用 v8d 管线设定(不重新发明):
  样本/控制/聚类/bootstrap 与 redesign_analysis_v8d.py 完全一致:
  T2c: 2019-2024, 29省(剔 Tibet/Xinjiang), dweight=nonhydro_weight_binding 省内diff,
       控制 CTRL_B = wind/solar_cap_growth_pct + consumption_growth_pct,
       省份+年份 FE, 省份聚类 SE, wild cluster bootstrap (FWL, Rademacher, 999次)
  T1:  2018-2024, treat = delta_target_i × post2024, 三重交互 treat×moderator
  heat_rigidity = yb_heat_supply_capacity_mw / installed_capacity_mw_thermal (2023; 同名单位相除, 均值0.460)
基准锚点(自检): T2c β2_heat ≈ 0.229; T1 β2_heat ≈ 0.095
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats as sstats
import os, warnings, json

warnings.filterwarnings('ignore')
BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
OUTDIR = os.path.join(BASE, '01_analysis_current')
np.random.seed(20260724)   # 与主管线同种子
NBOOT = 999
CTRL_B = ['wind_cap_growth_pct', 'solar_cap_growth_pct', 'consumption_growth_pct']

RESULTS, WARNINGS = [], []

# ---------------- 数据与变量构造 ----------------
df = pd.read_csv(os.path.join(BASE, 'master_panel_v8d.csv'), encoding='utf-8-sig')
df['province_en'] = df['province_en'].astype(object)
df['year'] = df['year'].astype(int)
df = df.sort_values(['province_en', 'year']).reset_index(drop=True)

pall = df[df['year'].between(2018, 2023)]   # 冲击前窗口(构造波动率/趋势)
p23 = df[df['year'] == 2023].set_index('province_en')

# 机制变量(2023 基年; 论文口径: 同名单位相除, 均值0.460; z化后回归不变)
heat = (p23['yb_heat_supply_capacity_mw'].fillna(0) / p23['installed_capacity_mw_thermal']).where(
    p23['installed_capacity_mw_thermal'].notna())
heat.name = 'heat_rigidity'

# 对手变量
base_level = p23['wind_curtailment_pct'].rename('base_level')
# 2018 基线: 唯一完全处理前的弃风年份(RPS 2019-05 试行), T2c 的内生性问题兜底
base_2018 = df[df['year'] == 2018].set_index('province_en')['wind_curtailment_pct'].rename('base_2018')
base_vol = pall.groupby('province_en')['wind_curtailment_pct'].std().rename('base_vol')
atcap = (p23['wind_curtailment_pct'] >= 4.0).astype(float).rename('atcap')
slope = pall[pall['year'] <= 2022].groupby('province_en').apply(
    lambda g: np.polyfit(g['year'], g['wind_curtailment_pct'], 1)[0] if len(g) >= 3 else np.nan
).rename('pretrend_slope')
dT = df[df['year'] == 2024].set_index('province_en')['delta_target_i'].rename('delta_target')

prov = pd.concat([heat, base_level, base_2018, base_vol, atcap, slope, dT], axis=1)
prov = prov.loc[[p for p in prov.index if p not in ('Tibet', 'Xinjiang')]]
print(f'省级变量表: {len(prov)} 省')
print(prov[['heat_rigidity', 'base_level', 'base_2018', 'base_vol', 'delta_target']].describe().loc[
      ['count', 'mean', 'std', 'min', 'max']].to_string())

# ---------------- 相关矩阵(含 p 值) ----------------
corr_pairs = [('heat_rigidity', 'base_level'), ('heat_rigidity', 'base_2018'), ('heat_rigidity', 'base_vol'),
              ('heat_rigidity', 'pretrend_slope'), ('delta_target', 'heat_rigidity'),
              ('delta_target', 'base_level'), ('delta_target', 'pretrend_slope'),
              ('base_level', 'base_vol')]
corr_rows = []
for a, b in corr_pairs:
    sub = prov[[a, b]].dropna()
    r, p = sstats.pearsonr(sub[a], sub[b])
    corr_rows.append(dict(var_a=a, var_b=b, n=len(sub), r=r, p=p))
    print(f'corr({a},{b}) = {r:+.3f} (p={p:.4f}, n={len(sub)})')
pd.DataFrame(corr_rows).to_csv(os.path.join(OUTDIR, 'e1_correlations.csv'),
                               index=False, encoding='utf-8-sig')
c_heat_base = corr_rows[0]['r']
if abs(c_heat_base) > 0.8:
    WARNINGS.append(f'corr(heat, base_level)={c_heat_base:.3f} > 0.8: 赛马共线陷阱触发, 以子样本/正交化为准')

# z 标准化: 与管线完全一致——矩(均值/标准差)取自分析样本的映射列(省×年), 再映射回省
def zmap_on(d, s):
    col = d['province_en'].map(s)
    return ((s - col.mean()) / col.std()).to_dict()
# 正交化: heat ~ base_level + base_vol 的残差(省级原始量纲, z 化在映射时做)
Xb = np.column_stack([np.ones(len(prov)), prov['base_level'].values, prov['base_vol'].values])
bh, *_ = np.linalg.lstsq(Xb, prov['heat_rigidity'].values, rcond=None)
heat_resid = pd.Series(prov['heat_rigidity'].values - Xb @ bh, index=prov.index)

# ---------------- 回归框架(复用 v8d 实现) ----------------
def absorb_mat(d):
    provs = d['province_en'].values
    years = d['year'].values
    M = [np.ones(len(d))]
    for g in np.unique(provs)[1:]:
        M.append((provs == g).astype(float))
    for yy in np.unique(years)[1:]:
        M.append((years == yy).astype(float))
    return np.column_stack(M), np.unique(provs), provs

def wild_boot_p(d, outcome, target, other_x, beta_obs, se_obs, n_boot=NBOOT):
    """对 target 系数做 wild cluster bootstrap(受限模型 H0: beta_target=0, 其余回归元保留)"""
    M, uniq, provs = absorb_mat(d)
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

def run_spec(d, outcome, targets, extra, eq, spec, bootstrap_targets=None):
    """targets=关注的交互项列表; extra=其他 RHS(dweight/treat 主效应、双倍交互、控制)"""
    d = d.copy()
    rhs_all = targets + extra + CTRL_B
    need = [outcome, 'province_en', 'year'] + rhs_all
    d = d.dropna(subset=[c for c in need if c in d.columns])
    n_obs, n_prov = len(d), d['province_en'].nunique()
    rhs = ' + '.join(rhs_all) + ' + C(province_en) + C(year)'
    try:
        m = smf.ols(f'{outcome} ~ {rhs}', data=d).fit(
            cov_type='cluster', cov_kwds={'groups': d['province_en']})
    except Exception as e:
        WARNINGS.append(f'[{eq}|{spec}] 回归失败: {e}')
        return
    others = [c for c in rhs_all if True]
    for t in targets:
        if t not in m.params.index:
            WARNINGS.append(f'[{eq}|{spec}] {t} 被吸收, 跳过')
            continue
        beta, se, p = m.params[t], m.bse[t], m.pvalues[t]
        pb = np.nan
        if bootstrap_targets and t in bootstrap_targets:
            ox = [c for c in rhs_all if c != t]
            try:
                pb = wild_boot_p(d, outcome, t, ox, beta, se)
            except Exception as e:
                WARNINGS.append(f'[{eq}|{spec}] {t} bootstrap失败: {e}')
        RESULTS.append(dict(eq=eq, spec=spec, outcome=outcome, coef=t, beta=beta,
                            se_cluster=se, p_cluster=p, p_wildboot=pb,
                            n_obs=n_obs, n_prov=n_prov))
    # VIF(交互项之间, 对 FE 残差化后)
    if len(targets) > 1:
        M, _, _ = absorb_mat(d)
        resid = {}
        for t in targets:
            v = d[t].values.astype(float)
            b, *_ = np.linalg.lstsq(M, v, rcond=None)
            resid[t] = v - M @ b
        for t in targets:
            r = resid[t]
            Xo = np.column_stack([resid[o] for o in targets if o != t])
            bo, *_ = np.linalg.lstsq(Xo, r, rcond=None)
            rr = r - Xo @ bo
            r2 = 1 - (rr ** 2).sum() / max((r ** 2).sum(), 1e-20)
            WARNINGS.append(f'[{eq}|{spec}] VIF({t}) = {1 / max(1 - r2, 1e-10):.2f}')

# ---------------- 样本构造 ----------------
def add_mods(d):
    d = d.copy()
    for v in ['heat_rigidity', 'base_level', 'base_2018', 'base_vol', 'pretrend_slope']:
        d[v + '_z'] = d['province_en'].map(zmap_on(d, prov[v]))
    d['heat_resid_z'] = d['province_en'].map(zmap_on(d, heat_resid))
    return d

# T2c: 2019-2024
d_full = df[(df['year'].between(2018, 2024)) &
            (~df['province_en'].isin(['Tibet', 'Xinjiang']))].copy()
d_full['dweight'] = d_full.groupby('province_en')['nonhydro_weight_binding'].diff()
d2 = add_mods(d_full[d_full['year'].between(2019, 2024)].copy())
d2['dw_heat'] = d2['dweight'] * d2['heat_rigidity_z']
d2['dw_base'] = d2['dweight'] * d2['base_level_z']
d2['dw_base2018'] = d2['dweight'] * d2['base_2018_z']
d2['dw_pretrend'] = d2['dweight'] * d2['pretrend_slope_z']
d2['dw_basevol'] = d2['dweight'] * d2['base_vol_z']
d2['dw_atcap'] = d2['dweight'] * d2['province_en'].map(prov['atcap'])
d2['dw_heatres'] = d2['dweight'] * d2['heat_resid_z']

# T1: 2018-2024
d1 = add_mods(d_full.copy())
d1['post'] = (d1['year'] >= 2024).astype(int)
d1['dt_post'] = d1['delta_target_i'] * d1['post']
d1['t_heat'] = d1['dt_post'] * d1['heat_rigidity_z']
d1['t_base'] = d1['dt_post'] * d1['base_level_z']
d1['t_base2018'] = d1['dt_post'] * d1['base_2018_z']
d1['t_basevol'] = d1['dt_post'] * d1['base_vol_z']
d1['base_post'] = d1['base_level_z'] * d1['post']

OUT = 'wind_curtailment_pct'

# ================= T2c 家族 =================
print('== T2c ==')
run_spec(d2, OUT, ['dw_heat'], ['dweight'], 'T2c', 'bench_heat', ['dw_heat'])
run_spec(d2, OUT, ['dw_base'], ['dweight'], 'T2c', 'scale_only', ['dw_base'])
run_spec(d2, OUT, ['dw_heat', 'dw_base'], ['dweight'], 'T2c', 'race', ['dw_heat', 'dw_base'])
run_spec(d2, OUT, ['dw_heat', 'dw_base', 'dw_basevol'], ['dweight'], 'T2c', 'race_plus_vol',
         ['dw_heat', 'dw_base', 'dw_basevol'])
# 处理前基线(2018): T2c 的处理跨 2019-2024, 2023 基线可能已被早期处理污染, 2018 兜底
run_spec(d2, OUT, ['dw_heat', 'dw_base2018'], ['dweight'], 'T2c', 'race_base2018',
         ['dw_heat', 'dw_base2018'])
# 第三尺度代理: 处理前上升趋势
run_spec(d2, OUT, ['dw_heat', 'dw_base', 'dw_pretrend'], ['dweight'], 'T2c', 'race_plus_pretrend',
         ['dw_heat', 'dw_base', 'dw_pretrend'])
run_spec(d2, OUT, ['dw_heat', 'dw_base', 'dw_atcap'], ['dweight'], 'T2c', 'race_plus_atcap',
         ['dw_heat', 'dw_base'])
# base×yearFE(允许高基线省任意逐年摆动): 手动构造交互列
d2y = d2.copy()
for yy in sorted(d2y['year'].unique()):
    d2y[f'base_y{yy}'] = d2y['base_level_z'] * (d2y['year'] == yy).astype(float)
basey_cols = [f'base_y{yy}' for yy in sorted(d2y['year'].unique())][1:]  # 去基期
run_spec(d2y, OUT, ['dw_heat', 'dw_base'], ['dweight'] + basey_cols, 'T2c',
         'race_baseXyearFE', ['dw_heat', 'dw_base'])
# 供热省子样本(剔 4 个结构零省)
d2h = d2[d2['province_en'].map(prov['heat_rigidity']) > 0].copy()
run_spec(d2h, OUT, ['dw_heat'], ['dweight'], 'T2c', 'heating_only_bench', ['dw_heat'])
run_spec(d2h, OUT, ['dw_heat', 'dw_base'], ['dweight'], 'T2c', 'heating_only_race',
         ['dw_heat', 'dw_base'])
# log1p 被解释变量
d2['log_curt'] = np.log1p(d2[OUT])
run_spec(d2, 'log_curt', ['dw_heat'], ['dweight'], 'T2c', 'log_bench', ['dw_heat'])
run_spec(d2, 'log_curt', ['dw_heat', 'dw_base'], ['dweight'], 'T2c', 'log_race',
         ['dw_heat', 'dw_base'])
# 正交化 lock-in
run_spec(d2, OUT, ['dw_heatres'], ['dweight'], 'T2c', 'orth_resid_only', ['dw_heatres'])
run_spec(d2, OUT, ['dw_heatres', 'dw_base'], ['dweight'], 'T2c', 'orth_race',
         ['dw_heatres', 'dw_base'])

# ================= T1 家族 =================
print('== T1 ==')
run_spec(d1, OUT, ['t_heat'], ['dt_post'], 'T1', 'bench_heat', ['t_heat'])
run_spec(d1, OUT, ['t_base'], ['dt_post'], 'T1', 'scale_only', ['t_base'])
run_spec(d1, OUT, ['t_heat', 't_base'], ['dt_post'], 'T1', 'race', ['t_heat', 't_base'])
run_spec(d1, OUT, ['t_heat', 't_base'], ['dt_post', 'base_post'], 'T1', 'race_baseXpost',
         ['t_heat', 't_base'])
run_spec(d1, OUT, ['t_heat', 't_base2018'], ['dt_post', 'base_post'], 'T1', 'race_base2018',
         ['t_heat', 't_base2018'])
run_spec(d1, OUT, ['t_heat', 't_base', 't_basevol'], ['dt_post', 'base_post'], 'T1',
         'race_full', ['t_heat', 't_base', 't_basevol'])

# ================= 输出与判定 =================
res = pd.DataFrame(RESULTS)
res.to_csv(os.path.join(OUTDIR, 'e1_baseline_horserace.csv'), index=False, encoding='utf-8-sig')

def get(eq, spec, coef):
    s = res[(res['eq'] == eq) & (res['spec'] == spec) & (res['coef'] == coef)]
    return s.iloc[0] if len(s) else None

bench = get('T2c', 'bench_heat', 'dw_heat')
race_h = get('T2c', 'race', 'dw_heat')
race_b = get('T2c', 'race', 'dw_base')
sanity = ''
if bench is not None:
    ok = abs(bench['beta'] - 0.229) < 0.01
    sanity = (f"T2c bench β2_heat = {bench['beta']:.3f} (锚点 0.229) -> "
              + ('自检通过' if ok else '⚠️ 自检不通过, 检查数据/设定!'))
print(sanity)

verdict, why = 'INDETERMINATE', ''
if race_h is not None and race_b is not None and bench is not None:
    ratio = race_h['beta'] / bench['beta'] if abs(bench['beta']) > 1e-12 else np.nan
    base_wins = (race_b['p_cluster'] < 0.05) and (race_h['p_cluster'] >= 0.05)
    if ratio >= 0.7 and race_h['p_cluster'] < 0.05:
        verdict, why = 'A_存活', f'赛马后 β2_heat = {race_h["beta"]:.3f} (保留 {ratio:.0%}), p={race_h["p_cluster"]:.4f}'
    elif ratio < 0.4 or base_wins:
        verdict, why = 'C_死亡', f'赛马后 β2_heat = {race_h["beta"]:.3f} (保留 {ratio:.0%}), p={race_h["p_cluster"]:.4f}; base β={race_b["beta"]:.3f}, p={race_b["p_cluster"]:.4f}'
    else:
        verdict, why = 'B_半活', f'赛马后 β2_heat = {race_h["beta"]:.3f} (保留 {ratio:.0%}), p={race_h["p_cluster"]:.4f}'
print(f'预注册判定: {verdict} -- {why}')

# ---------------- 报告 ----------------
def stars(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))

lines = ['# E1 尺度效应 vs 机制 赛马报告', '',
         f'- 生成: 2026-09-30, 脚本 01_analysis_current/e1_baseline_horserace.py',
         f'- 预注册判定: **{verdict}** -- {why}', f'- 自检: {sanity}', '',
         '## 1. 相关矩阵(29 省, 含 p 值)', '',
         '| 变量对 | r | p | n |', '|---|---|---|---|']
for c in corr_rows:
    lines.append(f"| {c['var_a']} × {c['var_b']} | {c['r']:+.3f} | {c['p']:.4f} | {c['n']} |")
lines += ['', '## 2. 赛马结果(被解释变量: wind_curtailment_pct)', '',
          '| 设计 | 设定 | 系数项 | β | se | p(cluster) | p(wildboot) | n |',
          '|---|---|---|---|---|---|---|---|']
for _, r in res.iterrows():
    pb = '' if pd.isna(r['p_wildboot']) else f"{r['p_wildboot']:.4f}"
    lines.append(f"| {r['eq']} | {r['spec']} | {r['coef']} | {r['beta']:.4f}{stars(r['p_cluster'])} "
                 f"| {r['se_cluster']:.4f} | {r['p_cluster']:.4f} | {pb} | {int(r['n_obs'])} |")
lines += ['', '## 3. 诊断与警告', '']
lines += ['- ' + w for w in WARNINGS] if WARNINGS else ['- 无']
lines += ['', '## 4. 预注册判定标准(跑前写明)', '',
          '- 存活(A): 赛马列 β2_heat 量级保留 ≥70% 且 p<0.05',
          '- 半活(B): 40-70% 或边缘显著 -> 降级表述, 主打供热省子样本+月度季节签名',
          '- 死(C): base 显著且 β2_heat <40% 或不显著 -> 标题级结论不成立',
          '- 共线陷阱: corr(heat, base_level)>0.8 时赛马无分辨力, 以子样本/正交化为准']
with open(os.path.join(OUTDIR, 'e1_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'\n完成: {len(res)} 条系数, {len(WARNINGS)} 条诊断')
print('输出: e1_baseline_horserace.csv / e1_correlations.csv / e1_report.md')
