# -*- coding: utf-8 -*-
"""
make_fig_e1_horserace.py
E1 赛马结果森林图(投稿版): lock-in 交互跨设定稳健性 vs 尺度效应对手
数据: e1_baseline_horserace.csv (T2c 家族)
 claim: The heating lock-in interaction survives — and strengthens — when
        baseline-curtailment interactions enter the same regression.
输出: 改投准备/fig_e1_horserace.{png,pdf}
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = r"D:/Project/01_科研项目/论文3_RPS消纳_RPS"
OUT = os.path.join(BASE, '改投准备', 'fig_e1_horserace')
res = pd.read_csv(os.path.join(BASE, '01_analysis_current', 'e1_baseline_horserace.csv'),
                  encoding='utf-8-sig')

# (标签, spec, 取 lock-in 系数还是对手系数)
SPECS = [
    ('Benchmark (main spec)',          'bench_heat',         ['dw_heat']),
    ('+ 2023 baseline in race',        'race',               ['dw_heat', 'dw_base']),
    ('+ baseline volatility',          'race_plus_vol',      ['dw_heat', 'dw_base', 'dw_basevol']),
    ('+ at-cap dummy (red line)',      'race_plus_atcap',    ['dw_heat', 'dw_base']),
    ('baseline × year FE',             'race_baseXyearFE',   ['dw_heat', 'dw_base']),
    ('pre-treatment 2018 baseline',    'race_base2018',      ['dw_heat', 'dw_base2018']),
    ('+ pre-trend slope',              'race_plus_pretrend', ['dw_heat', 'dw_base', 'dw_pretrend']),
    ('heating provinces only',         'heating_only_race',  ['dw_heat', 'dw_base']),
    ('orthogonalized lock-in',         'orth_race',          ['dw_heatres', 'dw_base']),
]
OPP_LABEL = {'dw_base': 'baseline level', 'dw_basevol': 'baseline volatility',
             'dw_base2018': '2018 baseline', 'dw_pretrend': 'pre-trend slope'}

rows = []
for label, spec, coefs in SPECS:
    for c in coefs:
        s = res[(res['spec'] == spec) & (res['coef'] == c)]
        if len(s) == 0:
            continue
        r = s.iloc[0]
        is_heat = c in ('dw_heat', 'dw_heatres')
        rows.append(dict(label=label, kind='heat' if is_heat else 'opp',
                         opp=OPP_LABEL.get(c, ''), beta=r['beta'],
                         lo=r['beta'] - 1.96 * r['se_cluster'],
                         hi=r['beta'] + 1.96 * r['se_cluster'],
                         p=r['p_cluster']))
D = pd.DataFrame(rows)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'font.size': 7,
    'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8,
})
BLUE, GRAY, RED = '#0072B2', '#8C8C8C', '#D55E00'

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True,
                         gridspec_kw={'width_ratios': [1.25, 1]})
ylabels = [s[0] for s in SPECS]
ypos = np.arange(len(SPECS))[::-1]

def stars(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))

# 左: lock-in 交互
ax = axes[0]
for y, (label, spec, coefs) in zip(ypos, SPECS):
    h = D[(D['label'] == label) & (D['kind'] == 'heat')]
    if len(h) == 0:
        continue
    h = h.iloc[0]
    ax.errorbar(h['beta'], y, xerr=[[h['beta'] - h['lo']], [h['hi'] - h['beta']]],
                fmt='o', color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=2.5, ms=4)
    ax.text(h['hi'] + 0.012, y, f"{h['beta']:.3f}{stars(h['p'])}", va='center',
            fontsize=6.2, color=BLUE)
ax.axvline(0.229, color=RED, ls='--', lw=0.9)
ax.text(0.229, -0.75, 'benchmark = 0.229', color=RED, fontsize=6, ha='center', va='top')
ax.set_ylim(-1.2, len(SPECS))
ax.axvline(0, color='black', lw=0.7)
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels, fontsize=6.8)
ax.set_xlabel(r'$\beta_2$: $\Delta$weight × heating lock-in (pp per pp per SD)', fontsize=7)
ax.set_title('a  Lock-in interaction across specifications', fontsize=7.5, loc='left')
ax.set_xlim(-0.45, 0.55)

# 右: 对手系数(基线族)
ax = axes[1]
seen = set()
for y, (label, spec, coefs) in zip(ypos, SPECS):
    opp = D[(D['label'] == label) & (D['kind'] == 'opp')]
    xoff = 0.0
    for _, o in opp.iterrows():
        lab = o['opp'] if o['opp'] not in seen else ''
        seen.add(o['opp'])
        ax.errorbar(o['beta'], y + xoff, xerr=[[o['beta'] - o['lo']], [o['hi'] - o['beta']]],
                    fmt='s', color=GRAY, ecolor=GRAY, elinewidth=1.1, capsize=2, ms=3.5,
                    label=lab)
        xoff += 0.18
ax.axvline(0, color='black', lw=0.7)
ax.set_xlabel('opponent interaction with $\\Delta$weight', fontsize=7)
ax.set_title('b  Scale-effect opponents (same regressions)', fontsize=7.5, loc='left')
ax.set_xlim(-0.45, 0.55)
ax.legend(loc='lower right', fontsize=5.8, frameon=False, handletextpad=0.2)

fig.tight_layout()
fig.savefig(OUT + '.png', dpi=600, bbox_inches='tight')
fig.savefig(OUT + '.pdf', bbox_inches='tight')
print('saved:', OUT + '.png/.pdf')
