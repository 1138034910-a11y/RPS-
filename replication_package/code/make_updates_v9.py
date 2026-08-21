# -*- coding: utf-8 -*-
"""
v9 update: four journal-grade additions for the RPS/heat-rigidity paper.

  Task 1  fig9_marginal_effect      (png+pdf) + tables/fig9_marginal_effect_data.csv
  Task 2  tables/table3_main_results rewrite: progressive T2c columns (1)-(3)
  Task 3  tables/table5_counterfactual (tex+md) + tables/counterfactual_text.md
  Task 4  fig10_mechanism           (png+pdf)

Panel: master_panel_v8d.csv. T2c spec mirrors redesign_analysis_v8d.py:
  wind_curtailment_pct ~ dw_weak + dweight + controls + C(province_en) + C(year),
  2019-2024, 29 provinces (Tibet/Xinjiang excluded), province-clustered SEs.
heat_z: 2023 base-year heat rigidity (paper scale mean 0.460, sd 0.303),
z-scored over the 29-province estimation sample (scale-free).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, "figs")
TAB = os.path.join(BASE, "tables")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F"
DARK = "#333333"
Z90 = 1.645
EXCL = ["Tibet", "Xinjiang"]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#CCCCCC", "grid.linestyle": "--", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "tight",
})

import re


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def fmtp(p):
    if pd.isna(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tex_escape(s):
    return s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


class Table:
    def __init__(self, headers, rows, caption, label, notes=None):
        self.headers, self.rows, self.caption, self.label = headers, rows, caption, label
        self.notes = notes or []

    def to_tex(self):
        aligns = "l" + "c" * (len(self.headers) - 1)
        out = ["\\begin{table}[htbp]", "\\centering",
               f"\\caption{{{tex_escape(self.caption)}}}", f"\\label{{{self.label}}}",
               "\\begin{threeparttable}", f"\\begin{{tabular}}{{{aligns}}}", "\\toprule",
               " & ".join(tex_escape(h) for h in self.headers) + " \\\\", "\\midrule"]
        for r in self.rows:
            out.append("\\midrule" if r == "MID" else " & ".join(str(c) for c in r) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        if self.notes:
            out.append("\\begin{tablenotes}[flushleft]\\footnotesize")
            out += [f"\\item {n}" for n in self.notes]
            out.append("\\end{tablenotes}")
        out += ["\\end{threeparttable}", "\\end{table}"]
        return "\n".join(out) + "\n"

    def to_md(self):
        out = [f"**{self.caption}**", "",
               "| " + " | ".join(self.headers) + " |",
               "|" + "|".join(["---"] * len(self.headers)) + "|"]
        for r in self.rows:
            if r != "MID":
                out.append("| " + " | ".join(str(c) for c in r) + " |")
        for n in self.notes:
            n = re.sub(r"\\emph\{([^}]*)\}", r"\1", n)
            n = n.replace("$<$", "<").replace("$>$", ">").replace("--", "\u2013").replace("\\%", "%")
            out.append("- " + n)
        return "\n".join(out) + "\n"

    def save(self, name):
        with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as f:
            f.write(self.to_tex())
        with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
            f.write(self.to_md())
        print("saved", name)


# ---------------------------------------------------------------- data & estimation
df = pd.read_csv(os.path.join(BASE, "master_panel_v8d.csv"))
df = df.sort_values(["province_en", "year"]).reset_index(drop=True)

# 29-province 2023 base-year heat rigidity, paper scale (mean ~0.460)
p23 = df[(df["year"] == 2023) & (~df["province_en"].isin(EXCL))].set_index("province_en")
HEAT = (p23["yb_heat_supply_capacity_mw"].fillna(0.0) * 10.0
        / p23["installed_capacity_mw_thermal"]) / 10.0

d = df[(df["year"].between(2018, 2024)) & (~df["province_en"].isin(EXCL))].copy()
d["dweight"] = d.groupby("province_en")["nonhydro_weight_binding"].diff()
d = d[d["year"].between(2019, 2024)].copy()
col = d["province_en"].map(HEAT)
# estimation moments: 174-obs panel (ddof=1) -> mean 0.460, sd 0.303, as in the pipeline
HEAT_M, HEAT_S = col.mean(), col.std()
d["heat_z"] = (col - HEAT_M) / HEAT_S
d["dw_weak"] = d["dweight"] * d["heat_z"]
print(f"heat_rigidity (estimation sample): mean={HEAT_M:.4f} sd={HEAT_S:.4f} "
      f"max={HEAT.max():.3f}")

CTRL = ["wind_cap_growth_pct", "solar_cap_growth_pct", "consumption_growth_pct"]
FITS = {}
for k, ctrls in {1: [], 2: CTRL[:2], 3: CTRL}.items():
    rhs = "dw_weak + dweight" + ("" if not ctrls else " + " + " + ".join(ctrls)) \
          + " + C(province_en) + C(year)"
    m = smf.ols(f"wind_curtailment_pct ~ {rhs}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["province_en"]})
    FITS[k] = m
    print(f"col{k}: b1={m.params['dweight']:.4f} (p={m.pvalues['dweight']:.4f})  "
          f"b2={m.params['dw_weak']:.4f} (se={m.bse['dw_weak']:.4f}, p={m.pvalues['dw_weak']:.6f})  n={int(m.nobs)}")

M3 = FITS[3]
B1, B2 = M3.params["dweight"], M3.params["dw_weak"]
V = M3.cov_params().loc[["dweight", "dw_weak"], ["dweight", "dw_weak"]].values
assert abs(B2 - 0.229) < 0.005, f"col(3) beta2={B2:.4f} deviates from 0.229"
print(f"alignment check passed: col(3) beta2={B2:.6f} vs pipeline 0.229451")

# wild-bootstrap p for the main spec, from the v8d pipeline output
rd8 = pd.read_csv(os.path.join(BASE, "redesign_results_v8d.csv"))
P_BOOT_C3 = rd8[(rd8["eq"] == "T2c") & (rd8["outcome"] == "wind_curtailment_pct")
                & (rd8["weakabs"] == "heat_rigidity") & (rd8["coef"] == "beta2_triple")
                ]["p_wildboot"].iloc[0]
print(f"col(3) wild-boot p from v8d pipeline: {P_BOOT_C3}")


def me_stats(m):
    """ME at mean (=b1) and at +1sd (=b1+b2), with delta-method SEs."""
    b1, b2 = m.params["dweight"], m.params["dw_weak"]
    v = m.cov_params().loc[["dweight", "dw_weak"], ["dweight", "dw_weak"]].values
    return b1, np.sqrt(v[0, 0]), b1 + b2, np.sqrt(v[0, 0] + v[1, 1] + 2 * v[0, 1])


# ================================================================ Task 1: fig 9
def fig9():
    r = np.linspace(0.0, 0.9, 400)
    z = (r - HEAT_M) / HEAT_S
    me = B1 + B2 * z
    se = np.sqrt(V[0, 0] + z ** 2 * V[1, 1] + 2 * z * V[0, 1])
    lo, hi = me - Z90 * se, me + Z90 * se
    out = pd.DataFrame({"heat_rigidity": r, "z_score": z, "ME": me, "SE": se,
                        "ci90_lo": lo, "ci90_hi": hi})
    out.to_csv(os.path.join(TAB, "fig9_marginal_effect_data.csv"), index=False)
    print("saved fig9_marginal_effect_data.csv")

    # thresholds
    z_zero = -B1 / B2                                  # ME = 0
    r_zero = HEAT_M + z_zero * HEAT_S
    sig = r[lo > 0]
    r_sig = sig.min() if len(sig) else None
    me_mean, _, me_1sd, se_1sd = me_stats(M3)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.fill_between(r, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(r, me, color=BLUE, lw=2.0, zorder=4)
    ax.axhline(0, color=DARK, lw=0.9)
    # rug: 29-province moderator values
    rug_y = -0.30
    ax.plot(HEAT.values, np.full(len(HEAT), rug_y), "|", color=ORANGE, ms=9,
            mew=1.4, zorder=5)
    # annotations
    ax.plot([HEAT_M], [B1], "o", color=DARK, ms=5, zorder=6)
    ax.annotate(f"ME at mean lock-in = $\\beta_1$ = {B1:.3f} (n.s.)",
                xy=(HEAT_M, B1), xytext=(0.03, -0.75), fontsize=8.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    ax.plot([HEAT_M + HEAT_S], [me_1sd], "o", color=DARK, ms=5, zorder=6)
    ax.annotate(f"ME at +1 SD lock-in = {me_1sd:.3f}\n($\\beta_2$ = {B2:.3f} per +1 SD, p<0.001)",
                xy=(HEAT_M + HEAT_S, me_1sd), xytext=(0.52, 0.52), fontsize=8.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    ax.axvline(r_zero, color=GRAY, ls=":", lw=1.1)
    ax.text(r_zero + 0.008, -1.32, f"ME = 0 at lock-in\n= {r_zero:.3f}", fontsize=8,
            color=DARK)
    if r_sig is not None and r_sig <= 0.9:
        ax.axvline(r_sig, color=ORANGE, ls="--", lw=1.1)
        ax.text(r_sig + 0.008, 0.62, f"ME significant (90% CI)\nabove lock-in = {r_sig:.3f}",
                fontsize=8, color=ORANGE)
    ax.text(0.03, 0.62, "Rug: 29-province moderator distribution\n(common support; "
            "Hainmueller-Mummolo-Xu 2019)", fontsize=8, color=GRAY)
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-1.6, 0.95)
    ax.set_xlabel("Heating lock-in (2023 base year)")
    ax.set_ylabel("Marginal effect of $\\Delta$weight on\nwind curtailment (pp per pp)")
    fig.savefig(os.path.join(FIGS, "fig9_marginal_effect.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig9_marginal_effect.pdf"))
    plt.close(fig)
    print(f"saved fig9_marginal_effect  (ME=0 at r={r_zero:.3f}; "
          f"90%-significant above r={r_sig if r_sig is None else round(r_sig,3)})")


# ================================================================ Task 2: table 3
def table3():
    def cell(m, key):
        return (f"{fmt(m.params[key])}{stars(m.pvalues[key])} "
                f"({fmt(m.bse[key])})")
    me_rows = [[], [], [], []]
    cols = []
    for k in (1, 2, 3):
        m = FITS[k]
        mm, sm, m1, s1 = me_stats(m)
        cols.append((m, mm, sm, m1, s1))
    rows = [
        ["$\\beta_1$: $\\Delta$weight main effect"] + [cell(c[0], "dweight") for c in cols],
        ["$\\beta_2$: $\\Delta$weight $\\times$ heat rigidity"] + [cell(c[0], "dw_weak") for c in cols],
        ["p (cluster), $\\beta_2$"] + [fmtp(c[0].pvalues["dw_weak"]) for c in cols],
        ["p (wild bootstrap), $\\beta_2$"] + ["--", "--", fmtp(P_BOOT_C3)],
        "MID",
        ["Controls: capacity growth", "No", "Yes", "Yes"],
        ["Controls: consumption growth", "No", "No", "Yes"],
        ["Province FE / Year FE", "Yes / Yes", "Yes / Yes", "Yes / Yes"],
        ["N"] + [int(c[0].nobs) for c in cols],
        ["Provinces", 29, 29, 29],
        "MID",
        ["ME at mean rigidity"] + [f"{fmt(c[1])} ({fmt(c[2])})" for c in cols],
        ["ME at +1 SD rigidity"] + [f"{fmt(c[3])} ({fmt(c[4])})" for c in cols],
    ]
    t = Table(
        ["", "(1) No controls", "(2) + Capacity growth", "(3) Main spec"],
        rows,
        "Table 3. Main results: heat rigidity amplifies the curtailment response",
        "tab:main",
        notes=[
            "Outcome: wind curtailment rate (\\%), 2019--2024, 29 provinces (Tibet and Xinjiang "
            "excluded). Heat rigidity is z-standardized over the 29-province 2023 base-year "
            "distribution (mean 0.460, SD 0.303). Cluster-robust SEs by province in parentheses; "
            "stars from cluster-robust p-values: *** p$<$0.01, ** p$<$0.05, * p$<$0.1.",
            "Wild-cluster bootstrap p-value (999 Rademacher replications, province clusters) for "
            "column (3) from the main estimation pipeline; columns (1)--(2) report cluster-robust "
            "p-values. Marginal-effect rows: ME at mean = $\\beta_1$; ME at +1 SD = $\\beta_1$+"
            "$\\beta_2$; SEs by the delta method from each column's covariance matrix.",
        ],
    )
    t.save("table3_main_results")


# ================================================================ Task 3: counterfactual
def counterfactual():
    d22 = df[(df["year"] == 2022) & (~df["province_en"].isin(EXCL))].set_index("province_en")
    d23b = df[(df["year"] == 2023) & (~df["province_en"].isin(EXCL))].set_index("province_en")
    d24 = df[(df["year"] == 2024) & (~df["province_en"].isin(EXCL))].set_index("province_en")

    sim = pd.DataFrame(index=sorted(HEAT.index))
    sim["rigidity"] = HEAT
    sim["dT_2024"] = d24["delta_target_i"]
    sim["dT_2023"] = d23b["nonhydro_weight_binding"] - d22["nonhydro_weight_binding"]
    sim["excess_E"] = (sim["dT_2024"] - sim["dT_2023"]).clip(lower=0)
    sim["ME"] = B1 + B2 * (sim["rigidity"] - HEAT_M) / HEAT_S
    sim["ME_pos"] = sim["ME"].clip(lower=0)          # only provinces with ME>0 count
    sim["avoided_pp"] = sim["ME_pos"] * sim["excess_E"]
    # wind_generation is in 10^8 kWh (1e8 kWh = 0.1 TWh); national 2024 sum 921.4 TWh
    sim["wind_gen_1e8kwh"] = d24["wind_generation"]
    sim["avoided_twh"] = sim["avoided_pp"] / 100.0 * sim["wind_gen_1e8kwh"] / 10.0
    med = sim["rigidity"].median()
    sim["group"] = np.where(sim["rigidity"] >= med, "High rigidity", "Low rigidity")

    PRICE = 0.35  # CNY per kWh, coal benchmark tariff (rough upper bound)
    sim["value_mio_cny"] = sim["avoided_twh"] * 1e9 * PRICE / 1e6

    tot_E = sim["excess_E"].sum()
    tot_twh = sim["avoided_twh"].sum()
    tot_val = sim["value_mio_cny"].sum()
    n_pos = int((sim["ME"] > 0).sum())
    print(f"\n== Counterfactual: 2024 increments rolled back to 2023 pace ==")
    print(f"national excess pressure sum E = {tot_E:.2f} pp "
          f"(mean {sim['excess_E'].mean():.2f} pp)")
    print(f"provinces with ME>0: {n_pos}/29; ME=0 threshold rigidity = "
          f"{HEAT_M + (-B1 / B2) * HEAT_S:.3f}")
    print(f"avoided wind curtailment energy = {tot_twh:.3f} TWh")
    print(f"economic value @0.35 CNY/kWh = {tot_val:.0f} million CNY "
          f"({tot_val / 100:.2f} yi CNY)")
    print(sim.groupby('group')[['excess_E', 'avoided_pp', 'avoided_twh',
                                'value_mio_cny']].sum().round(3).to_string())

    def grp_row(name, sub):
        wpp = 100 * sub["avoided_twh"].sum() * 10 / sub["wind_gen_1e8kwh"].sum() \
            if sub["wind_gen_1e8kwh"].sum() > 0 else 0.0
        return [name, len(sub), f"{sub['excess_E'].sum():.1f}", f"{wpp:.2f}",
                f"{sub['avoided_twh'].sum():.3f}", f"{sub['value_mio_cny'].sum():.0f}"]

    top5 = sim.sort_values("avoided_twh", ascending=False).head(5)
    rows = [
        grp_row("High heat rigidity ($\\geq$ median)", sim[sim["group"] == "High rigidity"]),
        grp_row("Low heat rigidity ($<$ median)", sim[sim["group"] == "Low rigidity"]),
        grp_row("National (29 provinces)", sim),
        "MID",
    ]
    for p, r in top5.iterrows():
        rows.append([f"\\quad memo: {p}", 1, f"{r['excess_E']:.1f}",
                     f"{r['avoided_pp']:.2f}", f"{r['avoided_twh']:.3f}",
                     f"{r['value_mio_cny']:.0f}"])
    t = Table(
        ["Group", "Provinces", "Excess pressure $\\Sigma E$ (pp)",
         "Avoided curtailment (pp)", "Avoided energy (TWh)", "Value (million CNY)"],
        rows,
        "Table 5. Counterfactual: rolling the 2024 increment back to the 2023 pace",
        "tab:counterfactual",
        notes=[
            "Back-of-envelope partial-equilibrium calculation. Excess pressure $E_i$ = max(0, "
            "$\\Delta T_{i,2024}$ $-$ $\\Delta T_{i,2023}$); avoided curtailment = ME$(r_i)"
            "\\times E_i$, counted only where the marginal effect is positive "
            f"({n_pos} of 29 provinces). Avoided energy applies the avoided rate to 2024 wind "
            "generation (panel column wind\\_generation, in $10^8$ kWh; national total "
            "921.4 TWh). Monetary conversion at 0.35 CNY/kWh coal-benchmark tariff (rough "
            "upper bound).",
            "Caveats: (i) partial equilibrium --- no price or dispatch feedback; (ii) linear "
            "extrapolation of the marginal-effect function; (iii) no supply-side response "
            "(investment, retirement, or operational adaptation).",
        ],
    )
    t.save("table5_counterfactual")

    text = f"""If each province's 2024 increase in the binding non-hydro RPS weight had been
held to its 2023 pace, the aggregate "excess pressure" removed would be {tot_E:.1f}
percentage points (pp) across the 29 sample provinces (mean {sim['excess_E'].mean():.1f} pp).
Mapping this counterfactual through the estimated marginal-effect schedule --- which is
positive only above a heat-rigidity threshold of {HEAT_M + (-B1 / B2) * HEAT_S:.2f}, so that
{n_pos} of 29 provinces contribute --- yields an avoided wind-curtailment increase of about
{tot_twh:.2f} TWh in 2024, roughly {100 * tot_twh * 10 / sim['wind_gen_1e8kwh'].sum():.1f}%
of actual wind generation. The high-rigidity group accounts for the overwhelming share
({sim[sim['group'] == 'High rigidity']['avoided_twh'].sum():.2f} TWh). Valued at the
0.35 CNY/kWh coal-benchmark tariff, the avoided loss is on the order of {tot_val:.0f}
million CNY. This is a deliberately crude back-of-envelope figure: it is partial
equilibrium (no price or dispatch feedback), relies on linear extrapolation of the
marginal effects, and assumes no supply-side response such as investment, retirement, or
operational adaptation.
"""
    with open(os.path.join(TAB, "counterfactual_text.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("saved counterfactual_text.md")
    return sim


# ================================================================ Task 4: fig 10
def fig10():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec, fs=7.6, tc=DARK, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                           fc=fc, ec=ec, lw=1.4, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4, fontweight="bold" if bold else "normal",
                linespacing=1.35)

    def arrow(x1, y1, x2, y2, color=DARK):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                            lw=1.5, color=color, zorder=2, shrinkA=1, shrinkB=1)
        ax.add_patch(a)

    YT, YB, H = 42, 8, 11          # top / bottom lane y, box height
    CT, CB = YT + H / 2, YB + H / 2
    # two-lane flow: col1 -> col2 -> col3 (branch) -> col4 (mechanism) -> col5 (outcome)
    box(1, 26, 17, 12, "RPS target\nacceleration\n(2024, +4.1pp\nmean $\\Delta$weight)",
        "#EAF2F8", BLUE)
    box(22, 26, 17, 12, "Target pressure\non provincial\ndispatch ($\\Delta$weight)",
        "#EAF2F8", BLUE)
    box(43, YT, 17, H, "(a) Flexible\nsystems (low\nheating lock-in)", "#EAF6F0", GREEN,
        bold=True)
    box(43, YB, 17, H, "(b) CHP-locked\nsystems (high\nheating lock-in)", "#FDF0E7", ORANGE,
        bold=True)
    box(64, YT, 17, H, "Absorption via\ntransmission,\nstorage, demand\nresponse",
        "#EAF6F0", GREEN, fs=7.2)
    box(64, YB, 17, H, "Heat-determined\nelectricity output\nfloor (winter\nnights)",
        "#FDF0E7", ORANGE, fs=7.2)
    box(85, YT, 14, H, "No\ncurtailment\nresponse", "white", GREEN, bold=True)
    box(85, YB, 14, H, "Wind\ncurtailment\nrebound\n(+0.96pp)", "white", ORANGE, bold=True,
        fs=7.2)

    arrow(18.6, 32, 21.6, 32)
    arrow(39.6, 34, 42.6, CT - 2, color=GREEN)
    arrow(39.6, 30, 42.6, CB + 2, color=ORANGE)
    arrow(60.6, CT, 63.6, CT, color=GREEN)
    arrow(60.6, CB, 63.6, CB, color=ORANGE)
    arrow(81.6, CT, 84.6, CT, color=GREEN)
    arrow(81.6, CB, 84.6, CB, color=ORANGE)

    # excluded mechanisms dashed box (bottom strip)
    b = FancyBboxPatch((22, 0.4), 77, 5.8, boxstyle="round,pad=0.5,rounding_size=1.4",
                       fc="#F4F4F4", ec=GRAY, lw=1.2, ls="--", zorder=3)
    ax.add_patch(b)
    ax.text(60.5, 3.3, "Excluded competing mechanisms (horse race, Fig. 4):\n"
            "weak export capacity $\\times$,   resource volatility $\\times$,   "
            "pipeline pressure $\\times$",
            ha="center", va="center", fontsize=7.2, color=GRAY, zorder=4,
            linespacing=1.5)

    fig.savefig(os.path.join(FIGS, "fig10_mechanism.png"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig10_mechanism.pdf"))
    plt.close(fig)
    print("saved fig10_mechanism")


if __name__ == "__main__":
    fig9()
    table3()
    counterfactual()
    fig10()
    print("v9 updates done.")
