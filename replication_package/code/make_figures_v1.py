# -*- coding: utf-8 -*-
"""
Publication-grade figures for:
"Heating Rigidity and the Limits of Mandates: How Heat-Determined Power Operation
 Amplified Wind Curtailment under China's Accelerated RPS"

Outputs: figs/figN_*.png (300 dpi) + figs/figN_*.pdf (vector)
All English labels, colorblind-safe palette, no CJK glyphs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- paths
BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, "figs")
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------- style
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F"
DARK = "#333333"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#CCCCCC",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})


def save(fig, name):
    png = os.path.join(FIGS, name + ".png")
    pdf = os.path.join(FIGS, name + ".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- data
EXCL = ["Tibet", "Xinjiang"]
panel = pd.read_csv(os.path.join(BASE, "master_panel_v8c.csv"))
p29 = panel[~panel["province_en"].isin(EXCL)].copy()

# heat rigidity, 2023 base year.
# raw scale (yb_heat*10/thermal, 0-8.75) matches gapfill_g2_pretrend.csv;
# paper scale divides by 10 -> mean 0.460, sd 0.307, max 0.875 (Tianjin).
d23 = p29[p29["year"] == 2023].copy()
hr_raw = d23["yb_heat_supply_capacity_mw"].fillna(0.0) * 10.0 / d23["installed_capacity_mw_thermal"]
d23["heat_rigidity"] = (hr_raw / 10.0).where(d23["yb_heat_supply_capacity_mw"].notna(), 0.0)
HEAT = d23.set_index("province_en")["heat_rigidity"]          # 29 provinces, 0 = structural
print("heat_rigidity: mean=%.3f sd=%.3f max=%.3f zeros=%d"
      % (HEAT.mean(), HEAT.std(), HEAT.max(), (HEAT == 0).sum()))

d24 = p29[p29["year"] == 2024].set_index("province_en")
DELTA = d24["delta_target_i"]                                  # 2024 target shock, pp
print("delta_target 2024: mean=%.3f, n>=5pp: %d" % (DELTA.mean(), (DELTA >= 5).sum()))

CURT23 = p29[p29["year"] == 2023].set_index("province_en")["wind_curtailment_pct"]
CURT24 = d24["wind_curtailment_pct"]

Z90 = 1.645  # 90% CI


# ================================================================ Fig 1
def fig1():
    reps = {
        "Heilongjiang": (BLUE, "o", "Heilongjiang / Jilin (large jumps)"),
        "Jilin": (BLUE, "o", None),
        "Tianjin": (ORANGE, "s", "Tianjin / Liaoning (high heating lock-in)"),
        "Liaoning": (ORANGE, "s", None),
        "Guangdong": (GREEN, "^", "Guangdong / Zhejiang (low heating lock-in)"),
        "Zhejiang": (GREEN, "^", None),
    }
    yrs = list(range(2019, 2025))
    piv_w = p29.pivot_table(index="year", columns="province_en", values="nonhydro_weight_binding")
    w25 = p29[p29["year"] == 2025].set_index("province_en")["rps_nonhydro_weight_binding"]
    w26 = p29[p29["year"] == 2026].set_index("province_en")["rps_nonhydro_weight_expected"]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    # representative provinces (thin)
    for prov, (c, mk, lab) in reps.items():
        y = piv_w.loc[yrs, prov].values
        ax.plot(yrs, y, color=c, lw=1.1, marker=mk, ms=3.5, alpha=0.85)
        ax.plot([2024, 2025, 2026], [y[-1], w25[prov], w26[prov]],
                color=c, lw=1.1, ls="--", alpha=0.6)
    # 29-province mean (thick)
    mean = piv_w.loc[yrs].mean(axis=1).values
    ax.plot(yrs, mean, color=DARK, lw=2.6, marker="o", ms=5, zorder=5)
    ax.plot([2024, 2025, 2026], [mean[-1], w25.mean(), w26.mean()],
            color=DARK, lw=2.6, ls="--", zorder=5)

    ax.axvline(2023.5, color=GRAY, ls=":", lw=1.2)
    ax.annotate("2024 acceleration (+4.1pp mean)", xy=(2023.55, 27.5), fontsize=9, color=DARK)
    ax.annotate("2025-26: notified/expected\ntargets (dashed)", xy=(2025.0, 10.2),
                fontsize=8, color=GRAY, ha="center")

    handles = [Line2D([0], [0], color=DARK, lw=2.6, marker="o", ms=5, label="29-province mean")]
    for prov, (c, mk, lab) in reps.items():
        if lab:
            handles.append(Line2D([0], [0], color=c, lw=1.2, marker=mk, ms=4, label=lab))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)
    ax.set_xlim(2018.8, 2026.3)
    ax.set_xticks(range(2019, 2027))
    ax.set_xlabel("Year")
    ax.set_ylabel("Binding non-hydro RPS weight (%)")
    save(fig, "fig1_rps_weights")


# ================================================================ Fig 2
def fig2():
    rates = p29.groupby("year")[["actual_total_rate", "actual_nonhydro_rate"]].mean()
    curt = p29.groupby("year")[["wind_curtailment_pct", "solar_curtailment_pct"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    ax = axes[0]
    yrs = rates.loc[2015:2024].index
    ax.plot(yrs, rates.loc[2015:2024, "actual_total_rate"], color=BLUE, lw=2, marker="o", ms=4,
            label="Total renewables")
    ax.plot(yrs, rates.loc[2015:2024, "actual_nonhydro_rate"], color=ORANGE, lw=2, marker="s", ms=4,
            label="Non-hydro renewables")
    ax.set_xlabel("Year")
    ax.set_ylabel("Consumption rate (%)")
    ax.set_title("(a) Consumption rates, 2015-2024", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_xticks(range(2015, 2025, 3))

    ax = axes[1]
    yrs = curt.loc[2018:2024].index
    ax.plot(yrs, curt.loc[2018:2024, "wind_curtailment_pct"], color=BLUE, lw=2, marker="o", ms=4,
            label="Wind")
    ax.plot(yrs, curt.loc[2018:2024, "solar_curtailment_pct"], color=ORANGE, lw=2, marker="s", ms=4,
            label="Solar")
    w23 = curt.loc[2023, "wind_curtailment_pct"]
    w24 = curt.loc[2024, "wind_curtailment_pct"]
    ax.annotate("+0.96pp, first rebound\nin 7 years", xy=(2023.85, w24 - 0.03), xytext=(2020.6, 2.25),
                fontsize=8.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9))
    ax.set_xlabel("Year")
    ax.set_ylabel("Curtailment rate (%)")
    ax.set_title("(b) Curtailment rates, 2018-2024", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_ylim(0, 2.9)
    fig.tight_layout()
    save(fig, "fig2_trends")


# ================================================================ Fig 3
def fig3():
    hr = HEAT.sort_values(ascending=False)
    colors = [GRAY if v == 0 else BLUE for v in hr.values]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(hr))
    ax.bar(x, hr.values, color=colors, width=0.72)
    n_above = int((hr >= 0.460).sum())
    ax.axvline(n_above - 0.5, color=DARK, ls="--", lw=1.1)
    ax.text(n_above - 0.7, 0.90, "Mean = 0.460", fontsize=9, color=DARK, ha="right")
    for i in range(3):
        ax.text(x[i], hr.values[i] + 0.015, f"{hr.values[i]:.3f}", ha="center", fontsize=8,
                color=DARK, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(hr.index, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Heating lock-in (2023)")
    ax.set_ylim(0, 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="District-heating provinces"),
               plt.Rectangle((0, 0), 1, 1, color=GRAY, label="No heating supply (structural zero)")]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.5,
              bbox_to_anchor=(1.0, 0.93))
    save(fig, "fig3_heat_rigidity")


# ================================================================ Fig 4
def fig4():
    shock = DELTA.sort_values(ascending=False)
    med = HEAT.median()
    colors = [ORANGE if HEAT[p] >= med else BLUE for p in shock.index]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(shock))
    ax.bar(x, shock.values, color=colors, width=0.72)
    ax.axhline(4.06, color=DARK, ls="--", lw=1.1)
    ax.text(len(shock) - 1.5, 4.25, "Mean = 4.06pp", fontsize=9, color=DARK, ha="right")
    ax.text(8.0, 6.55, "12 provinces ≥ 5pp", fontsize=9, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(shock.index, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("2024 target shock $\\Delta$Target (pp)")
    ax.set_ylim(0, 7.9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="High heating lock-in (≥ median)"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Low heating lock-in (< median)")]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.5)
    save(fig, "fig4_delta_target")


# ================================================================ Fig 5
def fig5():
    es = pd.read_csv(os.path.join(BASE, "event_study_v8.csv"))
    w = es[es["outcome"] == "wind_curtailment_pct"].sort_values("year")
    pre = w[w["year"] < 2023]
    base = w[w["year"] == 2023]
    post = w[w["year"] > 2023]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for grp in (pre, post):
        err = Z90 * grp["se"].values
        ax.errorbar(grp["year"], grp["beta"], yerr=err, fmt="o-", color=BLUE, ms=5, lw=1.4,
                    capsize=3, ecolor=BLUE, elinewidth=1.2, zorder=4)
    ax.plot([2022, 2023, 2024],
            [pre[pre["year"] == 2022]["beta"].values[0], 0.0, post["beta"].values[0]],
            color=BLUE, lw=1.4, zorder=3)
    ax.plot(2023, 0.0, "o", color=BLUE, ms=5, zorder=4)
    ax.axhline(0, color=DARK, lw=0.9)
    ax.axvline(2023.5, color=GRAY, ls="--", lw=1.1)
    ax.text(2023.55, 0.95, "2024 acceleration", fontsize=9, color=DARK)
    ax.text(2018.0, -0.62, "Pre-shock coefficients (2018-2022) statistically\n"
            "indistinguishable from zero (min p = 0.19)", fontsize=8.5, color=DARK)
    ax.set_xticks(range(2018, 2025))
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient on heating lock-in intensity ($\\beta$)")
    ax.set_ylim(-0.85, 1.35)
    save(fig, "fig5_event_study")


# ================================================================ Fig 6
def fig6():
    dr = pd.read_csv(os.path.join(BASE, "dose_response_data_v8.csv"))
    gmap = {"Weak absorption (high)": ("High heating lock-in", ORANGE, "s"),
            "Strong absorption (low)": ("Low heating lock-in", BLUE, "o")}
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for g, (lab, c, mk) in gmap.items():
        sub = dr[dr["group"] == g]
        slope, intercept = np.polyfit(sub["delta_target"], sub["d_wind_curt"], 1)
        ax.scatter(sub["delta_target"], sub["d_wind_curt"], color=c, marker=mk, s=38,
                   alpha=0.85, edgecolor="white", linewidth=0.5, zorder=4,
                   label=f"{lab} (slope = {slope:.3f})")
        xs = np.linspace(sub["delta_target"].min(), sub["delta_target"].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=c, lw=1.8, zorder=3)
    ax.axhline(0, color=DARK, lw=0.8, alpha=0.6)
    ax.set_xlabel("2024 target shock $\\Delta$Target (pp)")
    ax.set_ylabel("$\\Delta$ Wind curtailment 2023$\\rightarrow$2024 (pp)")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    save(fig, "fig6_dose_response")


# ================================================================ Fig 7
def fig7():
    rd = pd.read_csv(os.path.join(BASE, "redesign_results_v8.csv"))
    mods = [("weak_export", "Weak export capacity"),
            ("resource_vol", "Resource volatility"),
            ("pipeline_pressure", "Pipeline pressure"),
            ("heat_rigidity", "Heating lock-in")]
    sub = rd[(rd["eq"].isin(["T1", "T2c"])) & (rd["outcome"] == "wind_curtailment_pct")
             & (rd["coef"] == "beta2_triple") & (rd["form"] == "continuous")]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ypos = {m: len(mods) - 1 - i for i, (m, _) in enumerate(mods)}
    ax.axhspan(ypos["heat_rigidity"] - 0.42, ypos["heat_rigidity"] + 0.42,
               color="#F5E6DC", zorder=0)
    for eq, c, mk, off in [("T1", BLUE, "o", -0.14), ("T2c", ORANGE, "s", 0.14)]:
        s = sub[sub["eq"] == eq].set_index("weakabs")
        for m, _ in mods:
            b, se = s.loc[m, "beta"], s.loc[m, "se_cluster"]
            ax.errorbar(b, ypos[m] + off, xerr=Z90 * se, fmt=mk, color=c, ms=6,
                        capsize=3, elinewidth=1.2, zorder=4)
    ax.axvline(0, color=DARK, lw=0.9)
    ax.set_yticks([ypos[m] for m, _ in mods])
    labels = []
    for m, lab in mods:
        labels.append(lab)
    ax.set_yticklabels(labels)
    for tick, (m, _) in zip(ax.get_yticklabels(), mods):
        if m == "heat_rigidity":
            tick.set_fontweight("bold")
    handles = [Line2D([0], [0], marker="o", color=BLUE, lw=1.2, ms=6,
                      label="T1: 2024 single-shock triple interaction"),
               Line2D([0], [0], marker="s", color=ORANGE, lw=1.2, ms=6,
                      label="T2c: full-panel time-varying intensity")]
    ax.legend(handles=handles, frameon=False, loc="center right", fontsize=8.5)
    ax.set_xlabel("Triple-interaction coefficient $\\beta_2$ (90% CI)")
    ax.set_ylim(-0.6, len(mods) - 0.4)
    save(fig, "fig7_horse_race")


# ================================================================ Fig 8
def fig8():
    loo = pd.read_csv(os.path.join(BASE, "robustness_v8_loo.csv"))
    b = loo[loo["spec"] == "baseline"].sort_values("beta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    y = np.arange(len(b))
    err = Z90 * b["se_cluster"].values
    ax.errorbar(b["beta"], y, xerr=err, fmt="o", color=BLUE, ms=4.5, capsize=2.5,
                elinewidth=1.0, zorder=4)
    ax.axvline(0.253, color=ORANGE, ls="--", lw=1.3, zorder=3)
    ax.text(0.258, -0.62, "Full-sample $\\beta_2$ = 0.253", fontsize=9, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels(b["left_out"], fontsize=8)
    ax.set_xlabel("Estimated $\\beta_2$ with province left out (90% CI)")
    ax.set_ylim(-0.8, len(b) - 0.2)
    save(fig, "fig8_loo")


# ================================================================ Appx A1
def figA1():
    sp = pd.read_csv(os.path.join(BASE, "solar_pretrend_by_province.csv"))
    piv = p29.pivot_table(index="year", columns="province_en", values="solar_curtailment_pct")
    yrs = list(range(2018, 2025))
    med = DELTA.median()
    hi = [p for p in piv.columns if DELTA.get(p, np.nan) >= med]
    lo = [p for p in piv.columns if DELTA.get(p, np.nan) < med]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    for p in piv.columns:
        c = ORANGE if p in hi else BLUE
        ax.plot(yrs, piv.loc[yrs, p], color=c, lw=0.6, alpha=0.30)
    ax.plot(yrs, piv.loc[yrs, hi].mean(axis=1), color=ORANGE, lw=2.2, marker="o", ms=4,
            label="High $\\Delta$Target (≥ median), group mean")
    ax.plot(yrs, piv.loc[yrs, lo].mean(axis=1), color=BLUE, lw=2.2, marker="s", ms=4,
            label="Low $\\Delta$Target, group mean")
    ax.axvline(2023.5, color=GRAY, ls="--", lw=1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Solar curtailment (%)")
    ax.set_title("(a) Solar curtailment paths, 2018-2024", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xticks(yrs)

    ax = axes[1]
    top = sp.set_index("province_en")["chg_2018_2022"].sort_values(ascending=False).head(10)
    top = top.iloc[::-1]
    ax.barh(np.arange(len(top)), top.values, color=ORANGE, height=0.65)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Change 2018$\\rightarrow$2022 (pp)")
    ax.set_title("(b) Pre-period risers (top 10)", fontsize=10)
    fig.tight_layout()
    save(fig, "figA1_solar_pretrend")


# ================================================================ Appx A2
def figA2():
    g2 = pd.read_csv(os.path.join(BASE, "gapfill_g2_pretrend.csv"))
    x = g2["heat_rigidity"] / 10.0   # rescale to paper definition (mean 0.460)
    y = g2["wind_pretrend_slope"]
    r = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(x, y, color=BLUE, s=38, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=4)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=ORANGE, lw=1.8, zorder=3)
    ax.axhline(0, color=DARK, lw=0.8, alpha=0.6)
    ax.text(0.40, 0.96, "corr = -0.178 (p = 0.357)\nNo pre-trend confounding",
            transform=ax.transAxes, fontsize=9.5, color=DARK, va="top")
    ax.set_xlabel("Heating lock-in (2023)")
    ax.set_ylabel("Pre-period wind-curtailment trend (pp/yr, 2018-2022)")
    save(fig, "figA2_g2_confounding")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); figA1(); figA2()
    print("all figures done ->", FIGS)
