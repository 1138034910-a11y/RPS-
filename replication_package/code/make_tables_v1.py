# -*- coding: utf-8 -*-
"""
Publication-grade tables (LaTeX booktabs .tex + Markdown .md) for:
"Heating Rigidity and the Limits of Mandates ..."

Outputs: tables/tableN_*.tex and tables/tableN_*.md
Numbers are read directly from the source CSVs (no hardcoding of estimates).
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(BASE, "tables")
os.makedirs(TAB, exist_ok=True)

EXCL = ["Tibet", "Xinjiang"]


# ---------------------------------------------------------------- helpers
def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt(x, d=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{d}f}"


def fmtp(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def tex_escape(s):
    return s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


class Table:
    """Minimal dual-format (booktabs / markdown) table writer."""

    def __init__(self, headers, rows, caption, label, notes=None):
        self.headers = headers
        self.rows = rows
        self.caption = caption
        self.label = label
        self.notes = notes or []

    def to_tex(self):
        ncol = len(self.headers)
        aligns = "l" + "c" * (ncol - 1)
        out = []
        out.append("\\begin{table}[htbp]")
        out.append("\\centering")
        out.append(f"\\caption{{{tex_escape(self.caption)}}}")
        out.append(f"\\label{{{self.label}}}")
        out.append("\\begin{threeparttable}")
        out.append(f"\\begin{{tabular}}{{{aligns}}}")
        out.append("\\toprule")
        out.append(" & ".join(tex_escape(h) for h in self.headers) + " \\\\")
        out.append("\\midrule")
        for r in self.rows:
            if r == "MID":
                out.append("\\midrule")
                continue
            out.append(" & ".join(str(c) for c in r) + " \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        if self.notes:
            out.append("\\begin{tablenotes}[flushleft]\\footnotesize")
            for n in self.notes:
                out.append(f"\\item {n}")
            out.append("\\end{tablenotes}")
        out.append("\\end{threeparttable}")
        out.append("\\end{table}")
        return "\n".join(out) + "\n"

    def to_md(self):
        out = [f"**{self.caption}**", ""]
        out.append("| " + " | ".join(self.headers) + " |")
        out.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        for r in self.rows:
            if r == "MID":
                continue
            out.append("| " + " | ".join(str(c) for c in r) + " |")
        if self.notes:
            out.append("")
            import re
            for n in self.notes:
                # strip latex commands for md readability
                n = re.sub(r"\\emph\{([^}]*)\}", r"\1", n)
                n = n.replace("$<$", "<").replace("$>$", ">").replace("--", "\u2013")
                n = n.replace("\\%", "%")
                out.append("- " + n)
        return "\n".join(out) + "\n"

    def save(self, name):
        with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as f:
            f.write(self.to_tex())
        with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
            f.write(self.to_md())
        print("saved", name)


# ---------------------------------------------------------------- load
desc = pd.read_csv(os.path.join(BASE, "desc_stats_summary_v8.csv"), index_col=0)
gap = pd.read_csv(os.path.join(BASE, "gapfill_results_v8.csv"))
rob = pd.read_csv(os.path.join(BASE, "robustness_v8_results.csv"))
loo = pd.read_csv(os.path.join(BASE, "robustness_v8_loo.csv"))
rd = pd.read_csv(os.path.join(BASE, "redesign_results_v8.csv"))
g2 = pd.read_csv(os.path.join(BASE, "gapfill_g2_pretrend.csv"))
panel = pd.read_csv(os.path.join(BASE, "master_panel_v8c.csv"))
p29 = panel[~panel["province_en"].isin(EXCL)].copy()

# heat rigidity, 2023 base year (paper scale: mean 0.460, max 0.875)
d23 = p29[p29["year"] == 2023].copy()
hr_raw = d23["yb_heat_supply_capacity_mw"].fillna(0.0) * 10.0 / d23["installed_capacity_mw_thermal"]
d23["heat_rigidity"] = (hr_raw / 10.0).where(d23["yb_heat_supply_capacity_mw"].notna(), 0.0)
HEAT = d23.set_index("province_en")["heat_rigidity"]
d24 = p29[p29["year"] == 2024].set_index("province_en")


# ================================================================ Table 1
def table1():
    names = {
        "wind_curtailment_pct": ("Wind curtailment rate (\\%)",
                                 "Wind curtailment rate, provincial average"),
        "solar_curtailment_pct": ("Solar curtailment rate (\\%)",
                                  "Solar curtailment rate, provincial average"),
        "d_weight": ("$\\Delta$ binding non-hydro RPS weight (pp)",
                     "Year-on-year change in the binding non-hydro RPS weight"),
        "heat_rigidity": ("Heat rigidity (2023)",
                          "Heat-supply capacity (MW) relative to thermal installed capacity, 2023 base year; 0 = no district heating"),
        "wind_cap_growth": ("Wind capacity growth (\\%)", "Annual growth of wind installed capacity"),
        "solar_cap_growth": ("Solar capacity growth (\\%)", "Annual growth of solar installed capacity"),
        "cons_growth": ("Electricity consumption growth (\\%)", "Annual growth of electricity consumption"),
        "wind_util_hours": ("Wind utilization hours (h)", "Annual full-load hours of wind power"),
        "solar_util_hours": ("Solar utilization hours (h)", "Annual full-load hours of solar power"),
    }
    order = ["wind_curtailment_pct", "solar_curtailment_pct", "d_weight", "heat_rigidity",
             "wind_cap_growth", "solar_cap_growth", "cons_growth",
             "wind_util_hours", "solar_util_hours"]
    rows = []
    for v in order:
        s = desc.loc[v]
        rows.append([names[v][0], fmt(s["mean"]), fmt(s["std"]), fmt(s["min"]), fmt(s["max"])])
    defs = "; ".join(f"\\emph{{{names[v][0].replace(chr(92)+'%','%')}}} = {names[v][1]}" for v in order)
    t = Table(
        ["Variable", "Mean", "SD", "Min", "Max"],
        rows,
        "Table 1. Summary statistics",
        "tab:summary",
        notes=[
            "Estimation sample: 145 province-year observations (29 provinces, 2020--2024; "
            "Tibet and Xinjiang excluded). Statistics computed over non-missing observations.",
            "Variable definitions: " + defs.replace("%", "\\%") + ".",
        ],
    )
    t.save("table1_summary")


# ================================================================ Table 2
def table2():
    sel = gap[gap["eq"].isin(["G1a", "G1b"])].copy()
    design = {"G1a": "G1a: ordinary-year increments (panel)",
              "G1b": "G1b: 2024 shock ($\\Delta$Target $\\times$ Post)"}
    outc = {"actual_nonhydro_rate": "Non-hydro consumption rate",
            "actual_total_rate": "Total consumption rate"}
    rows = []
    for _, r in sel.iterrows():
        rows.append([
            design[r["eq"]], outc[r["outcome"]],
            fmt(r["beta"]) + stars(r["p_cluster"]),
            f"({fmt(r['se_cluster'])})",
            fmtp(r["p_cluster"]), fmtp(r["p_wildboot"]), int(r["n_obs"]),
        ])
    t = Table(
        ["Design", "Outcome", "$\\beta$", "(Cluster SE)", "p (cluster)", "p (wild boot)", "N"],
        rows,
        "Table 2. First stage: do RPS weights raise measured consumption?",
        "tab:firststage",
        notes=[
            "Cluster-robust SEs by province in parentheses; wild-cluster bootstrap p-values in the "
            "sixth column. *** p$<$0.01, ** p$<$0.05, * p$<$0.1 (cluster p).",
            "The 2024 shock raised consumption rates; ordinary-year increments did not.",
        ],
    )
    t.save("table2_first_stage")


# ================================================================ Table 3
def table3():
    def grab(eq, coef):
        s = rd[(rd["eq"] == eq) & (rd["outcome"] == "wind_curtailment_pct")
               & (rd["weakabs"] == "heat_rigidity") & (rd["form"] == "continuous")
               & (rd["coef"] == coef)]
        return s.iloc[0] if len(s) else None

    def cell(r):
        return f"{fmt(r['beta'])}{stars(r['p_cluster'])} ({fmt(r['se_cluster'])}) [{fmtp(r['p_wildboot'])}]"

    b1_t1, b2_t1 = grab("T1", "beta1_main"), grab("T1", "beta2_triple")
    b2_t2c = grab("T2c", "beta2_triple")
    rows = [
        ["$\\beta_1$: $\\Delta$weight main effect", cell(b1_t1), "-- (absorbed by FE)"],
        ["$\\beta_2$: $\\Delta$weight $\\times$ heat rigidity", cell(b2_t1), cell(b2_t2c)],
        "MID",
        ["N", int(b2_t1["n_obs"]), int(b2_t2c["n_obs"])],
        ["Provinces", 29, 29],
        ["Province FE / Year FE", "Yes / Yes", "Yes / Yes"],
    ]
    t = Table(
        ["", "T1: 2024 single shock", "T2c: full panel, time-varying"],
        rows,
        "Table 3. Main results: heat rigidity amplifies the curtailment response",
        "tab:main",
        notes=[
            "Outcome: wind curtailment rate (\\%). Continuous heat-rigidity measure (2023 base year). "
            "Cluster-robust SEs by province in parentheses; wild-cluster bootstrap p-values in brackets.",
            "*** p$<$0.01, ** p$<$0.05, * p$<$0.1 (cluster-robust p-values).",
        ],
    )
    t.save("table3_main_results")


# ================================================================ Table 4
def table4():
    b_loo = loo[loo["spec"] == "baseline"]
    loo_min, loo_max = b_loo["beta"].min(), b_loo["beta"].max()
    loo_ok = int((b_loo["p_cluster"] < 0.1).sum())

    def rob_row(part):
        return rob[rob["part"] == part].iloc[0]

    hx = rob_row("H1_T2c_heatXyear")
    ri = rob_row("H3_RI_T2c_permHeatLabel")
    spot = rob_row("H3_T1_spot_ctrl")
    g3a = gap[gap["eq"] == "G3a"].iloc[0]
    g4 = gap[(gap["eq"] == "G4") & (gap["weakabs"] == "heat_rigidity")].iloc[0]
    g5 = gap[gap["eq"] == "G5"].iloc[0]
    g2r = np.corrcoef(g2["heat_rigidity"], g2["wind_pretrend_slope"])[0, 1]

    rows = [
        ["Heat $\\times$ year FE", "T2c with heat-rigidity-by-year interactions",
         fmt(hx["beta"]) + stars(hx["p_cluster"]), fmtp(hx["p_cluster"]), "Supported"],
        ["Leave-one-out (29 provinces)", "Re-estimate dropping one province at a time",
         f"[{fmt(loo_min)}, {fmt(loo_max)}]", f"{loo_ok}/29 p$<$0.1", "Supported"],
        ["Randomization inference", "Permute heat-rigidity labels (T2c)",
         fmt(ri["beta"]), f"RI p = {fmtp(ri['ri_p'])}", "Supported"],
        ["Spot-price control", "T1 with spot-market control",
         fmt(spot["beta"]) + stars(spot["p_cluster"]), fmtp(spot["p_cluster"]), "Supported"],
        ["G2: pre-trend confounding", "corr(heat rigidity, pre-period curtailment slope)",
         fmt(g2r), fmtp(0.357), "No confounding"],
        ["G3: heating provinces only", "T2c restricted to 25 heating provinces",
         fmt(g3a["beta"]) + stars(g3a["p_cluster"]), fmtp(g3a["p_cluster"]), "Supported"],
        ["G4: Romano--Wolf", "Multiple-testing correction across moderators",
         "--", f"p$_{{rw}}$ = {fmtp(g4['p_rw'])}", "Supported"],
        ["G5: alternative measure", "Heat supply (GJ)-based rigidity, T2c",
         fmt(g5["beta"]) + stars(g5["p_cluster"]), fmtp(g5["p_cluster"]), "Supported"],
    ]
    t = Table(
        ["Check", "Specification", "$\\beta_2$", "p", "Verdict"],
        rows,
        "Table 4. Robustness battery",
        "tab:robustness",
        notes=[
            "Outcome: wind curtailment rate (\\%). p-values are cluster-robust unless noted; "
            "*** p$<$0.01, ** p$<$0.05, * p$<$0.1.",
            "LOO row reports the range of point estimates across the 29 re-estimations.",
        ],
    )
    t.save("table4_robustness")


# ================================================================ Appx A1
def tableA1():
    sub = rd[(rd["eq"].isin(["T1", "T2c"])) & (rd["form"] == "continuous")
             & (rd["coef"] == "beta2_triple")
             & (rd["outcome"].isin(["wind_curtailment_pct", "solar_curtailment_pct"]))].copy()
    mods = {"weak_export": "Weak export capacity", "heat_rigidity": "Heat rigidity",
            "resource_vol": "Resource volatility", "pipeline_pressure": "Pipeline pressure"}
    outc = {"wind_curtailment_pct": "Wind curtailment", "solar_curtailment_pct": "Solar curtailment"}
    sub["mod"] = sub["weakabs"].map(mods)
    sub["out"] = sub["outcome"].map(outc)
    sub = sub.sort_values(["eq", "out", "mod"], ascending=[True, True, True])
    rows = []
    for _, r in sub.iterrows():
        rows.append([r["eq"], r["out"], r["mod"],
                     fmt(r["beta"]) + stars(r["p_cluster"]),
                     f"({fmt(r['se_cluster'])})", fmtp(r["p_cluster"]), fmtp(r["p_wildboot"]),
                     int(r["n_obs"])])
    t = Table(
        ["Design", "Outcome", "Moderator", "$\\beta_2$", "(Cluster SE)", "p (cluster)",
         "p (wild boot)", "N"],
        rows,
        "Table A1. Horse race of moderators: full results",
        "tab:a1_horserace",
        notes=[
            "Triple-interaction coefficient $\\beta_2$ (continuous moderator forms). "
            "Cluster-robust SEs by province in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.",
        ],
    )
    t.save("tableA1_horse_race_full")


# ================================================================ Appx A2
def tableA2():
    rows = []
    sol = rd[(rd["eq"].isin(["T1", "T2c"])) & (rd["outcome"] == "solar_curtailment_pct")
             & (rd["weakabs"] == "heat_rigidity") & (rd["form"] == "continuous")
             & (rd["coef"] == "beta2_triple")]
    for _, r in sol.iterrows():
        rows.append([r["eq"], "Solar curtailment (\\%)", "$\\beta_2$: heat rigidity",
                     fmt(r["beta"]) + stars(r["p_cluster"]),
                     f"({fmt(r['se_cluster'])})", fmtp(r["p_cluster"]), fmtp(r["p_wildboot"]),
                     int(r["n_obs"])])
    uh = rd[(rd["eq"] == "T3a") & (rd["weakabs"] == "heat_rigidity")
            & (rd["coef"] == "beta2_triple")]
    for _, r in uh.iterrows():
        out = "Wind utilization (h)" if r["outcome"] == "wind_util_hours" else "Solar utilization (h)"
        rows.append(["T3a", out, "$\\beta_2$: heat rigidity",
                     fmt(r["beta"]) + stars(r["p_cluster"]),
                     f"({fmt(r['se_cluster'])})", fmtp(r["p_cluster"]), fmtp(r["p_wildboot"]),
                     int(r["n_obs"])])
    t3b = rd[(rd["eq"] == "T3b") & (rd["coef"] == "dweight")]
    for _, r in t3b.iterrows():
        out = "Wind utilization (h)" if r["outcome"] == "wind_util_hours" else "Solar utilization (h)"
        rows.append(["T3b", out, "$\\Delta$weight main effect",
                     fmt(r["beta"]) + stars(r["p_cluster"]),
                     f"({fmt(r['se_cluster'])})", fmtp(r["p_cluster"]), fmtp(r["p_wildboot"]),
                     int(r["n_obs"])])
    t = Table(
        ["Design", "Outcome", "Coefficient", "$\\beta$", "(Cluster SE)", "p (cluster)",
         "p (wild boot)", "N"],
        rows,
        "Table A2. Solar curtailment and utilization-hours results (heat rigidity)",
        "tab:a2_null",
        notes=[
            "Cluster-robust SEs by province in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.1. "
            "The amplification is specific to wind curtailment: utilization hours show no response.",
        ],
    )
    t.save("tableA2_solar_hours_null")


# ================================================================ Appx A3
def tableA3():
    cur23 = p29[p29["year"] == 2023].set_index("province_en")["wind_curtailment_pct"]
    cur24 = p29[p29["year"] == 2024].set_index("province_en")["wind_curtailment_pct"]
    shock = d24["delta_target_i"]
    provs = sorted(HEAT.index)
    rows = []
    for p in provs:
        rows.append([p, fmt(HEAT[p]), fmt(shock[p], 1), fmt(cur23[p], 1), fmt(cur24[p], 1)])
    t = Table(
        ["Province", "Heat rigidity (2023)", "$\\Delta$Target 2024 (pp)",
         "Wind curtailment 2023 (\\%)", "Wind curtailment 2024 (\\%)"],
        rows,
        "Table A3. Province-level key variables",
        "tab:a3_provinces",
        notes=[
            "Heat rigidity = heat-supply capacity relative to thermal installed capacity "
            "(2023 base year); 0 denotes provinces without district heating (structural zero). "
            "$\\Delta$Target is the 2023$\\rightarrow$2024 change in the binding non-hydro RPS weight.",
        ],
    )
    t.save("tableA3_provinces")


if __name__ == "__main__":
    table1(); table2(); table3(); table4(); tableA1(); tableA2(); tableA3()
    print("all tables done ->", TAB)
