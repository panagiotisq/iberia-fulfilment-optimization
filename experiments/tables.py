"""Generate LaTeX tables (report/tables/*.tex) directly from experiment outputs.

No number in the report's tables is typed by hand.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fulfillopt import config, pipeline
from fulfillopt.viz import LABEL

R = config.RESULTS
OUT = config.ROOT / "report" / "tables"


def _write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def eur(x, d=0):
    return f"{x:,.{d}f}".replace(",", "{,}")


def tab_forecast():
    df = pd.read_csv(R / "forecast_metrics.csv").query("split == 'test'")
    order = ["snaive", "snaive_cqr", "chronos_small", "chronos_small_cqr", "chronos_base", "chronos_base_cqr",
             "lgbm", "lgbm_cqr", "oracle"]
    lines = []
    for m in order:
        if m not in set(df.method):
            continue
        g = df[df.method == m].set_index("h")
        a = g.loc[1]
        b = g.loc[7]
        lines.append(f"{LABEL[m]} & {100*a.wmape:.1f} & {100*b.wmape:.1f} & {100*a.bias:+.1f} & {100*a.sql:.2f} & "
                     f"{100*b.sql:.2f} & {100*a.cov80:.1f} & {100*a.cov90:.1f} & {100*a.width80:.1f} \\\\")
        if m in ("snaive_cqr", "chronos_base_cqr", "lgbm_cqr"):
            lines.append("\\midrule")
    body = ("\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule\n"
            " & \\multicolumn{2}{c}{wMAPE (\\%)} & Bias (\\%) & \\multicolumn{2}{c}{SQL (\\%)} & "
            "\\multicolumn{2}{c}{Coverage (\\%)} & Width \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-9}\n"
            "Method & $h{=}1$ & $h{=}7$ & $h{=}1$ & $h{=}1$ & $h{=}7$ & 80\\% & 90\\% & 80\\% \\\\ \\midrule\n"
            + "\n".join(lines[:-1] if lines[-1] == "\\midrule" else lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("forecast", body)


def tab_scenarios():
    df = pd.read_csv(R / "scenario_scores.csv")
    g = df.groupby(["method", "copula"]).agg(es=("es", "mean"), vs=("vs", "mean"), c50=("nat_in50", "mean"),
                                             c90=("nat_in90", "mean"), sd=("nat_sd", "mean"))
    rows = [("snaive_cqr", "indep", "Seasonal naive + CP, independent"), ("snaive_cqr", "copula", "Seasonal naive + CP, copula"),
            ("lgbm_cqr", "indep", "LightGBM-Q + CP, independent"), ("lgbm_cqr", "copula", "LightGBM-Q + CP, copula"),
            ("oracle", "true", "True conditional distribution")]
    lines = []
    for m, c, name in rows:
        r = g.loc[(m, c)]
        lines.append(f"{name} & {r.es:.2f} & {r.vs:.0f} & {100*r.c50:.0f} & {100*r.c90:.0f} & {100*r.sd:.1f} \\\\")
    body = ("\\begin{tabular}{@{}lrrrrr@{}}\\toprule\n"
            "Scenario generator & Energy & Variogram & \\multicolumn{2}{c}{National PI coverage (\\%)} & National \\\\\n"
            " & score & score & 50\\% & 90\\% & CV (\\%) \\\\ \\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("scenarios", body)


def _value_frame():
    d = pd.read_csv(R / "value_of_forecasts.csv")
    ref = d[d.method == "sp_oracle"].set_index("date").exp_total
    d["regret"] = d.exp_total - d.date.map(ref)
    d["regret_pct"] = 100 * d.regret / d.date.map(ref)
    d["event"] = d.date.isin(pipeline.EVENT_DAYS)
    return d


def tab_value():
    d = _value_frame()
    order = ["det_p90", "det_point", "sp_snaive", "sp_lgbm_indep", "sp_lgbm_cop", "sp_oracle"]
    lines = []
    for m in order:
        g = d[d.method == m]
        reg = g[~g.event]
        ev = g[g.event]
        se = reg.regret.std(ddof=1) / np.sqrt(len(reg))
        lines.append(f"{LABEL[m]} & {eur(reg.regret.mean())} & {eur(se)} & {reg.regret_pct.mean():.2f} & "
                     f"{eur(ev.regret.mean())} & {100*g.on_time.mean():.2f} & {100*g.unmet_rate.mean():.2f} & "
                     f"{g.trucks.mean():.0f} & {g.crews.mean():.0f} & {g.solve_time.mean():.1f} \\\\")
    n_reg = (~d[d.method == "sp_oracle"].event).sum()
    n_ev = d[d.method == "sp_oracle"].event.sum()
    body = ("\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\n"
            f" & \\multicolumn{{3}}{{c}}{{Regular days ($n={n_reg}$)}} & Events ($n={n_ev}$) & \\multicolumn{{5}}{{c}}{{All days (means)}} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-10}\n"
            "Plan & Regret & s.e. & \\% & Regret & On-time & Unmet & Trucks & Crews & Time \\\\\n"
            " & \\EUR{}/day & & & \\EUR{}/day & \\% & \\% & & & s \\\\ \\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\\end{tabular}\n")
    _write("value", body)


def tab_saa():
    b = pd.read_csv(R / "saa_bounds.csv")
    lines = []
    for day, g in b.groupby("day", sort=False):
        first = True
        for _, r in g.iterrows():
            lab = f"\\multirow{{{len(g)}}}{{*}}{{{day}}}" if first else ""
            first = False
            lines.append(f"{lab} & {int(r.N)} & {eur(r.lb)} $\\pm$ {eur(r.lb_hw)} & {eur(r.ub_best)} & "
                         f"{eur(r.ub_mean)} & {r.gap_pct:.3f} & {r.gap_ci_pct:.3f} & {r.time:.1f} \\\\")
        lines.append("\\midrule")
    body = ("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n"
            "Day & $N$ & Lower bound (95\\% CI) & Best UB & Mean UB & Gap (\\%) & Gap CI (\\%) & Time (s) \\\\ \\midrule\n"
            + "\n".join(lines[:-1]) + "\n\\bottomrule\\end{tabular}\n")
    _write("saa", body)
    s = pd.read_csv(R / "saa_summary.csv")
    lines = []
    for _, r in s.iterrows():
        lines.append(f"{r.day} & {eur(r.RP)} & {eur(r.EEV)} & {eur(r.VSS)} $\\pm$ {eur(1.96*r.VSS_se)} & "
                     f"{eur(r.WS)} & {eur(r.EVPI)} $\\pm$ {eur(1.96*r.EVPI_se)} & "
                     f"{100*r.RP_unmet:.2f} / {100*r.EEV_unmet:.2f} & {r.RP_trucks:.0f} / {r.EEV_trucks:.0f} & "
                     f"{r.RP_crews:.0f} / {r.EEV_crews:.0f} \\\\")
    body = ("\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule\n"
            "Day & RP & EEV & VSS & WS & EVPI & Unmet \\% & Trucks & Crews \\\\\n"
            " & \\EUR{}/day & \\EUR{}/day & \\EUR{}/day & \\EUR{}/day & \\EUR{}/day & RP / EEV & RP / EEV & RP / EEV \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("vss", body)


def tab_scaling():
    d = pd.read_csv(R / "scaling.csv")
    a = d[d.part == "a"]
    lines = []
    for S, g in a.groupby("S"):
        ef = g[g.method == "extensive"].iloc[0]
        b1 = g[g.method == "benders_w1"].iloc[0]
        b24 = g[g.method == "benders_w24"].iloc[0]
        efs = f"{ef.time:.0f}" + ("$^\\dagger$" if ef.time > 1790 else "")
        efg = f"{100*ef.gap:.3f}" if np.isfinite(ef.gap) else "--"
        sp = ef.time / b24.time
        sps = f"{sp:.1f}" if sp < 10 else f"{sp:.0f}"
        lines.append(f"{int(S)} & {eur(ef.cols)} & {efs} & {efg} & {b1.time:.1f} & {b24.time:.1f} & "
                     f"{100*b24.gap:.3f} & {int(b24.iterations)} & {sps}$\\times$ \\\\")
    body = ("\\begin{tabular}{@{}rrrrrrrrr@{}}\\toprule\n"
            " & \\multicolumn{3}{c}{Extensive form (HiGHS MIP)} & \\multicolumn{4}{c}{Benders} & Speed-up \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
            "$N$ & columns & time (s) & gap (\\%) & 1 worker (s) & 24 workers (s) & gap (\\%) & iter. & \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("scaling", body)
    c = d[d.part == "ablation"]
    order = list(dict.fromkeys(c.method))
    lines = []
    for m in order:
        g = c[c.method == m]
        m_tex = m + ("$^\\dagger$" if (g.gap > 5e-4).any() else "")
        lines.append(f"{m_tex} & {g.time.mean():.1f} & {g.time.max():.1f} & {g.iterations.mean():.0f} & "
                     f"{g.lp_iterations.mean():.0f} & {g.cuts.mean():.0f} & {100*g.gap.max():.3f} & "
                     f"{100*g.t_master.sum()/g.time.sum():.0f} \\\\")
    body = ("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n"
            "Variant & mean time (s) & max time (s) & iterations & LP iter. & cuts & max gap (\\%) & master (\\%) \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("ablation", body)


def tab_online():
    d = pd.read_csv(R / "online.csv")
    hs = d[d.policy == "hindsight"].set_index(["date", "rep"]).cost
    d["regret_pct"] = 100 * (d.cost / d.set_index(["date", "rep"]).index.map(hs) - 1)
    d["on_time"] = 1 - d.late / d.units
    order = ["greedy", "bid_static", "bid_resolve", "lp_tracking", "hindsight"]
    greedy = d[d.policy == "greedy"].regret_pct.mean()
    lines = []
    for m in order:
        g = d[d.policy == m]
        closed = 100 * (1 - g.regret_pct.mean() / greedy) if m != "greedy" else 0.0
        lines.append(f"{LABEL[m]} & {eur(g.cost.mean())} & {g.regret_pct.mean():.2f} & "
                     f"{g.regret_pct.std(ddof=1)/np.sqrt(len(g)):.2f} & {closed:.0f} & {100*g.on_time.mean():.2f} & "
                     f"{100*(g.unmet/g.units).mean():.2f} & {eur(g.spot.mean())} & {eur(g.overtime.mean())} & "
                     f"{g.lp_solves.mean():.0f} \\\\")
    body = ("\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\n"
            "Policy & Recourse cost & Regret & s.e. & Gap closed & On-time & Unmet & Spot & Overtime & LPs \\\\\n"
            " & \\EUR{}/day & \\% & & \\% & \\% & \\% & pkg-eq & labour u. & \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("online", body)


def tab_network():
    d = pd.read_csv(R / "network_design.csv")
    d["event"] = d.date.isin(pipeline.EVENT_DAYS)
    lines = []
    for lam, g in d.groupby("lam"):
        b = g[g.network == "base"].set_index("date")
        c = g[g.network == "with_LIS1"].set_index("date")
        sav = b.exp_total - c.exp_total
        sav_cost = b.exp_cost - c.exp_cost
        ev = b.event
        annual = 365 * sav[~ev].mean()
        lines.append(f"{lam:.1f} & {eur(sav[~ev].mean())} & {eur(sav[~ev].std(ddof=1)/np.sqrt((~ev).sum()))} & "
                     f"{eur(sav_cost[~ev].mean())} & {eur(sav[ev].mean())} & "
                     f"{100*b.on_time_pt.mean():.1f} $\\to$ {100*c.on_time_pt.mean():.1f} & "
                     f"{100*b.on_time.mean():.2f} $\\to$ {100*c.on_time.mean():.2f} & "
                     f"{b.co2_t.mean():.1f} $\\to$ {c.co2_t.mean():.1f} & {annual/1e6:.2f} \\\\")
    body = ("\\begin{tabular}{@{}rrrrrrrrr@{}}\\toprule\n"
            "$\\lambda$ & \\multicolumn{3}{c}{Daily saving, regular days (\\EUR{})} & Saving, & On-time PT & On-time all & CO$_2$ (t/day) & Annualised \\\\\n"
            "\\cmidrule(lr){2-4}\n"
            " & objective & s.e. & cost excl. late & events & (\\%) & (\\%) & & (M\\EUR{}/yr) \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("network", body)


def tab_sensitivity():
    d = pd.read_csv(R / "sensitivity.csv")
    g = d.groupby(["spot_premium", "overtime_frac", "unmet_penalty"]).agg(
        VSS=("VSS", "mean"), VSS_pct=("VSS_pct", "mean"), se=("VSS_se", lambda s: np.sqrt((s ** 2).mean() / len(s))),
        rp_t=("rp_trucks", "mean"), mv_t=("mv_trucks", "mean"), rp_c=("rp_crews", "mean"), mv_c=("mv_crews", "mean"),
        rp_u=("rp_unmet", "mean"), mv_u=("mv_unmet", "mean"), rp_cv=("rp_cvar", "mean"), mv_cv=("mv_cvar", "mean"),
    ).reset_index()
    lines = []
    for _, r in g.iterrows():
        lines.append(f"{r.spot_premium:.1f} & {100*r.overtime_frac:.0f} & {r.unmet_penalty:.0f} & {eur(r.VSS)} & "
                     f"{eur(1.96*r.se)} & {r.VSS_pct:.2f} & {r.rp_t:.0f} / {r.mv_t:.0f} & {r.rp_c:.0f} / {r.mv_c:.0f} & "
                     f"{100*r.rp_u:.2f} / {100*r.mv_u:.2f} & {eur(r.mv_cv - r.rp_cv)} \\\\")
    body = ("\\begin{tabular}{@{}rrrrrrrrrr@{}}\\toprule\n"
            "Spot & Overtime & Unmet & VSS & 95\\% & VSS & Trucks & Crews & Unmet (\\%) & $\\Delta$CVaR$_{95}$ \\\\\n"
            "premium & cap (\\%) & penalty & \\EUR{}/day & $\\pm$ & \\% & RP / EEV & RP / EEV & RP / EEV & \\EUR{}/day \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
    _write("sensitivity", body)


ALL = ["forecast", "scenarios", "value", "saa", "scaling", "online", "network", "sensitivity"]

if __name__ == "__main__":
    for n in sys.argv[1:] or ALL:
        try:
            globals()[f"tab_{n}"]()
        except FileNotFoundError as e:
            print("skip", n, e)
