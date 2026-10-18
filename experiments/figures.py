"""All report figures. Each function reads cached experiment outputs; run `python -m experiments.figures [names]`."""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fulfillopt import config, viz
from fulfillopt.viz import CAT, INK2, LABEL, METHOD_COLOR, NEUTRAL, save
from fulfillopt.world import world

R = config.RESULTS


def _sites(ax, net, label=True, fc_size=46):
    ax.scatter(net.sc_xy[:, 1], net.sc_xy[:, 0], marker="s", s=20, color=CAT[1], edgecolor="white", lw=0.6,
               zorder=5)
    ax.scatter(net.fc_xy[:, 1], net.fc_xy[:, 0], marker="^", s=fc_size, color="#2b2a27", edgecolor="white", lw=0.7,
               zorder=6)
    if label:
        offs = {"MAD1": (4, 3), "MAD2": (-4, -9), "BCN1": (-26, -9), "BCN2": (4, 2), "LIS1": (4, -3)}
        for i, n in enumerate(net.fc_names):
            ax.annotate(n, (net.fc_xy[i, 1], net.fc_xy[i, 0]), xytext=offs.get(n, (4, 2)), textcoords="offset points",
                        fontsize=6, color="#2b2a27", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))


def _inset_colorbar(fig, ax, mappable, label, loc=(0.035, 0.06, 0.028, 0.36)):
    """Vertical colorbar inside the map, over the Atlantic."""
    cax = ax.inset_axes(loc)
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cb.set_label(label, fontsize=6.2, labelpad=2)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)
    return cb


def fig_network():
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap
    from fulfillopt import geo_plot
    net, hist = world()
    dem = hist.y[(hist.dates >= "2025-01-01") & (hist.dates < "2026-01-01")].sum(axis=2).mean(axis=0)
    cmap = LinearSegmentedColormap.from_list("seq", viz.SEQ[:6])
    norm = LogNorm(vmin=max(dem.min(), 300), vmax=dem.max())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax = axes[0]
    geo_plot.basemap(ax)
    geo_plot.choropleth(ax, net, dem, cmap, norm)
    for j in range(net.J):
        h = net.zone_sc[j]
        ax.plot([net.zone_xy[j, 1], net.sc_xy[h, 1]], [net.zone_xy[j, 0], net.sc_xy[h, 0]], color="#5d5b55", lw=0.35,
                alpha=0.6, zorder=3)
    _sites(ax, net)
    _inset_colorbar(fig, ax, ScalarMappable(norm, cmap), "units / day (2025)")
    ax.scatter([], [], marker="^", color="#2b2a27", label="fulfilment centre")
    ax.scatter([], [], marker="s", color=CAT[1], label="sort centre")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.85, frameon=True, edgecolor="none")
    ax.set_title("(a) Demand by province, sites and last-mile catchments")
    ax = axes[1]
    geo_plot.basemap(ax)
    for i in range(net.I):
        for h in range(net.H):
            ok = not net.lane_late[i, h]
            if ok:
                ax.plot([net.fc_xy[i, 1], net.sc_xy[h, 1]], [net.fc_xy[i, 0], net.sc_xy[h, 0]], color=CAT[2], lw=1.1,
                        alpha=0.85, zorder=3)
    _sites(ax, net, label=False)
    ax.plot([], [], color=CAT[2], lw=1.2, label="lane within the 6.5 h overnight window")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.85, frameon=True, edgecolor="none")
    ax.set_title(f"(b) Next-day-feasible lanes ({(~net.lane_late).sum()} of {net.lane_late.size})")
    fig.tight_layout()
    save(fig, "network")


def fig_demand():
    net, hist = world()
    tot = pd.Series(hist.y.sum(axis=(1, 2)), index=hist.dates)
    tot = tot[: "2026-10-16"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    ax.plot(tot.index, tot / 1e3, color=CAT[0], lw=0.6)
    ax.plot(tot.rolling(28, center=True).mean() / 1e3, color=INK2, lw=1.2, label="28-day mean")
    ax.axvspan(pd.Timestamp(config.TEST_START), pd.Timestamp("2026-09-30"), color="#eef3fb", zorder=0)
    ax.text(pd.Timestamp("2026-03-31"), 1230, "test year", ha="center", fontsize=7, color=INK2)
    ax.axvspan(pd.Timestamp(config.FORECAST_START), pd.Timestamp(config.TEST_START), color="#f6f1e6", zorder=0)
    ax.text(pd.Timestamp("2025-07-31"), 1230, "val.", ha="center", fontsize=7, color=INK2)
    for d, lab in [("2024-11-29", "BF"), ("2024-07-16", "PD"), ("2025-11-28", "BF"), ("2026-07-14", "PD")]:
        ax.annotate(lab, (pd.Timestamp(d), tot[d] / 1e3), xytext=(0, 4), textcoords="offset points", fontsize=6.5,
                    ha="center", color=INK2)
    ax.set_ylabel("units / day (thousands)")
    ax.set_ylim(0, 1300)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    ax.set_title("(a) Simulated national outbound demand")
    ax = axes[1]
    w = hist.dates.dayofweek.values
    sel = (hist.dates >= "2025-02-01") & (hist.dates < "2025-06-01")
    for k, name in enumerate(net.cat_names):
        prof = pd.Series(hist.y[sel, :, k].sum(1)).groupby(w[sel]).mean()
        ax.plot(range(7), prof / prof.mean(), marker="o", color=CAT[k], label=name, lw=1.2, ms=3.5)
    ax.set_xticks(range(7), ["M", "T", "W", "T", "F", "S", "S"])
    ax.set_ylabel("relative to weekly mean")
    ax.legend(ncol=2, title="category", title_fontsize=7)
    ax.set_title("(b) Weekly shipping profile")
    save(fig, "demand")


def fig_forecast():
    from fulfillopt.forecast.metrics import quantile_hits
    df = pd.read_csv(R / "forecast_metrics.csv").query("split == 'test'")
    methods = [m for m in ["snaive_cqr", "chronos_base_cqr", "lgbm", "lgbm_cqr", "oracle"] if m in set(df.method)]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8))
    axes = axs.ravel()
    for m in methods:
        g = df[df.method == m]
        c = METHOD_COLOR[m]
        ls = "--" if m == "lgbm" else "-"
        axes[0].plot(g.h, 100 * g.wmape, marker="o", color=c, ls=ls, label=LABEL[m], ms=3)
        axes[1].plot(g.h, 100 * g.sql, marker="o", color=c, ls=ls, ms=3)
        axes[2].plot(g.h, 100 * g.cov80, marker="o", color=c, ls=ls, ms=3)
    axes[2].axhline(80, color=INK2, lw=0.8, ls=":")
    axes[0].set_title("(a) wMAPE of the median (%)")
    axes[1].set_title("(b) Scaled quantile loss (%)")
    axes[2].set_title("(c) Coverage of the 80% interval (%)")
    for ax in axes[:3]:
        ax.set_xlabel("horizon (days)")
        ax.set_xticks(range(1, 8))
    axes[0].legend(fontsize=6.5, loc="center right")
    # (d) calibration error per quantile level, h = 1
    net, hist = world()
    f = np.load(config.CACHE / "forecasts.npz")
    o = f["origins"]
    test = o >= hist.index_of(config.TEST_START)
    Y = hist.y.reshape(hist.T, -1)
    levels = f["quantiles"]
    ax = axes[3]
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    for m in ["snaive", "chronos_base", "lgbm", "lgbm_cqr", "oracle"]:
        if m not in f.files:
            continue
        hits = quantile_hits(f[m][test, 0], Y[o[test] + 1])
        ax.plot(levels, 100 * (hits - levels), marker="o", ms=3, color=METHOD_COLOR[m], label=LABEL[m],
                ls="--" if m in ("lgbm", "chronos_base", "snaive") else "-")
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("empirical - nominal (pp)")
    ax.legend(fontsize=6.2, ncol=2, loc="lower left")
    ax.set_title("(d) Quantile calibration error, h = 1")
    fig.tight_layout()
    save(fig, "forecast_accuracy")


def fig_scenarios():
    df = pd.read_csv(R / "scenario_scores.csv")
    g = df.groupby(["method", "copula"]).agg(nat_in90=("nat_in90", "mean"), nat_in50=("nat_in50", "mean"),
                                             es=("es", "mean"), nat_sd=("nat_sd", "mean")).reset_index()
    order = [("snaive_cqr", "indep"), ("snaive_cqr", "copula"), ("lgbm_cqr", "indep"), ("lgbm_cqr", "copula"),
             ("oracle", "true")]
    names = ["S-naive\nindep.", "S-naive\ncopula", "LightGBM\nindep.", "LightGBM\ncopula", "True\ndistr."]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    x = np.arange(len(order))
    v50 = [g.query("method==@m and copula==@c").nat_in50.iloc[0] * 100 for m, c in order]
    v90 = [g.query("method==@m and copula==@c").nat_in90.iloc[0] * 100 for m, c in order]
    ax.bar(x - 0.19, v50, 0.36, color=CAT[0], label="50% interval")
    ax.bar(x + 0.19, v90, 0.36, color=CAT[3], label="90% interval")
    ax.axhline(50, color=CAT[0], lw=0.8, ls=":")
    ax.axhline(90, color=CAT[3], lw=0.8, ls=":")
    ax.set_xticks(x, names, fontsize=6.8)
    ax.set_ylabel("coverage of national total (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", fontsize=6.5, ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("(a) Calibration of the national total")
    # (b) example fan chart of national total over a month for indep vs copula (lgbm)
    ax = axes[1]
    from fulfillopt import pipeline
    from fulfillopt.forecast.scenarios import sample_scenarios
    net, hist = world()
    days = pd.date_range("2026-02-02", "2026-03-15")
    rng = np.random.default_rng(0)
    q = {"indep": [], "copula": []}
    for d in days:
        t = hist.index_of(d)
        qq = pipeline.day_quantiles("lgbm_cqr", t)
        for name, corr in [("indep", None), ("copula", pipeline.day_copula("lgbm_cqr", t))]:
            s = sample_scenarios(qq, 1000, rng, corr).sum(1)
            q[name].append(np.quantile(s, [0.05, 0.5, 0.95]))
    y = hist.y[[hist.index_of(d) for d in days]].sum(axis=(1, 2))
    for name, col in [("copula", CAT[0]), ("indep", CAT[6])]:
        a = np.array(q[name]) / 1e3
        ax.fill_between(days, a[:, 0], a[:, 2], color=col, alpha=0.25 if name == "copula" else 0.45, lw=0,
                        label=f"90% band, {'copula' if name == 'copula' else 'independent'}")
    ax.plot(days, y / 1e3, color="#0b0b0b", lw=0.9, marker="o", ms=2, label="realised")
    ax.set_ylabel("national units (thousands)")
    ax.legend(fontsize=6.2, loc="upper left", ncol=1)
    ax.set_title("(b) Day-ahead national total, Feb-Mar 2026")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save(fig, "scenarios")


def fig_value():
    from fulfillopt import pipeline
    d = pd.read_csv(R / "value_of_forecasts.csv")
    ref = d[d.method == "sp_oracle"].set_index("date").exp_total
    d["regret"] = d.exp_total - d.date.map(ref)
    d["event"] = d.date.isin(pipeline.EVENT_DAYS)
    order = ["det_p90", "sp_snaive", "det_point", "sp_lgbm_indep", "sp_lgbm_cop"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    rng = np.random.default_rng(0)
    for n, m in enumerate(order):
        g = d[(d.method == m) & ~d.event]
        x = n + rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(x, g.regret / 1e3, s=10, color=METHOD_COLOR[m], alpha=0.55, lw=0)
        mu, se = g.regret.mean() / 1e3, g.regret.std(ddof=1) / np.sqrt(len(g)) / 1e3
        ax.errorbar(n + 0.32, mu, yerr=1.96 * se, fmt="o", color=METHOD_COLOR[m], ms=4.5, capsize=2.5, lw=1.2)
        ax.annotate(f"{mu:.1f}", (n + 0.32, mu), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                    va="center", color=INK2)
    ax.set_xticks(range(len(order)), [LABEL[m].replace(" + ", "\n+ ").replace(" (", "\n(") for m in order],
                  fontsize=6.4)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("regret vs SP with true distribution\n(k EUR / day, symlog)")
    ax.set_title(f"(a) Regular days (n={(~d[d.method == 'sp_oracle'].event).sum()}), mean and 95% CI")
    ax = axes[1]
    ev = d[d.event & d.method.isin(["det_point", "sp_lgbm_cop", "sp_snaive"])]
    days = sorted(ev.date.unique())
    w = 0.26
    for n, m in enumerate(["sp_snaive", "det_point", "sp_lgbm_cop"]):
        g = ev[ev.method == m].set_index("date").reindex(days)
        ax.bar(np.arange(len(days)) + (n - 1) * w, np.maximum(g.regret / 1e3, 0.1), w * 0.92,
               color=METHOD_COLOR[m], label=LABEL[m])
    ax.set_yscale("log")
    ax.set_xticks(range(len(days)), [pd.Timestamp(x).strftime("%d %b\n%Y") for x in days], fontsize=6.2)
    ax.set_ylabel("regret (k EUR / day, log)")
    ax.legend(fontsize=6.2, loc="upper right")
    ax.set_title("(b) Event days")
    fig.tight_layout()
    save(fig, "value_of_forecasts")


def fig_saa():
    b = pd.read_csv(R / "saa_bounds.csv")
    runs = pd.read_csv(R / "saa_runs.csv")
    days = list(dict.fromkeys(b.day))
    fig, axes = plt.subplots(1, len(days), figsize=(7.2, 2.5))
    for ax, day in zip(np.atleast_1d(axes), days):
        g = b[b.day == day]
        r = runs[runs.day == day]
        ax.errorbar(g.N, g.lb / 1e6, yerr=g.lb_hw / 1e6, fmt="o-", color=CAT[0], ms=3.5, capsize=2,
                    label="lower bound (mean SAA value, 95% CI)")
        ax.plot(g.N, g.ub_mean / 1e6, "s-", color=CAT[1], ms=3.5, label="out-of-sample cost (mean of plans)")
        ax.plot(g.N, g.ub_best / 1e6, "^-", color=CAT[2], ms=3.5, label="best plan (upper bound)")
        ax.scatter(r.N * np.exp(np.random.default_rng(0).uniform(-0.08, 0.08, len(r))), r.oos / 1e6, s=6,
                   color=CAT[1], alpha=0.35, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("scenarios N")
        ax.set_ylabel("EUR million / day")
        ax.set_title(f"{day}" + (" (Black Friday)" if day == "2025-11-28" else ""))
    np.atleast_1d(axes)[0].legend(fontsize=6.2)
    fig.tight_layout()
    save(fig, "saa")


def fig_convergence():
    import glob
    files = sorted(glob.glob(str(config.CACHE / "conv_2026-03-05_*.csv")))
    if not files:
        return
    names = {"classic": "classic", "+LP_phase": "+ LP phase", "+in-out": "+ in-out", "+Jensen_(phase_1)": "+ Jensen (proposed)"}
    cols = {"classic": CAT[1], "+LP_phase": CAT[3], "+in-out": CAT[6], "+Jensen_(phase_1)": CAT[0]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for f in files:
        key = f.split("conv_2026-03-05_")[1][:-4]
        if key not in names:
            continue
        h = pd.read_csv(f)
        lb = h.lb.cummax()  # LP-phase bounds are valid lower bounds for the MIP too
        lp = h.phase == "lp"
        if lp.any():  # LP phase: gap of the relaxation (dashed)
            best = h.ub[lp].cummin()
            axes[0].plot(h.t[lp], 100 * (best - lb[lp]) / best, color=cols[key], lw=1.0, ls="--")
        best = h.ub[~lp].cummin()  # integer phase: true optimality gap (solid)
        axes[0].plot(h.t[~lp], 100 * (best - lb[~lp]) / best, color=cols[key], label=names[key], lw=1.3)
        axes[1].plot(h.it, h.lb / 1e6, color=cols[key], lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("wall-clock time (s)")
    axes[0].set_ylabel("optimality gap (%)")
    axes[0].axhline(0.05, color=INK2, ls=":", lw=0.8)
    axes[0].plot([], [], color=INK2, lw=1.0, ls="--", label="LP phase (relaxation)")
    axes[0].legend(fontsize=6, loc="upper right", framealpha=0.9)
    axes[0].set_ylim(5e-3, 400)
    axes[0].set_title("(a) Gap vs time (N = 50, 24 workers)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("lower bound (EUR million)")
    axes[1].set_title("(b) Lower bound by iteration")
    fig.tight_layout()
    save(fig, "convergence")


def fig_scaling():
    d = pd.read_csv(R / "scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    a = d[d.part == "a"]
    for m, c, lab in [("extensive", CAT[1], "extensive form (HiGHS)"), ("benders_w1", CAT[3], "Benders, 1 worker"),
                      ("benders_w24", CAT[0], "Benders, 24 workers")]:
        g = a[a.method == m].sort_values("S")
        ax.plot(g.S, g.time, "o-", color=c, label=lab, ms=3.5)
    b = d[d.part == "b"].sort_values("S")
    ax.plot(b.S, b.time, "o--", color=CAT[0], ms=3.5, label="Benders, 24 workers (large N)")
    ax.axhline(1800, color=INK2, ls=":", lw=0.8)
    ax.text(5, 2100, "time limit", fontsize=6.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("scenarios N"); ax.set_ylabel("time to 0.05% gap (s)")
    ax.legend(fontsize=6.3)
    ax.set_title("(a) Scaling with the number of scenarios")
    ax = axes[1]
    sp = d[d.part == "speedup"].sort_values("workers")
    t1 = sp[sp.workers == 1].iloc[0]
    ax.plot(sp.workers, t1.t_sub / sp.t_sub, "o-", color=CAT[0], ms=3.5, label="subproblem phase")
    ax.plot(sp.workers, t1.time / sp.time, "s-", color=CAT[1], ms=3.5, label="total (incl. serial master)")
    ax.plot(sp.workers, sp.workers, color=INK2, ls=":", lw=0.8, label="ideal")
    ax.set_xlabel("worker processes"); ax.set_ylabel("speed-up")
    ax.legend(fontsize=6.3)
    ax.set_title("(b) Parallel speed-up (N = 500)")
    fig.tight_layout()
    save(fig, "scaling")


def fig_pareto():
    d = pd.read_csv(R / "pareto.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    g = d[d.study == "weighted_S100"].sort_values("param")
    base = g.exp_cost.iloc[0]
    ax.plot(100 * g.on_time, (g.exp_cost - base) / 1e3, "o-", color=CAT[0], ms=3.5)
    lab = {0.0: (4, -2), 0.25: (4, -8), 0.5: (-28, 2), 1.0: (-26, 4), 20.0: (-30, 3)}
    for _, r in g.iterrows():
        if r.param in lab:
            ax.annotate(f"$\\lambda$={r.param:g}", (100 * r.on_time, (r.exp_cost - base) / 1e3), xytext=lab[r.param],
                        textcoords="offset points", fontsize=6.2, color=INK2)
    ax.set_xlabel("on-time rate (%)"); ax.set_ylabel("extra cost vs $\\lambda$=0 (k EUR/day)")
    ax.set_title("(a) Price of speed (N=100)")
    ax = axes[1]
    w = d[d.study == "weighted_S10"].sort_values("param")
    e = d[d.study == "epsilon_S10"].sort_values("param")
    b0 = e.ef_obj.min()
    ax.plot(100 * e.ef_on_time, (e.ef_obj - b0) / 1e3, "o-", color=CAT[2], ms=3.2, label="$\\varepsilon$-constraint")
    ax.scatter(100 * w.in_on_time, (w.in_cost - b0) / 1e3, marker="x", s=24, color=CAT[1], label="weighted sum",
               zorder=3)
    ax.set_xlabel("on-time rate (%)"); ax.set_ylabel("extra cost (k EUR/day)")
    ax.legend(fontsize=6.3)
    ax.set_title("(b) Exact frontier (N=10)")
    ax = axes[2]
    c = d[d.study == "carbon"].sort_values("param")
    c0 = c.iloc[0]
    ax.plot(c.co2_t - c0.co2_t, (c.exp_cost - c0.exp_cost) / 1e3, "o-", color=CAT[5], ms=3.5)
    lab = {0: (-4, 5), 800: (-20, 6), 1600: (4, 2), 3200: (4, -3)}
    for _, r in c.iterrows():
        if int(r.param) in lab:
            ax.annotate(f"{r.param:g} EUR/t", (r.co2_t - c0.co2_t, (r.exp_cost - c0.exp_cost) / 1e3),
                        xytext=lab[int(r.param)], textcoords="offset points", fontsize=6.2, color=INK2)
    ax.set_xlabel("change in CO$_2$ (t/day)"); ax.set_ylabel("extra cost (k EUR/day)")
    ax.set_title("(c) Carbon price on line-haul")
    fig.tight_layout()
    save(fig, "pareto")


def fig_online():
    d = pd.read_csv(R / "online.csv")
    hs = d[d.policy == "hindsight"].set_index(["date", "rep"]).cost
    d["regret"] = 100 * (d.cost / d.set_index(["date", "rep"]).index.map(hs) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    order = ["greedy", "bid_static", "bid_resolve", "lp_tracking"]
    rng = np.random.default_rng(1)
    for n, m in enumerate(order):
        g = d[d.policy == m]
        ax.scatter(n + rng.uniform(-0.2, 0.2, len(g)), g.regret, s=7, alpha=0.45, color=METHOD_COLOR[m], lw=0)
        ax.errorbar(n + 0.33, g.regret.mean(), yerr=1.96 * g.regret.std(ddof=1) / np.sqrt(len(g)), fmt="o",
                    color=METHOD_COLOR[m], ms=4.5, capsize=2.5)
        ax.annotate(f"{g.regret.mean():.2f}%", (n + 0.33, g.regret.mean()), xytext=(4, 0), textcoords="offset points",
                    fontsize=6.5, va="center", color=INK2)
    ax.set_xticks(range(4), [LABEL[m].replace(" (", "\n(") for m in order], fontsize=6.6)
    ax.set_ylabel("cost above hindsight LP (%)")
    ax.set_title("(a) Online regret, 31 days x 4 replications")
    ax = axes[1]
    f = R / "online_sensitivity.csv"
    if f.exists():
        s = pd.read_csv(f)
        for (pol, learn), g in s.groupby(["policy", "learn"]):
            m = g.groupby("every").regret.mean() * 100
            ax.plot(m.index / 4, m.values, marker="o", ms=3, color=METHOD_COLOR[pol],
                    ls="-" if learn else "--", label=f"{LABEL[pol]}{'' if learn else ', no learning'}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 24], ["15m", "30m", "1h", "2h", "4h", "8h", "24h"])
        ax.set_xlabel("re-solve interval")
        ax.set_ylabel("mean regret (%)")
        ax.legend(fontsize=6.0)
        ax.set_title("(b) Re-solve frequency and demand learning")
    fig.tight_layout()
    save(fig, "online")


def fig_network_design():
    d = pd.read_csv(R / "network_design.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for k, (lam, g) in enumerate(d.groupby("lam")):
        b = g[g.network == "base"].set_index("date")
        c = g[g.network == "with_LIS1"].set_index("date")
        sav = (b.exp_total - c.exp_total) / 1e3
        dates = pd.to_datetime(sav.index)
        axes[0].plot(dates, sav.values, "o-", ms=3, color=CAT[k * 2], label=f"$\\lambda$ = {lam:g} EUR/late unit", lw=1)
        axes[1].scatter(100 * b.on_time_pt, 100 * c.on_time_pt, s=12, color=CAT[k * 2], label=f"$\\lambda$ = {lam:g}")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("daily saving with LIS1 (k EUR)")
    axes[0].legend(fontsize=6.4)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("(a) Expected daily saving per test day")
    lo = min(axes[1].get_xlim()[0], axes[1].get_ylim()[0])
    axes[1].plot([lo, 100], [lo, 100], color=INK2, ls=":", lw=0.8)
    axes[1].set_xlabel("Portugal on-time, base network (%)")
    axes[1].set_ylabel("with Lisbon FC (%)")
    axes[1].legend(fontsize=6.4)
    axes[1].set_title("(b) Portuguese service level")
    fig.tight_layout()
    save(fig, "network_design")


def _zone_service(net, inv, y, w, scen, lam=None):
    """Per-zone expected on-time rate and served share under plan (y, w) over scenarios."""
    from fulfillopt.planning.benders import SubproblemSolver
    from fulfillopt.planning.model import Recourse
    rec = Recourse(net, inv)
    if lam is not None:
        rec.set_late_penalty(lam)
    sp = SubproblemSolver(rec)
    late = np.zeros(net.J)
    units = np.zeros(net.J)
    flow = np.zeros((net.I, net.H))
    for d in scen:
        _, _, _, x = sp.solve(y, w, d, want_primal=True)
        xa = x[rec.c_x]
        u = x[rec.c_u].reshape(net.J, net.K).sum(1)
        np.add.at(late, rec.arc_j[rec.arc_late], xa[rec.arc_late])
        late += u
        units += d.sum(1)
        np.add.at(flow, (rec.arc_i, rec.arc_h), xa * net.cat_volume[rec.arc_k])
    return 1 - late / units, flow / len(scen)


def fig_plan_map():
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from fulfillopt import geo_plot, pipeline
    net, hist = world()
    day = "2026-03-05"
    t = hist.index_of(day)
    f = np.load(config.CACHE / "plans" / f"{day}_sp_lgbm_cop.npz")
    y, w = f["y"], f["w"]
    inv = pipeline.day_inventory(t)
    scen = hist.oracle_samples(t, 300, np.random.default_rng(5))
    ontime, flow = _zone_service(net, inv, y, w, scen)
    cmap = LinearSegmentedColormap.from_list("ontime", ["#e34948", "#f6d9a8", "#fbfaf7", "#9fd8bf", "#1baf7a"])
    norm = Normalize(0.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ax = axes[0]
    geo_plot.basemap(ax)
    for i in range(net.I):
        for h in range(net.H):
            if y[i, h] > 0:
                ax.plot([net.fc_xy[i, 1], net.sc_xy[h, 1]], [net.fc_xy[i, 0], net.sc_xy[h, 0]],
                        color=CAT[0] if not net.lane_late[i, h] else CAT[1], lw=0.25 + 0.09 * y[i, h], alpha=0.75,
                        solid_capstyle="round", zorder=3)
    _sites(ax, net, label=False, fc_size=0)
    ax.scatter(net.fc_xy[:, 1], net.fc_xy[:, 0], marker="^", s=12 + 1.6 * w, color="#2b2a27", edgecolor="white",
               lw=0.6, zorder=6)
    ax.plot([], [], color=CAT[0], lw=2, label="next-day lane")
    ax.plot([], [], color=CAT[1], lw=2, label="late lane")
    ax.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none",
              title=f"{int(y.sum())} trucks, {int(w.sum())} crews", title_fontsize=6.2)
    ax.set_title(f"(a) Day-ahead plan, {day}: width ~ trucks")
    ax = axes[1]
    geo_plot.basemap(ax, provinces=False)
    geo_plot.choropleth(ax, net, ontime, cmap, norm, edgecolor="#d8d6d0")
    _sites(ax, net, label=False)
    _inset_colorbar(fig, ax, ScalarMappable(norm, cmap), "expected on-time share")
    ax.set_title("(b) Expected next-day delivery by province")
    fig.tight_layout()
    save(fig, "plan_map")


def fig_lisbon_map():
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from fulfillopt import geo, geo_plot
    from fulfillopt.planning.model import PlanningInstance, Recourse, position_inventory
    from fulfillopt.planning.solve import solve_benders
    base, hist = world()
    cand = base.add_fcs(geo.CANDIDATE_FCS)
    day = "2026-03-05"
    t = hist.index_of(day)
    scen = hist.oracle_samples(t, 100, np.random.default_rng(t))
    ev = hist.oracle_samples(t, 300, np.random.default_rng(6))
    res = {}
    for name, net in [("base", base), ("cand", cand)]:
        inv = position_inventory(net, hist.oracle_mean(t))
        plan = solve_benders(PlanningInstance(net, inv, scen), Recourse(net, inv), workers=16)
        res[name] = (plan, _zone_service(net, inv, plan.y, plan.w, ev))
    delta = 100 * (res["cand"][1][0] - res["base"][1][0])
    cmap = LinearSegmentedColormap.from_list("gain", ["#fbfaf7", "#b5e3cf", "#1baf7a", "#0b6b48"])
    norm = Normalize(0, max(5.0, np.ceil(delta.max() / 10) * 10))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ax = axes[0]
    geo_plot.basemap(ax, provinces=False)
    geo_plot.choropleth(ax, base, np.clip(delta, 0, None), cmap, norm, edgecolor="#d8d6d0")
    _sites(ax, cand, label=False)
    i = cand.fc_names.index("LIS1")
    ax.scatter([cand.fc_xy[i, 1]], [cand.fc_xy[i, 0]], marker="*", s=140, color=CAT[0], edgecolor="white", lw=0.8,
               zorder=8, label="candidate FC (Azambuja)")
    ax.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none")
    _inset_colorbar(fig, ax, ScalarMappable(norm, cmap), "on-time gain (pp)")
    ax.set_title(f"(a) Change in next-day delivery with LIS1 ({day})")
    ax = axes[1]
    geo_plot.basemap(ax)
    pt_sc = [h for h, n in enumerate(cand.sc_names) if n in ("SC-LIS", "SC-OPO")]
    scale = 0.00011
    for name, col, off in [("base", CAT[1], -0.06), ("cand", CAT[0], 0.06)]:
        net_ = base if name == "base" else cand
        flow = res[name][1][1]
        for h in pt_sc:
            for k in range(net_.I):
                v = flow[k, h]
                if v < 300:
                    continue
                ax.plot([net_.fc_xy[k, 1] + off, net_.sc_xy[h, 1] + off], [net_.fc_xy[k, 0] + off, net_.sc_xy[h, 0] + off],
                        color=col, lw=0.3 + scale * v, alpha=0.75, solid_capstyle="round", zorder=3)
    _sites(ax, cand, label=False)
    ax.scatter([cand.fc_xy[i, 1]], [cand.fc_xy[i, 0]], marker="*", s=140, color=CAT[0], edgecolor="white", lw=0.8,
               zorder=8)
    for h in pt_sc:
        tot_b = res["base"][1][1][:, h].sum() / 1e3
        late_b = (res["base"][1][1][:, h] * base.lane_late[:, h]).sum() / 1e3
        ax.annotate(f"{cand.sc_names[h]}: {tot_b:.0f}k pkg-eq/day\nlate lanes: {late_b:.0f}k -> "
                    f"{(res['cand'][1][1][:, h] * cand.lane_late[:, h]).sum() / 1e3:.0f}k",
                    (cand.sc_xy[h, 1], cand.sc_xy[h, 0]), xytext=(-12.0, cand.sc_xy[h, 0] + (0.5 if h == pt_sc[1] else -1.6)),
                    textcoords="data", fontsize=5.8, color=INK2, zorder=9, ha="left",
                    arrowprops=dict(arrowstyle="-", color="#b0aea7", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.plot([], [], color=CAT[1], lw=2, label="base network")
    ax.plot([], [], color=CAT[0], lw=2, label="with LIS1")
    ax.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none",
              title="flows into PT sort centres", title_fontsize=6)
    ax.set_title("(b) Where Portuguese volume ships from")
    fig.tight_layout()
    save(fig, "lisbon_map")


def fig_flex():
    d = pd.read_csv(R / "sensitivity.csv")
    d = d[d.unmet_penalty == 15.0]
    d["dcvar"] = d.mv_cvar - d.rp_cvar
    d["dtrucks"] = d.rp_trucks - d.mv_trucks
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for k, (ot, g) in enumerate(d.groupby("overtime_frac")):
        m = g.groupby("spot_premium")[["VSS", "dcvar", "dtrucks"]].mean()
        lab = f"overtime cap {100 * ot:.0f}%"
        col = CAT[0] if ot > 0.2 else CAT[1]
        axes[0].plot(m.index, m.VSS / 1e3, "o-", color=col, ms=3.5, label=lab)
        axes[0].plot(m.index, m.dcvar / 1e3, "s--", color=col, ms=3, alpha=0.8)
        axes[1].plot(m.index, m.dtrucks, "o-", color=col, ms=3.5, label=lab)
    axes[0].plot([], [], "k-", lw=1, label="VSS (mean)")
    axes[0].plot([], [], "k--", lw=1, label="CVaR$_{95}$ reduction")
    axes[0].set_xlabel("spot premium (x scheduled rate)"); axes[0].set_ylabel("k EUR / day")
    axes[0].legend(fontsize=5.8)
    axes[0].set_title("(a) Value of the stochastic solution")
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].axvline(2.0, color=INK2, lw=0.8, ls=":")
    axes[1].text(2.05, axes[1].get_ylim()[0] * 0.85, "critical\nratio = 0.5", fontsize=6, color=INK2)
    axes[1].set_xlabel("spot premium (x scheduled rate)"); axes[1].set_ylabel("trucks: SP plan - mean plan")
    axes[1].legend(fontsize=6)
    axes[1].set_title("(b) Direction of the hedge")
    r = pd.read_csv(R / "sensitivity_forecast_regimes.csv")
    P = r.pivot_table(index=["day", "regime"], columns="method", values="exp_total")
    ax = axes[2]
    rng = np.random.default_rng(2)
    for n, reg in enumerate(["base", "inflexible"]):
        x = P.xs(reg, level=1)
        for k, m in enumerate(["sp_lgbm_indep", "sp_lgbm_cop"]):
            sav = (x["det_point"] - x[m]) / 1e3
            pos = n + (k - 0.5) * 0.36
            ax.scatter(pos + rng.uniform(-0.07, 0.07, len(sav)), sav, s=9, color=METHOD_COLOR[m], alpha=0.6, lw=0)
            ax.plot([pos - 0.13, pos + 0.13], [sav.mean()] * 2, color=METHOD_COLOR[m], lw=2,
                    label=LABEL[m] if n == 0 else None)
            ax.annotate(f"{sav.mean():.0f}", (pos + 0.14, sav.mean()), fontsize=6.2, color=INK2, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks([0, 1], ["base recourse", "inflexible recourse\n(spot 3.5x, OT 10%)"], fontsize=6.6)
    ax.set_ylabel("saving vs deterministic plan\n(k EUR/day, symlog)")
    ax.legend(fontsize=5.8, loc="upper left")
    ax.set_title("(c) Forecast-based plans, paired by day")
    fig.tight_layout()
    save(fig, "flex")


ALL = ["network", "demand", "forecast", "scenarios", "value", "saa", "convergence", "scaling",
       "pareto", "online", "network_design", "plan_map", "lisbon_map", "flex"]

if __name__ == "__main__":
    viz.setup()
    names = sys.argv[1:] or ALL
    for n in names:
        globals()[f"fig_{n}"]()
        print("ok", n)
