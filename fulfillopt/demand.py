"""Generative model of daily outbound demand (units to ship) per zone x category.

log mu[t,j,k] =  log base[j,k] + trend[t,j] + weekly[k,dow] + annual[k,country](doy)
               + events[t,k,country] + holidays[t,j] + promo[t,k] + weather[t,j,k]   (deterministic given info)
               + eta[t] + xi[t, region(j)] + zeta[t,k]                                (latent AR(1) shocks)
y[t,j,k] ~ NegBin(mean = mu, dispersion = r_k)

Because the latent states are stored, the *true* conditional distribution of tomorrow's
demand given today's information can be sampled exactly (``DemandHistory.oracle_samples``).
This gives an oracle forecaster and an unbiased way to score the economic value of
forecasts, which is impossible with real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .network import Network

UNITS_PER_KPOP = 7.4   # mean units/day per 1000 inhabitants in 2025 (ES); calibrates ~420k units/day

ES_HOLIDAYS = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]
PT_HOLIDAYS = ["01-01", "04-25", "05-01", "06-10", "08-15", "10-05", "11-01", "12-01", "12-08", "12-25"]
REGIONAL_HOLIDAYS = {
    "CAT": ["06-24", "09-11", "12-26"], "MAD": ["05-02"], "AND": ["02-28"], "GAL": ["05-17", "07-25"],
    "PVA": ["07-25"], "VAL": ["03-19", "10-09"], "ARA": ["04-23"], "CYL": ["04-23"], "CLM": ["05-31"],
    "MUR": ["06-09"], "EXT": ["09-08"], "AST": ["09-08"], "CAN": ["07-28"], "NAV": ["12-03"], "RIO": ["06-09"],
    "PT-N": ["06-24"], "PT-LVT": ["06-13"], "PT-C": [], "PT-S": [],
}
EASTER = {2023: date(2023, 4, 9), 2024: date(2024, 3, 31), 2025: date(2025, 4, 20),
          2026: date(2026, 4, 5), 2027: date(2027, 3, 28)}
# Stylised retail event dates (start day, number of days)
PRIME_DAY = {2023: (date(2023, 7, 11), 2), 2024: (date(2024, 7, 16), 2), 2025: (date(2025, 7, 8), 4),
             2026: (date(2026, 7, 14), 2)}
PRIME_OCT = {2023: (date(2023, 10, 10), 2), 2024: (date(2024, 10, 8), 2), 2025: (date(2025, 10, 7), 2),
             2026: (date(2026, 10, 13), 2)}
INTERIOR = {"MAD", "CLM", "CYL", "ARA", "EXT", "NAV", "RIO"}
WET_NORTH = {"GAL", "AST", "CAN", "PVA", "NAV", "PT-N"}

# weekly multipliers (Mon..Sun): units to *ship*; weekend orders ship Monday
WEEKLY = {
    "S": [1.20, 1.08, 1.02, 0.98, 0.95, 0.84, 0.93],
    "M": [1.22, 1.08, 1.02, 0.97, 0.94, 0.83, 0.94],
    "B": [1.25, 1.10, 1.03, 0.98, 0.93, 0.78, 0.93],
    "A": [1.15, 1.05, 1.00, 0.97, 0.97, 0.90, 0.96],
}


def black_friday(year: int) -> date:
    nov1 = date(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + timedelta(days=22)


@dataclass
class DGPParams:
    growth_es: float = 0.10       # yearly volume growth
    growth_pt: float = 0.16
    pen_pt: float = 0.72          # e-commerce penetration relative to ES
    phi_nat: float = 0.80
    sd_nat: float = 0.035
    phi_reg: float = 0.60
    sd_reg: float = 0.045
    phi_cat: float = 0.70
    sd_cat: float = 0.045
    promo_rate: float = 1 / 28    # flash promotions per category per day
    rain_lift: float = 0.05
    cold_lift_apparel: float = 0.10
    heat_drag: float = -0.03


def _cat_code(c: str) -> str:
    return c[0]


class DemandHistory:
    """A simulated history plus everything needed for oracle conditional sampling."""

    def __init__(self, net: Network, dates: pd.DatetimeIndex, det_log: np.ndarray, eta, xi, zeta,
                 y: np.ndarray, temp: np.ndarray, temp_anom: np.ndarray, rain: np.ndarray, rain_z: np.ndarray,
                 promo: np.ndarray, event_flags: pd.DataFrame, holiday: np.ndarray, region_idx: np.ndarray,
                 params: DGPParams):
        self.net = net
        self.dates = dates
        self.det_log = det_log          # (T,J,K) deterministic log-mean
        self.eta, self.xi, self.zeta = eta, xi, zeta   # (T,), (T,R), (T,K)
        self.y = y                      # (T,J,K) realised demand
        self.temp, self.temp_anom = temp, temp_anom    # (T,J)
        self.rain, self.rain_z = rain, rain_z          # (T,J) bool, latent rain field
        self.promo = promo              # (T,K) log-lift of announced promotions
        self.event_flags = event_flags  # (T, n_events) DataFrame of calendar features
        self.holiday = holiday          # (T,J) 0 none, 1 holiday, 2 day after holiday
        self.region_idx = region_idx    # (J,)
        self.params = params

    @property
    def T(self):
        return len(self.dates)

    def index_of(self, d) -> int:
        return int(self.dates.get_loc(pd.Timestamp(d)))

    def mean(self, t: int) -> np.ndarray:
        """Realised conditional mean mu[t] (J,K) including the latent shocks."""
        return np.exp(self.det_log[t] + self.eta[t] + self.xi[t][self.region_idx][:, None] + self.zeta[t][None, :])

    def oracle_samples(self, t: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples of y[t] (n,J,K) from the true distribution given information at t-1.

        The oracle knows every deterministic driver of day t (calendar, promotions, the
        weather that will occur) and the latent states of day t-1; it does not know the
        day-t shock innovations nor the count noise.
        """
        p = self.params
        eta = p.phi_nat * self.eta[t - 1] + p.sd_nat * rng.standard_normal(n)
        xi = p.phi_reg * self.xi[t - 1][None, :] + p.sd_reg * rng.standard_normal((n, self.xi.shape[1]))
        zeta = p.phi_cat * self.zeta[t - 1][None, :] + p.sd_cat * rng.standard_normal((n, self.zeta.shape[1]))
        log_mu = (self.det_log[t][None] + eta[:, None, None] + xi[:, self.region_idx][:, :, None]
                  + zeta[:, None, :])
        return _negbin(rng, np.exp(log_mu), self.net.cat_dispersion[None, None, :])

    def oracle_mean(self, t: int) -> np.ndarray:
        """E[y_t | info_{t-1}] in closed form (lognormal shock moments)."""
        p = self.params
        m = (self.det_log[t] + p.phi_nat * self.eta[t - 1] + p.phi_reg * self.xi[t - 1][self.region_idx][:, None]
             + p.phi_cat * self.zeta[t - 1][None, :])
        var = p.sd_nat ** 2 + p.sd_reg ** 2 + p.sd_cat ** 2
        return np.exp(m + var / 2)

    def to_frame(self) -> pd.DataFrame:
        """Long format: date, zone, cat, y (+ static attributes)."""
        T, J, K = self.y.shape
        net = self.net
        return pd.DataFrame({
            "date": np.repeat(self.dates.values, J * K),
            "zone": np.tile(np.repeat(np.arange(J), K), T),
            "cat": np.tile(np.arange(K), T * J),
            "y": self.y.reshape(-1),
        }).assign(country=lambda d: net.zone_country[d.zone.values])


def _negbin(rng, mu, r):
    """Gamma-Poisson sampling of NB(mean mu, dispersion r)."""
    lam = rng.gamma(shape=np.broadcast_to(r, mu.shape), scale=mu / r)
    return rng.poisson(lam).astype(np.float64)


def _ar1(rng, T, phi, sd, shape=()):
    x = np.zeros((T,) + shape)
    x[0] = sd / np.sqrt(1 - phi ** 2) * rng.standard_normal(shape)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sd * rng.standard_normal(shape)
    return x


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Retail-event calendar (known in advance), one row per date."""
    f = pd.DataFrame(index=dates)
    days = dates.date
    bf_off = np.full(len(dates), 99)
    pd_on = np.zeros(len(dates))
    pd_after = np.zeros(len(dates))
    po_on = np.zeros(len(dates))
    po_after = np.zeros(len(dates))
    for n, d in enumerate(days):
        bf = black_friday(d.year)
        off = (d - bf).days
        if -10 <= off <= 10:
            bf_off[n] = off
        if d.year in PRIME_DAY:
            s, L = PRIME_DAY[d.year]
            o = (d - s).days
            pd_on[n] = 0 <= o < L
            pd_after[n] = L <= o < L + 2
        if d.year in PRIME_OCT:
            s, L = PRIME_OCT[d.year]
            o = (d - s).days
            po_on[n] = 0 <= o < L
            po_after[n] = L <= o < L + 2
    f["bf_offset"] = bf_off                       # days from Black Friday (99 = far)
    f["prime_day"] = pd_on
    f["prime_day_after"] = pd_after
    f["prime_oct"] = po_on
    f["prime_oct_after"] = po_after
    md = dates.strftime("%m-%d")
    dec = (dates.month == 12)
    f["xmas_ramp"] = np.where(dec & (dates.day <= 20), dates.day / 20.0, 0.0)
    f["xmas_eve_to_26"] = np.isin(md, ["12-24", "12-25", "12-26"]).astype(float)
    f["nye"] = np.isin(md, ["12-31"]).astype(float)
    f["reyes_ramp"] = np.isin(md, ["01-02", "01-03", "01-04"]).astype(float)
    f["rebajas"] = ((dates.month == 1) & (dates.day >= 7) & (dates.day <= 15)).astype(float)
    f["back_to_school"] = ((dates.month == 9) & (dates.day <= 12)).astype(float)
    return f


def _event_log_lift(cal: pd.DataFrame, cat: str, country: str) -> np.ndarray:
    """log multiplier of retail events for one category and country."""
    bf = cal["bf_offset"].values
    bf_curve = {"S": {-4: .15, -3: .2, -2: .25, -1: .3, 0: .85, 1: .7, 2: .5, 3: .75, 4: .45, 5: .25, 6: .12},
                "M": {-4: .15, -3: .2, -2: .25, -1: .3, 0: .9, 1: .75, 2: .55, 3: .8, 4: .5, 5: .3, 6: .15},
                "B": {-4: .2, -3: .25, -2: .3, -1: .35, 0: 1.0, 1: .85, 2: .6, 3: .85, 4: .55, 5: .3, 6: .15},
                "A": {-4: .1, -3: .15, -2: .2, -1: .2, 0: .65, 1: .55, 2: .4, 3: .55, 4: .3, 5: .15, 6: .08}}[cat]
    lift = np.array([bf_curve.get(int(o), 0.0) for o in bf])
    if country == "PT":
        lift *= 0.85
    pdl = {"S": .75, "M": .8, "B": .85, "A": .5}[cat]
    lift += pdl * cal["prime_day"].values + 0.35 * pdl * cal["prime_day_after"].values
    lift += 0.55 * pdl * cal["prime_oct"].values + 0.2 * pdl * cal["prime_oct_after"].values
    xm = {"S": .32, "M": .30, "B": .18, "A": .28}[cat]
    lift += xm * cal["xmas_ramp"].values
    lift += np.log(0.35) * cal["xmas_eve_to_26"].values + np.log(0.6) * cal["nye"].values
    if country == "ES":
        lift += {"S": .30, "M": .28, "B": .1, "A": .15}[cat] * cal["reyes_ramp"].values
    lift += {"S": 0, "M": 0, "B": 0, "A": .35}[cat] * cal["rebajas"].values
    lift += {"S": .10, "M": .05, "B": 0, "A": .12}[cat] * cal["back_to_school"].values
    return lift


def _annual(doy: np.ndarray, cat: str, country: str) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    base = -0.04 * np.cos(w - 0.3) + 0.02 * np.sin(2 * w)
    aug_depth = 0.15 if country == "ES" else 0.08
    aug = -aug_depth * np.exp(-0.5 * ((doy - 225) / 11.0) ** 2)
    if cat == "A":
        base += 0.04 * np.cos(2 * w - 0.5)
    return base + aug


def _holiday_matrix(dates: pd.DatetimeIndex, net: Network) -> np.ndarray:
    """0 = normal, 1 = public holiday in that zone, 2 = first working day after a holiday."""
    T, J = len(dates), net.J
    md = np.array(dates.strftime("%m-%d"))
    hol = np.zeros((T, J), dtype=bool)
    for j in range(J):
        c, r = net.zone_country[j], net.zone_region[j]
        days = set(ES_HOLIDAYS if c == "ES" else PT_HOLIDAYS) | set(REGIONAL_HOLIDAYS.get(r, []))
        hol[:, j] = np.isin(md, list(days))
    for n, d in enumerate(dates.date):
        e = EASTER.get(d.year)
        if e is None:
            continue
        off = (d - e).days
        es = net.zone_country == "ES"
        if off == -2:  # Good Friday everywhere
            hol[n, :] = True
        if off == -3:  # Holy Thursday: Spain except Catalonia/Valencia
            hol[n, es & ~np.isin(net.zone_region, ["CAT", "VAL"])] = True
        if off == 1:   # Easter Monday: Catalonia, Valencia, Basque Country, Navarra
            hol[n, np.isin(net.zone_region, ["CAT", "VAL", "PVA", "NAV"])] = True
        if off == 60:  # Corpus Christi (PT)
            hol[n, ~es] = True
    out = hol.astype(int)
    after = np.zeros_like(hol)
    after[1:] = hol[:-1] & ~hol[1:]
    out[after] = 2
    return out


def simulate(net: Network, start="2023-01-02", end="2026-12-31", seed: int = 2026,
             params: DGPParams | None = None) -> DemandHistory:
    p = params or DGPParams()
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T, J, K = len(dates), net.J, net.K
    cats = [_cat_code(c) for c in net.cat_names]
    doy = dates.dayofyear.values.astype(float)
    dow = dates.dayofweek.values
    years = (dates - pd.Timestamp("2025-01-01")).days.values / 365.25

    regions = sorted(set(net.zone_region))
    region_idx = np.array([regions.index(r) for r in net.zone_region])
    R = len(regions)
    is_pt = net.zone_country == "PT"

    # --- base level per zone x category
    urban = 1 + 0.06 * np.log(np.maximum(net.zone_pop, 50) / 500.0)
    pen = np.where(is_pt, p.pen_pt, 1.0)
    base = UNITS_PER_KPOP * net.zone_pop * urban * pen
    zone_mix = np.exp(0.08 * rng.standard_normal((J, K)))         # local category preferences
    share = net.cat_share[None, :] * zone_mix
    share /= share.sum(axis=1, keepdims=True)
    log_base = np.log(base[:, None] * share)                       # (J,K)

    # --- trend (log-linear with mild saturation)
    g = np.where(is_pt, p.growth_pt, p.growth_es)
    trend = np.log1p(g)[None, :] * years[:, None] - 0.01 * years[:, None] ** 2   # (T,J)

    # --- weekly, annual, events
    cal = calendar_features(dates)
    det = np.empty((T, J, K))
    for k, c in enumerate(cats):
        wk = np.log(np.array(WEEKLY.get(c, WEEKLY["S"])))[dow]          # (T,)
        for country in ("ES", "PT"):
            m = net.zone_country == country
            ann = _annual(doy, c, country)
            ev = _event_log_lift(cal, c, country)
            det[:, m, k] = (log_base[m, k][None, :] + trend[:, m] + (wk + ann + ev)[:, None])

    # --- holidays (zone specific)
    hol = _holiday_matrix(dates, net)
    det += np.where(hol == 1, np.log(0.45), np.where(hol == 2, np.log(1.18), 0.0))[:, :, None]

    # --- announced flash promotions per category (known to forecasters in advance)
    promo = np.zeros((T, K))
    for k in range(K):
        t = 0
        while t < T:
            t += rng.geometric(p.promo_rate)
            if t >= T:
                break
            L = rng.integers(1, 4)
            promo[t:t + L, k] = np.log(rng.uniform(1.12, 1.40))
            t += L
    det += promo[:, None, :]

    # --- weather: temperature anomalies and rain (spatially correlated)
    lat = net.zone_xy[:, 0]
    interior = np.isin(net.zone_region, list(INTERIOR))
    amp = 7.0 + 3.0 * interior
    t_clim = (15.5 - 0.9 * (lat - 40))[None, :] - amp[None, :] * np.cos(2 * np.pi * (doy[:, None] - 20) / 365.25)
    a_nat = _ar1(rng, T, 0.8, 1.2)
    a_reg = _ar1(rng, T, 0.7, 1.3, (R,))
    a_loc = 0.8 * rng.standard_normal((T, J))
    temp_anom = a_nat[:, None] + a_reg[:, region_idx] + a_loc
    temp = t_clim + temp_anom
    p_base = np.where(np.isin(net.zone_region, list(WET_NORTH)), 0.42, np.where(interior, 0.24, 0.17))
    p_rain = np.clip(p_base[None, :] * (1 + 0.65 * np.cos(2 * np.pi * (doy[:, None] - 15) / 365.25)), 0.02, 0.9)
    z_nat = _ar1(rng, T, 0.6, 0.8)
    z_reg = _ar1(rng, T, 0.5, 0.85, (R,))
    rain_z = (np.sqrt(0.35) * z_nat[:, None] / 1.0 + np.sqrt(0.40) * z_reg[:, region_idx] / 0.98
              + np.sqrt(0.25) * rng.standard_normal((T, J)))
    from scipy.stats import norm
    rain = rain_z > norm.ppf(1 - p_rain)
    lift_rain = np.array([p.rain_lift * (0.4 if c == "B" else 1.0) for c in cats])
    lift_cold = np.array([p.cold_lift_apparel if c == "A" else 0.0 for c in cats])
    det += rain[:, :, None] * lift_rain[None, None, :]
    det += (temp_anom < -3.5)[:, :, None] * lift_cold[None, None, :]
    det += (temp > 34)[:, :, None] * p.heat_drag

    # --- latent shocks
    eta = _ar1(rng, T, p.phi_nat, p.sd_nat)
    xi = _ar1(rng, T, p.phi_reg, p.sd_reg, (R,))
    zeta = _ar1(rng, T, p.phi_cat, p.sd_cat, (K,))
    log_mu = det + eta[:, None, None] + xi[:, region_idx][:, :, None] + zeta[:, None, :]
    y = _negbin(rng, np.exp(log_mu), net.cat_dispersion[None, None, :])

    return DemandHistory(net, dates, det, eta, xi, zeta, y, temp, temp_anom, rain, rain_z, promo, cal,
                         hol, region_idx, p)
