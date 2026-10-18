"""Feature construction for direct multi-horizon global forecasting.

A row is (origin day tau, horizon h, series s = zone x category); the target is
y[tau+h, s] / scale[tau, s], where scale is the trailing 28-day mean. Normalising by a
local scale lets one global model share structure across series whose volumes differ by
two orders of magnitude (Soria vs Madrid).

Only information available at the end of day tau is used, plus calendar facts and
announced promotions for the target day, and a *noisy* weather forecast whose error grows
with the horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..demand import PRIME_DAY, PRIME_OCT, DemandHistory, black_friday

HORIZONS = np.arange(1, 8)
SCALE_WIN = 28


class FeatureBuilder:
    def __init__(self, hist: DemandHistory, seed: int = 7):
        self.h = hist
        T, J, K = hist.y.shape
        self.T, self.J, self.K, self.S = T, J, K, J * K
        Y = hist.y.reshape(T, -1)
        self.Y = Y
        cs = np.vstack([np.zeros((1, self.S)), np.cumsum(Y, axis=0)])
        self.scale = np.full((T, self.S), np.nan)
        self.scale[SCALE_WIN - 1:] = (cs[SCALE_WIN:] - cs[:-SCALE_WIN]) / SCALE_WIN
        self.ma7 = np.full((T, self.S), np.nan)
        self.ma7[6:] = (cs[7:] - cs[:-7]) / 7
        self.zone_of = np.repeat(np.arange(J), K)
        self.cat_of = np.tile(np.arange(K), J)
        self.country_of = (hist.net.zone_country[self.zone_of] == "PT").astype(int)
        # national / regional momentum per category
        Yk = hist.y.sum(axis=1)                                          # (T,K)
        csk = np.vstack([np.zeros((1, K)), np.cumsum(Yk, axis=0)])
        m7 = np.full((T, K), np.nan); m28 = np.full((T, K), np.nan)
        m7[6:] = (csk[7:] - csk[:-7]) / 7
        m28[27:] = (csk[28:] - csk[:-28]) / 28
        self.nat_mom = m7 / m28
        self.nat_last = Yk / m28                                          # yesterday vs 28d level
        self.Yk, self.m28k = Yk, m28
        self.ly = self._event_aligned_last_year(hist.dates)
        R = hist.region_idx.max() + 1
        Yr = np.zeros((T, R))
        np.add.at(Yr.T, hist.region_idx, hist.y.sum(axis=2).T)
        csr = np.vstack([np.zeros((1, R)), np.cumsum(Yr, axis=0)])
        r7 = np.full((T, R), np.nan); r28 = np.full((T, R), np.nan)
        r7[6:] = (csr[7:] - csr[:-7]) / 7
        r28[27:] = (csr[28:] - csr[:-28]) / 28
        self.reg_mom = (r7 / r28)[:, hist.region_idx]                     # (T,J)
        # calendar
        cal = hist.event_flags.copy()
        self.cal = cal.values.astype(np.float32)
        self.cal_names = list(cal.columns)
        self.any_event = ((np.abs(cal["bf_offset"].values) <= 6) | (cal.drop(columns="bf_offset").values.sum(1) > 0))
        ev_cs = np.concatenate([[0], np.cumsum(self.any_event)])
        self.events_in_window = np.full(T, np.nan)
        self.events_in_window[SCALE_WIN - 1:] = ev_cs[SCALE_WIN:] - ev_cs[:-SCALE_WIN]
        self.dow = hist.dates.dayofweek.values
        self.doy = hist.dates.dayofyear.values
        self.month = hist.dates.month.values
        # noisy weather forecasts, error growing with lead time
        rng = np.random.default_rng(seed)
        self.temp_fc = {}
        self.rain_fc = {}
        for h in HORIZONS:
            sd_t = 0.8 + 0.35 * h
            self.temp_fc[h] = hist.temp + sd_t * rng.standard_normal(hist.temp.shape)
            sd_z = 0.25 + 0.12 * h
            zhat = hist.rain_z + sd_z * rng.standard_normal(hist.rain_z.shape)
            # P(rain) given the noisy latent forecast; threshold implied by realised climatology
            frac = hist.rain.mean(axis=0)
            thr = np.array([np.quantile(hist.rain_z[:, j], 1 - frac[j]) for j in range(J)])
            self.rain_fc[h] = norm.cdf((zhat - thr[None, :]) / sd_z)
        self.temp_clim_anom_fc = {h: self.temp_fc[h] - (hist.temp - hist.temp_anom) for h in HORIZONS}

    @staticmethod
    def _event_aligned_last_year(dates: pd.DatetimeIndex) -> np.ndarray:
        """Index of the comparable day one year earlier (-1 if outside the history).

        Default: same weekday 52 weeks earlier. Inside a retail-event window the day with the
        same offset from last year's event is used instead (Black Friday moves by up to a
        week, Prime events by several)."""
        T = len(dates)
        ly = np.arange(T) - 364
        pos = {d: n for n, d in enumerate(dates.date)}
        for n, d in enumerate(dates.date):
            bf, bf_prev = black_friday(d.year), black_friday(d.year - 1)
            off = (d - bf).days
            cand = None
            if -10 <= off <= 10:
                cand = bf_prev + pd.Timedelta(days=off)
            for ev in (PRIME_DAY, PRIME_OCT):
                if d.year in ev and d.year - 1 in ev:
                    (s0, L0), (s1, L1) = ev[d.year], ev[d.year - 1]
                    o = (d - s0).days
                    if 0 <= o < L0 + 2:
                        # map event days onto last year's event days, post-event days likewise
                        cand = s1 + pd.Timedelta(days=min(o, L1 - 1) if o < L0 else L1 + (o - L0))
            if cand is not None:
                ly[n] = pos.get(cand, -1)
        return ly

    # ------------------------------------------------------------------
    def build(self, origins: np.ndarray, horizons=HORIZONS, with_target: bool = True) -> pd.DataFrame:
        """Feature frame for all (origin, horizon, series) combinations."""
        origins = np.asarray(origins)
        blocks = []
        S = self.S
        sid = np.arange(S)
        for h in horizons:
            t = origins + h
            valid = t < self.T if with_target else np.ones_like(t, dtype=bool)
            o, t = origins[valid], t[valid]
            n = len(o)
            sc = self.scale[o]                                           # (n,S)
            d = {}
            d["origin"] = np.repeat(o, S)
            d["target"] = np.repeat(t, S)
            d["h"] = np.full(n * S, h, dtype=np.int8)
            d["series"] = np.tile(sid, n)
            d["zone"] = np.tile(self.zone_of, n)
            d["cat"] = np.tile(self.cat_of, n)
            d["pt"] = np.tile(self.country_of, n)
            d["log_scale"] = np.log(sc).reshape(-1)
            for lag in range(7):
                d[f"lag{lag}"] = (self.Y[o - lag] / sc).reshape(-1)
            d["same_dow_1w"] = (self.Y[t - 7] / sc).reshape(-1)
            d["same_dow_2w"] = (self.Y[t - 14] / sc).reshape(-1)
            sdow = np.mean([self.Y[t - 7 * m] for m in range(1, 9)], axis=0)
            d["same_dow_8w"] = (sdow / sc).reshape(-1)
            d["ma7"] = (self.ma7[o] / sc).reshape(-1)
            # year-over-year seasonal shape: last year's national level on the target weekday-
            # aligned date relative to last year's 28-day level at the aligned origin
            ty = self.ly[t]
            oy = ty - h
            ly_ok = oy - SCALE_WIN >= 0
            oy, ty = np.maximum(oy, SCALE_WIN), np.maximum(ty, 0)
            nat_yoy = np.where(ly_ok[:, None], self.Yk[ty] / self.m28k[oy], np.nan)
            d["nat_yoy"] = nat_yoy[:, self.cat_of].reshape(-1)
            ser_yoy = np.where(ly_ok[:, None], self.Y[ty] / self.scale[oy], np.nan)
            d["ser_yoy"] = ser_yoy.reshape(-1)
            d["nat_mom"] = self.nat_mom[o][:, self.cat_of].reshape(-1)
            d["nat_last"] = self.nat_last[o][:, self.cat_of].reshape(-1)
            d["reg_mom"] = self.reg_mom[o][:, self.zone_of].reshape(-1)
            d["events_in_window"] = np.repeat(self.events_in_window[o], S)
            d["origin_event"] = np.repeat(self.any_event[o].astype(np.int8), S)
            d["origin_hol"] = self.h.holiday[o][:, self.zone_of].reshape(-1)
            d["dow"] = np.repeat(self.dow[t], S)
            # coarse seasonality: finer day-of-year encodings let trees memorise dates of the
            # two training years (worse on the validation period, see report Sec. 3)
            d["month"] = np.repeat(self.month[t], S).astype(np.int8)
            for c, name in enumerate(self.cal_names):
                d[name] = np.repeat(self.cal[t, c], S)
            d["hol"] = self.h.holiday[t][:, self.zone_of].reshape(-1)
            d["promo"] = self.h.promo[t][:, self.cat_of].reshape(-1)
            d["promo_origin"] = self.h.promo[o][:, self.cat_of].reshape(-1)
            d["temp_fc"] = self.temp_fc[h][t][:, self.zone_of].reshape(-1)
            d["temp_anom_fc"] = self.temp_clim_anom_fc[h][t][:, self.zone_of].reshape(-1)
            d["rain_fc"] = self.rain_fc[h][t][:, self.zone_of].reshape(-1)
            if with_target:
                d["y"] = self.Y[t].reshape(-1)
                d["ratio"] = d["y"] / sc.reshape(-1)
            blocks.append(pd.DataFrame(d))
        df = pd.concat(blocks, ignore_index=True)
        for c in df.columns:
            if df[c].dtype == np.float64 and c not in ("y", "ratio", "log_scale"):
                df[c] = df[c].astype(np.float32)
        return df


def feature_columns(df: pd.DataFrame) -> list:
    drop = {"origin", "target", "series", "y", "ratio"}
    return [c for c in df.columns if c not in drop]
