"""E1b - Zero-shot foundation-model baseline: Chronos-Bolt (Ansari et al., 2024).

Chronos-Bolt is used without fine-tuning and without covariates (it only sees the history of
each series), on the same rolling origins as E1. It predicts the 0.1..0.9 quantiles; the
0.05/0.95 levels are obtained by linear extrapolation in normal-score space (same rule as
the scenario generator) and 0.25/0.75 by interpolation. Results are appended to
forecasts.npz and forecast_metrics.csv.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from scipy.stats import norm

from fulfillopt import config
from fulfillopt.forecast.features import HORIZONS
from fulfillopt.forecast.models import QUANTILES, conformalize
from fulfillopt.world import world

BOLT_LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)
CONTEXT = 2048


def map_levels(qb: np.ndarray) -> np.ndarray:
    """(..., 9) Chronos-Bolt quantiles -> (..., 7) at QUANTILES, in normal-score space."""
    zb, zt = norm.ppf(BOLT_LEVELS), norm.ppf(QUANTILES)
    qb = np.sort(qb, axis=-1)
    out = np.empty(qb.shape[:-1] + (len(QUANTILES),))
    for a, z in enumerate(zt):
        if z < zb[0]:
            i0, i1 = 0, 1
        elif z > zb[-1]:
            i0, i1 = len(zb) - 2, len(zb) - 1
        else:
            i1 = int(np.searchsorted(zb, z)); i1 = max(i1, 1); i0 = i1 - 1
        w = (z - zb[i0]) / (zb[i1] - zb[i0])
        out[..., a] = qb[..., i0] + w * (qb[..., i1] - qb[..., i0])
    return np.maximum(out, 0)


def run(model_name: str, tag: str):
    from chronos import BaseChronosPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, hist = world()
    f = dict(np.load(config.CACHE / "forecasts.npz"))
    origins = f["origins"]
    Y = hist.y.reshape(hist.T, -1)
    pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=dev, torch_dtype=torch.float32)
    S = Y.shape[1]
    out = np.empty((len(origins), len(HORIZONS), S, len(QUANTILES)), dtype=np.float32)
    t0 = time.time()
    B = 4  # origins per forward pass (4 x 260 series, fits 16 GB at 2048 context)
    for b0 in range(0, len(origins), B):
        ob = origins[b0:b0 + B]
        ctx = [torch.tensor(Y[max(0, o + 1 - CONTEXT):o + 1, s], dtype=torch.float32) for o in ob for s in range(S)]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=len(HORIZONS), quantile_levels=list(BOLT_LEVELS))
        q = map_levels(q.cpu().numpy()).reshape(len(ob), S, len(HORIZONS), len(QUANTILES))
        out[b0:b0 + len(ob)] = q.transpose(0, 2, 1, 3)
        print(tag, b0, f"{time.time() - t0:.0f}s", flush=True)
    f[tag] = out
    groups = np.tile(np.arange(net.K), net.J)
    f[tag + "_cqr"] = conformalize(out.astype(float), origins, Y, groups=groups).astype(np.float32)
    np.savez_compressed(config.CACHE / "forecasts.npz", **f)


if __name__ == "__main__":
    which = sys.argv[1:] or ["small"]
    for w in which:
        run(f"amazon/chronos-bolt-{w}", f"chronos_{w}")
    # re-run the metric table with all methods
    from experiments.e01_forecast_backtest import evaluate
    net, hist = world()
    f = dict(np.load(config.CACHE / "forecasts.npz"))
    origins = f.pop("origins"); f.pop("quantiles")
    evaluate({k: v.astype(float) for k, v in f.items()}, origins, hist)
