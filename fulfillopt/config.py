"""Experiment-wide constants (dates, seeds, paths)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGURES = ROOT / "report" / "figures"

DGP_SEED = 2026
HISTORY_START = "2023-01-02"
HISTORY_END = "2026-12-31"

# Forecasting protocol: models are retrained quarterly (expanding window). Origins before
# TEST_START are used for model selection (validation) and to warm up the rolling conformal
# and copula calibration; the test year is never used for any design decision.
FORECAST_START = "2025-06-01"        # first forecast origin (validation)
TEST_START = "2025-09-30"            # first test origin -> first target 2025-10-01
TEST_END = "2026-09-29"              # last test origin  -> last h=1 target 2026-09-30
RETRAIN_DATES = ["2025-06-01", "2025-09-01", "2025-12-01", "2026-03-01", "2026-06-01", "2026-09-01"]


def ensure_dirs():
    for p in (RESULTS, CACHE, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
