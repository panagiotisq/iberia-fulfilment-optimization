"""Thin helpers around highspy for matrix-form models."""
from __future__ import annotations

import highspy
import numpy as np
import scipy.sparse as sp

INF = highspy.kHighsInf


def _clip_inf(a):
    a = np.asarray(a, dtype=np.float64).copy()
    a[a >= 1e30] = INF
    a[a <= -1e30] = -INF
    return a


def make_highs(c, A: sp.csc_matrix, row_lo, row_hi, col_lo, col_hi, integrality=None, **options) -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    for k, v in options.items():
        h.setOptionValue(k, v)
    A = sp.csc_matrix(A)
    A.sort_indices()
    lp = highspy.HighsLp()
    lp.num_col_ = A.shape[1]
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = np.asarray(c, dtype=np.float64)
    lp.col_lower_ = _clip_inf(col_lo)
    lp.col_upper_ = _clip_inf(col_hi)
    lp.row_lower_ = _clip_inf(row_lo)
    lp.row_upper_ = _clip_inf(row_hi)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    if integrality is not None and np.any(integrality):
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in integrality]
    h.passModel(lp)
    return h


def set_row_bounds(h: highspy.Highs, lo, hi, idx=None):
    if idx is None:
        idx = np.arange(len(lo), dtype=np.int32)
    h.changeRowsBounds(len(idx), np.asarray(idx, dtype=np.int32), _clip_inf(lo), _clip_inf(hi))


def status_ok(h: highspy.Highs) -> bool:
    return h.getModelStatus() == highspy.HighsModelStatus.kOptimal


def objective(h: highspy.Highs) -> float:
    return h.getInfo().objective_function_value
