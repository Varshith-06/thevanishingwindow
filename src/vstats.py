"""Variance decomposition statistics from the plan's Measurement details section."""
import numpy as np


def read_time_share(hits, totals):
    """Noise-corrected V from per-task hack counts. Returns (V, ybar, raw_var, noise)."""
    hits, totals = np.asarray(hits, float), np.asarray(totals, float)
    keep = totals >= 2
    hits, totals = hits[keep], totals[keep]
    p = hits / totals
    ybar = hits.sum() / totals.sum()
    if not 0 < ybar < 1 or len(p) < 2:
        return float("nan"), float(ybar), float("nan"), float("nan")
    raw = p.var(ddof=1)
    noise = np.mean(p * (1 - p) / (totals - 1))
    return float((raw - noise) / (ybar * (1 - ybar))), float(ybar), float(raw), float(noise)


def bootstrap_v(hits, totals, n=2000, seed=0):
    """Percentile 95% interval for V, resampling tasks."""
    rng = np.random.default_rng(seed)
    hits, totals = np.asarray(hits), np.asarray(totals)
    idx = rng.integers(0, len(hits), (n, len(hits)))
    vs = np.array([read_time_share(hits[i], totals[i])[0] for i in idx])
    vs = vs[~np.isnan(vs)]
    if len(vs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))
