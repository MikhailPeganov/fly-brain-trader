"""Shared setup: universe split, periods, feature tensors."""
import numpy as np
import pandas as pd
from . import data, features

SPLIT_SEED = 2026
TRAIN = ("2014-01-01", "2021-12-31")
VAL = ("2022-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-10-23")
DAILY_COST = 2e-4      # 2 bps per unit notional traded
INTRADAY_COST = 1e-4   # 1 bp
DAILY_VOL_TARGET = 0.01


def split_universe():
    rng = np.random.default_rng(SPLIT_SEED)
    u = list(rng.permutation(data.UNIVERSE))
    return sorted(u[:40]), sorted(u[40:])


def daily_tensors(transform=None):
    """transform: optional fn(panel)->panel (used for the placebo)."""
    syms = data.UNIVERSE
    p = data.panel(syms + [data.MARKET])
    if transform is not None:
        p = transform(p)
    mkt = p["close"][data.MARKET]
    p = {k: v[syms] for k, v in p.items()}
    f, fwd, vol = features.daily_features(p, mkt)
    train_tk, _ = split_universe()
    idx = fwd.index
    tmask = (idx >= TRAIN[0]) & (idx <= TRAIN[1])
    # robust z-score using stats from the training period AND training stocks only,
    # then apply the same location/scale to every stock (hidden stocks included)
    fz_all = {}
    for k, v in f.items():
        x = v[train_tk][tmask].values.ravel(); x = x[np.isfinite(x)]
        med = np.median(x); iqr = np.subtract(*np.percentile(x, [75, 25]))
        s = iqr / 1.349 if iqr > 0 else np.std(x) + 1e-9
        fz_all[k] = ((v - med) / s).clip(-3, 3)
    names = list(features.DAILY_FACTORS)
    X = features.to_tensor(fz_all, names)
    return dict(X=X, fwd=fwd.values, vol=vol.values, dates=idx, tickers=list(fwd.columns),
                names=names)


def select(d, period, tickers):
    idx = d["dates"]
    tm = (idx >= period[0]) & (idx <= period[1])
    cols = [d["tickers"].index(t) for t in tickers]
    return dict(X=d["X"][tm][:, cols], fwd=d["fwd"][tm][:, cols], vol=d["vol"][tm][:, cols],
                dates=idx[tm], tickers=tickers)
