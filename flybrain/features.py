"""Quant-research factor library -> the brain's 'sensory' input.

Each factor is a well-documented signal from the empirical finance literature.
Think of each one as an odour receptor: it reports one aspect of the market.
All features at time t use only data available at the close of t (no look-ahead).
"""
import numpy as np
import pandas as pd

# (name, literature source) -- the "quant research pre-training" knowledge base
DAILY_FACTORS = {
    "rev_1d":      "Short-term reversal, 1 day (Lehmann 1990; Jegadeesh 1990)",
    "rev_5d":      "Short-term reversal, 1 week (Lehmann 1990)",
    "mom_1m":      "1-month return",
    "mom_3m":      "Intermediate momentum, 3 months (Jegadeesh & Titman 1993)",
    "mom_12_1":    "12-1 month momentum (Jegadeesh & Titman 1993; Carhart 1997)",
    "xs_mom_rank": "Cross-sectional momentum rank (Asness, Moskowitz & Pedersen 2013)",
    "high_52w":    "Distance to 52-week high (George & Hwang 2004)",
    "vol_20":      "Realised volatility, low-vol anomaly (Ang et al. 2006)",
    "vol_ratio":   "Volatility regime: 5d / 60d vol",
    "park_range":  "Parkinson high-low range volatility (Parkinson 1980)",
    "volu_shock":  "Abnormal volume (Gervais, Kaniel & Mingelgrin 2001)",
    "gap":         "Overnight gap return (Lou, Polk & Skouras 2019)",
    "boll_z":      "Price vs 20d mean in std units (mean reversion)",
    "rsi_14":      "RSI(14) technical oscillator",
    "resid_5d":    "5d beta-adjusted residual return (Blitz et al. 2013)",
    "mkt_5d":      "Market 5d return (time-series regime)",
}

INTRADAY_FACTORS = {
    "rev_1b":     "1-bar reversal (microstructure bounce)",
    "ret_3b":     "15-minute return",
    "ret_12b":    "1-hour return (intraday momentum)",
    "from_open":  "Return since the day's open (Gao et al. 2018 intraday momentum)",
    "vwap_dev":   "Deviation from session VWAP",
    "vol_12b":    "1-hour realised volatility",
    "volu_shock": "Abnormal bar volume vs time-of-day norm",
    "range_b":    "Bar high-low range",
    "resid_3b":   "15-minute residual vs market",
    "mkt_3b":     "Market 15-minute return",
    "tod":        "Time of day (U-shaped intraday patterns)",
    "prev_day":   "Previous day's close-to-close return",
}


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def daily_features(p, mkt_close):
    """p: dict of DataFrames (dates x tickers). Returns dict name -> DataFrame, plus fwd return & vol."""
    c, o, h, l, v = p["close"], p["open"], p["high"], p["low"], p["volume"]
    r = np.log(c).diff()
    mr = np.log(mkt_close).diff()
    vol20 = r.rolling(20).std()
    f = {}
    f["rev_1d"] = r / vol20
    f["rev_5d"] = r.rolling(5).sum() / (vol20 * np.sqrt(5))
    f["mom_1m"] = r.rolling(21).sum() / (vol20 * np.sqrt(21))
    f["mom_3m"] = r.rolling(63).sum() / (vol20 * np.sqrt(63))
    f["mom_12_1"] = r.shift(21).rolling(231).sum()
    f["xs_mom_rank"] = f["mom_12_1"].rank(axis=1, pct=True) - 0.5
    f["high_52w"] = np.log(c / c.rolling(252).max())
    f["vol_20"] = np.log(vol20)
    f["vol_ratio"] = np.log(r.rolling(5).std() / r.rolling(60).std())
    f["park_range"] = np.log(np.log(h / l).rolling(5).mean() + 1e-6)
    f["volu_shock"] = np.log(v / v.rolling(20).mean())
    f["gap"] = np.log(o / c.shift(1)) / vol20
    f["boll_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["rsi_14"] = _rsi(c)
    beta = pd.DataFrame({k: r[k].rolling(60).cov(mr) / mr.rolling(60).var() for k in r.columns})
    resid = r.sub(beta.mul(mr, axis=0))
    f["resid_5d"] = resid.rolling(5).sum() / (vol20 * np.sqrt(5))
    f["mkt_5d"] = pd.DataFrame(np.repeat(mr.rolling(5).sum().values[:, None], r.shape[1], 1),
                               index=r.index, columns=r.columns)
    fwd = r.shift(-1)  # return earned holding from close t to close t+1
    return f, fwd, vol20


def intraday_features(p, mkt):
    """p: dict of DataFrames (5-min bars x tickers), NY time. mkt: dict for SPY."""
    c, o, h, l, v = p["close"], p["open"], p["high"], p["low"], p["volume"]
    idx = c.index
    day = pd.Series(idx.date, index=idx)
    newday = day != day.shift(1)
    r = np.log(c).diff()
    r[newday.values] = np.nan            # don't learn from overnight jumps intrabar
    mr = np.log(mkt["close"]).diff()
    mr[newday.values] = np.nan
    vol = r.rolling(78, min_periods=30).std()
    f = {}
    f["rev_1b"] = r / vol
    f["ret_3b"] = r.rolling(3).sum() / (vol * np.sqrt(3))
    f["ret_12b"] = r.rolling(12).sum() / (vol * np.sqrt(12))
    day_open = o.groupby(day.values).transform("first")
    f["from_open"] = np.log(c / day_open) / (vol * np.sqrt(78))
    pv = (c * v).groupby(day.values).cumsum()
    cv = v.groupby(day.values).cumsum()
    f["vwap_dev"] = np.log(c / (pv / cv.replace(0, np.nan))) / vol
    f["vol_12b"] = np.log(r.rolling(12, min_periods=6).std() / vol)
    tod = pd.Series(idx.hour * 60 + idx.minute, index=idx)
    vnorm = v.groupby(tod.values).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    f["volu_shock"] = np.log((v + 1) / (vnorm + 1))
    f["range_b"] = np.log(np.log(h / l) / vol + 1e-3)
    beta = 1.0
    f["resid_3b"] = (r - beta * mr.values[:, None]).rolling(3).sum() / (vol * np.sqrt(3))
    m3 = mr.rolling(3).sum()
    f["mkt_3b"] = pd.DataFrame(np.repeat(m3.values[:, None], c.shape[1], 1), index=idx, columns=c.columns)
    t = (tod.values - 570) / 390.0
    f["tod"] = pd.DataFrame(np.repeat(t[:, None], c.shape[1], 1), index=idx, columns=c.columns)
    dclose = c.groupby(day.values).transform("last")
    # previous session's close-to-close return: shift(1) so day D only sees D-1 (no look-ahead)
    prev = np.log(dclose).groupby(day.values).first().diff().shift(1)
    f["prev_day"] = pd.DataFrame({k: day.map(prev[k]).values for k in c.columns}, index=idx) / (vol * np.sqrt(78))
    fwd = r.shift(-1)
    lastbar = newday.shift(-1, fill_value=True).values
    fwd[lastbar] = np.nan                 # flat at the close: no overnight holds
    return f, fwd, vol


def standardize(f, train_mask):
    """Robust z-score each factor using TRAINING-period statistics only, clipped to +-3."""
    out = {}
    for k, df in f.items():
        x = df[train_mask].values.ravel()
        x = x[np.isfinite(x)]
        med, iqr = np.median(x), np.subtract(*np.percentile(x, [75, 25]))
        s = iqr / 1.349 if iqr > 0 else (np.std(x) + 1e-9)
        out[k] = ((df - med) / s).clip(-3, 3)
    return out


def to_tensor(fz, names):
    """-> array (T, N, F)"""
    return np.stack([fz[k].values for k in names], axis=-1)
