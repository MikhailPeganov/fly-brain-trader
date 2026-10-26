"""Skill-vs-luck hypothesis tests.

H0 (primary): the fly's positions carry no information about future returns; any profit
is luck. Tested by a circular-shift permutation test: each stock's position series is
rotated by a random offset, which keeps the fly's long/short mix, position sizes,
turnover and autocorrelation exactly, but breaks the timing link to returns.
"""
import numpy as np
from scipy import stats as st


def sharpe(x, periods):
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return float(np.sqrt(periods) * x.mean() / s) if s > 0 else 0.0


def newey_west_t(x, lags=None):
    """t-stat of the mean with Newey-West HAC standard error."""
    x = np.asarray(x, float); n = len(x)
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(s / n)
    t = x.mean() / se if se > 0 else 0.0
    return float(t), float(1 - st.norm.cdf(t)), lags


def stationary_bootstrap_p(x, n_boot=5000, mean_block=10, seed=0):
    """One-sided p-value for H0: mean <= 0 (Politis & Romano 1994 stationary bootstrap)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); n = len(x); xc = x - x.mean()
    obs = x.mean(); cnt = 0
    p = 1.0 / mean_block
    for _ in range(n_boot):
        idx = np.empty(n, int); i = rng.integers(n)
        for k in range(n):
            if k > 0 and rng.random() < p:
                i = rng.integers(n)
            idx[k] = i; i = (i + 1) % n
        if xc[idx].mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n_boot + 1)


def _portfolio(w, ret, active, cost_rate):
    trade = np.abs(np.diff(np.vstack([np.zeros((1, w.shape[1])), w]), axis=0))
    pnl = w * ret - cost_rate * trade
    n = np.maximum(active.sum(1), 1)
    return pnl.sum(1) / n


def circular_shift_test(w, fwd_log, active, cost_rate, n_perm=2000, min_shift=5, seed=0,
                        periods=252):
    """Permutation test on the strategy's Sharpe ratio (net of costs).
    Shifts each stock's notional-weight series independently within its active rows."""
    rng = np.random.default_rng(seed)
    ret = np.expm1(np.nan_to_num(fwd_log))
    obs = sharpe(_portfolio(w, ret, active, cost_rate), periods)
    T, N = w.shape
    null = np.empty(n_perm)
    rows = [np.where(active[:, j])[0] for j in range(N)]
    for b in range(n_perm):
        ws = np.zeros_like(w)
        for j in range(N):
            r = rows[j]
            if len(r) <= 2 * min_shift:
                continue
            k = rng.integers(min_shift, len(r) - min_shift)
            ws[r, j] = np.roll(w[r, j], k)
        null[b] = sharpe(_portfolio(ws, ret, active, cost_rate), periods)
    p = (1 + (null >= obs).sum()) / (n_perm + 1)
    return dict(observed=obs, null=null, p=float(p),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)))


def alpha_regression(strat, bench, periods=252):
    """strat = a + b*bench + e, HAC t-stat on alpha."""
    X = np.column_stack([np.ones_like(bench), bench])
    beta, *_ = np.linalg.lstsq(X, strat, rcond=None)
    resid = strat - X @ beta
    t_res, p_res, _ = newey_west_t(resid + beta[0])  # alpha = mean of (resid + a)
    return dict(alpha_ann=float(beta[0] * periods), beta=float(beta[1]), t_alpha=t_res, p_alpha=p_res)


def hit_rate_test(pos, fwd_log, active):
    m = active & (pos != 0) & np.isfinite(fwd_log) & (fwd_log != 0)
    s, r = np.sign(pos[m]), np.sign(fwd_log[m])
    hits = int((s == r).sum()); n = int(m.sum())
    if n == 0:
        return dict(n=0, hit=np.nan, p0=np.nan, p=np.nan)
    # chance hit rate given how often it goes long and how often stocks go up
    pl, pu = (s > 0).mean(), (r > 0).mean()
    p0 = pl * pu + (1 - pl) * (1 - pu)
    p = st.binomtest(hits, n, p0, alternative="greater").pvalue
    return dict(n=n, hit=hits / n, p0=float(p0), p=float(p))
