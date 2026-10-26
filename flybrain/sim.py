"""Strictly causal trading simulation: decide at bar t using data up to t, earn t -> t+1,
then feel pain/happiness and update synapses before the next decision."""
import numpy as np


def run(brain, X, fwd_log, vol, cost_rate, vol_target, learn=True, explore=0.0, cap=2.0):
    """
    X: (T, N, F) standardized factors; fwd_log: (T, N) log return t->t+1; vol: (T, N) per-bar vol.
    cost_rate: cost per unit of notional traded (e.g. 2e-4 = 2 bps).
    Returns dict of (T, N) arrays: pos, w (notional weight), pnl, happy, pain; and port (T,) returns.
    """
    T, N, _ = X.shape
    pos = np.zeros((T, N)); w = np.zeros((T, N)); pnl = np.zeros((T, N))
    happy = np.full((T, N), np.nan); pain = np.full((T, N), np.nan)
    active = np.zeros((T, N), dtype=bool)
    w_prev = np.zeros(N); p_prev = np.zeros(N)
    for t in range(T):
        ok = np.isfinite(X[t]).all(1) & np.isfinite(fwd_log[t]) & np.isfinite(vol[t]) & (vol[t] > 0)
        active[t] = ok
        p_t = np.zeros(N); w_t = np.zeros(N)
        if ok.any():
            kc = brain.kenyon(X[t, ok])
            p, _ = brain.decide(kc, explore=explore)
            p_t[ok] = p
            scale = np.clip(vol_target / vol[t, ok], 0, cap)
            w_t[ok] = p * scale
        trade = np.abs(w_t - w_prev)
        ret = np.expm1(np.nan_to_num(fwd_log[t]))
        pnl[t] = w_t * ret - cost_rate * trade
        if ok.any() and learn:
            z = fwd_log[t, ok] / vol[t, ok]
            # trading cost felt by the fly, in vol units, for changing its own position
            cz = cost_rate * np.abs(p_t[ok] - p_prev[ok]) / vol[t, ok]
            h, pa = brain.learn(kc, p_t[ok], z, cz)
            happy[t, ok], pain[t, ok] = h, pa
        pos[t], w[t] = p_t, w_t
        w_prev, p_prev = w_t, p_t
    n_act = active.sum(1)
    port = np.where(n_act > 0, pnl.sum(1) / np.maximum(n_act, 1), 0.0)
    return dict(pos=pos, w=w, pnl=pnl, happy=happy, pain=pain, active=active, port=port)


def benchmark(fwd_log, vol, vol_target, active, cap=2.0):
    """Equal-weight, vol-scaled buy-and-hold of the same stocks (no costs)."""
    w = np.where(active, np.clip(vol_target / vol, 0, cap), 0.0)
    pnl = w * np.expm1(np.nan_to_num(fwd_log))
    n = active.sum(1)
    return np.where(n > 0, pnl.sum(1) / np.maximum(n, 1), 0.0)
