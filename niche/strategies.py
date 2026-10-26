"""Two long-only strategies for the pain/happiness fly brain in illiquid markets.

Shorting is impractical in these markets (banned for retail in Vietnam, scarce borrow in
Japanese/UK/Nordic small caps), so both strategies are long-only and judged on returns in
excess of an equal-weight portfolio of all tradeable stocks in the same market.

A  "Slow picker":   every week the fly scores every tradeable stock and holds its top K.
                    A holding is only sold once it drops out of the top 2K, so typical holds
                    last weeks to months. Pain/happiness = the holding's weekly return relative
                    to the market, net of the trading costs it caused.
B  "Shock rebound": when a stock crashes on its own (idiosyncratic move < -2.5 sigma), the fly
                    decides whether to buy at the next open and hold 10 days. Pain/happiness =
                    the 10-day market-relative return net of round-trip costs; skipped shocks
                    teach it through weaker 'what if' (fictive) dopamine.

Costs per side: half the quoted spread + price impact of a $10k order (Amihud, trailing 60d,
capped at 5%) + 0.10% commission; Vietnam also charges 0.1% tax on sales.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from flybrain.brain import MushroomBodyTrader
from . import data as D

POSITION_USD = 10_000
MIN_ADV_USD = 100_000

FEATURES_A = ["mom_12_1", "rev_1m", "rev_1w", "high_52w", "vol_60", "max_21", "illiq",
              "volu_trend", "ma200", "resid_3m", "zero_days", "mkt_1m"]
FEATURES_B = ["shock", "gap", "volu_day", "rev_1m", "mom_12_1", "high_52w", "vol_60",
              "illiq", "mkt_day", "n_shocks", "resid_3m", "max_21"]


# ----------------------------------------------------------------------------- prep
def prepare(key):
    p = D.prices(key)
    u = {x["symbol"]: x for x in D.universe(key)}
    sp = D.spreads(key)["spread"]
    C = p["Close"].copy()
    C = C.where(C > 0)
    O, H, L, V = [p[k].reindex_like(C) for k in ("Open", "High", "Low", "Volume")]
    O = O.where(O > 0)
    days = C.notna().sum(1) > 0.2 * C.shape[1]          # keep real trading days only
    C, O, H, L, V = [x[days] for x in (C, O, H, L, V)]
    lr = np.log(C).diff()
    lr = lr.where(lr.abs() < 1.0)                        # drop obvious data errors (> +172% / < -63%)
    fx = pd.Series({s: u[s]["fx_usd"] for s in C.columns})
    dv = (C * V).mul(fx, axis=1)
    adv = dv.rolling(60, min_periods=40).median()
    zero = (lr == 0).astype(float).where(lr.notna()).rolling(60, min_periods=40).mean()
    nvalid = lr.notna().rolling(60).sum()
    eligible = (adv >= MIN_ADV_USD) & (zero < 0.3) & (nvalid >= 50)
    vol = lr.rolling(60, min_periods=40).std()
    amihud = (lr.abs() / dv.replace(0, np.nan)).rolling(60, min_periods=30).mean()
    impact = (amihud * POSITION_USD).clip(upper=0.05)
    half_spread = pd.Series({s: sp.get(s, D.spreads(key)["median"]) / 2 for s in C.columns})
    cost_side = impact.add(half_spread, axis=1) + D.COMMISSION        # per side, as fraction
    sell_tax = D.SELL_TAX.get(key, 0.0)
    # equal-weight market of tradeable stocks (daily log return)
    mkt = lr.where(eligible.shift(1, fill_value=False)).mean(1)          # used as a feature only
    # benchmark: daily-rebalanced equal-weight portfolio of tradeable stocks (simple returns),
    # stored as log of its gross return so it can be summed over any window
    mkt_bench = np.log1p(np.expm1(lr).where(eligible.shift(1, fill_value=False)).mean(1))

    f = {}
    f["mom_12_1"] = lr.shift(21).rolling(231, min_periods=180).sum()
    f["rev_1m"] = lr.rolling(21, min_periods=15).sum()
    f["rev_1w"] = lr.rolling(5, min_periods=4).sum()
    f["high_52w"] = np.log(C / C.rolling(252, min_periods=180).max())
    f["vol_60"] = np.log(vol)
    f["max_21"] = lr.rolling(21, min_periods=15).max()
    f["illiq"] = np.log(amihud)
    f["volu_trend"] = np.log(dv.rolling(20, min_periods=10).mean() / dv.rolling(120, min_periods=60).mean())
    f["ma200"] = np.log(C / C.rolling(200, min_periods=150).mean())
    f["resid_3m"] = lr.rolling(63, min_periods=45).sum().sub(mkt.rolling(63, min_periods=45).sum(), axis=0)
    f["zero_days"] = zero
    f["mkt_1m"] = pd.DataFrame(np.repeat(mkt.rolling(21).sum().values[:, None], C.shape[1], 1),
                               index=C.index, columns=C.columns)
    # shock-specific
    vol_prev = vol.shift(1)
    idio = lr.sub(mkt, axis=0)
    f["shock"] = idio / vol_prev
    f["gap"] = np.log(O / C.shift(1)) / vol_prev
    f["volu_day"] = np.log(dv / dv.rolling(60, min_periods=40).median().shift(1))
    f["mkt_day"] = pd.DataFrame(np.repeat((mkt / mkt.rolling(60).std()).values[:, None], C.shape[1], 1),
                                index=C.index, columns=C.columns)
    f["n_shocks"] = (f["shock"] < -2.5).astype(float).rolling(60, min_periods=40).sum().shift(1)
    return dict(key=key, C=C, O=O, lr=lr, mkt=mkt, mkt_bench=mkt_bench, eligible=eligible, vol=vol, cost=cost_side,
                sell_tax=sell_tax, f=f, dates=C.index, tickers=list(C.columns))


def rank_gauss(x):
    """Cross-sectional normal scores (robust to the fat tails of small caps)."""
    r = pd.Series(x).rank().values
    n = np.isfinite(x).sum()
    out = norm.ppf((r - 0.5) / max(n, 1))
    out[~np.isfinite(x)] = 0.0
    return np.clip(out, -2.5, 2.5)


def snapshot(P, names, t, idx, ts_names=("mkt_1m", "mkt_day"), ts_scale=None):
    cols = []
    for k in names:
        x = P["f"][k].values[t, idx]
        if k in ts_names:
            mu, sd = ts_scale[k]
            cols.append(np.clip(np.nan_to_num((x - mu) / sd), -2.5, 2.5))
        else:
            cols.append(rank_gauss(x))
    return np.stack(cols, 1)


def ts_scale(P, end):
    m = P["dates"] <= pd.Timestamp(end)
    out = {}
    for k in ("mkt_1m", "mkt_day"):
        s = P["f"][k].iloc[:, 0][m].dropna()
        out[k] = (float(s.mean()), float(s.std() + 1e-9))
    return out


# ----------------------------------------------------------------------------- strategy A
def run_picker(P, brain, start, end, scale, K=20, step=5, keep_mult=2, explore=0.0, learn=True, rng=None, record=False):
    """Returns weekly records: net portfolio return, market return, trades, holdings history."""
    rng = rng or np.random.default_rng(0)
    dates = P["dates"]; lr = P["lr"].values; elig = P["eligible"].values; cost = P["cost"].values
    vol = P["vol"].values
    t_idx = np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0]
    t_idx = t_idx[::step]
    held = set(); pending = None; out = []
    cum = np.nan_to_num(lr).cumsum(0)                    # for fast multi-day sums
    for n, t in enumerate(t_idx[:-1]):
        t2 = t_idx[n + 1]
        # 1) feel pain/happiness about the week that just ended, update synapses
        if pending is not None and learn:
            kc, idx, pos, outcome, c_paid, v = pending
            ok = np.isfinite(outcome) & np.isfinite(v) & (v > 0)
            if ok.any():
                brain.learn(kc[ok], pos[ok], outcome[ok] / v[ok], c_paid[ok] / v[ok])
        # 2) score tradeable stocks
        idx = np.where(elig[t])[0]
        if len(idx) < 15:
            pending = None; out.append(None); continue
        k_t = min(K, len(idx) // 3)
        X = snapshot(P, FEATURES_A, t, idx, ts_scale=scale)
        kc = brain.kenyon(X)
        _, score = brain.decide(kc)
        if explore:
            score = score + rng.normal(0, explore, len(score))
        order = idx[np.argsort(-score)]
        rank = {j: r for r, j in enumerate(order)}
        keep = {j for j in held if j in rank and rank[j] < keep_mult * k_t}
        sells = held - keep
        buys = []
        for j in order:
            if len(keep) + len(buys) >= k_t:
                break
            if j not in keep:
                buys.append(j)
        new = keep | set(buys)
        # 3) returns over the coming week (log -> simple), costs
        wk = cum[t2] - cum[t]
        valid = np.isfinite(lr[t + 1:t2 + 1]).any(0)
        r_simple = np.where(valid, np.expm1(wk), 0.0)
        mkt_set = idx
        mkt_ret = float(np.mean(r_simple[mkt_set]))
        hold_arr = np.array(sorted(new))
        gross = float(np.mean(r_simple[hold_arr])) if len(hold_arr) else 0.0
        w = 1.0 / max(k_t, 1)
        c_buy = sum(cost[t, j] for j in buys) * w
        c_sell = sum(np.nan_to_num(cost[t, j], nan=0.05) + P["sell_tax"] for j in sells) * w
        net = gross * len(hold_arr) / max(k_t, 1) - c_buy - c_sell
        rec = dict(date=dates[t], net=net, gross=gross, mkt=mkt_ret, n_hold=len(hold_arr), k=k_t,
                   n_buy=len(buys), n_sell=len(sells), n_elig=len(idx))
        if record:
            rec.update(held=list(map(int, hold_arr)), sells=list(map(int, sells)), buys=list(map(int, buys)),
                       elig=list(map(int, idx)))
        out.append(rec)
        # 4) store the experience for learning next step (market-relative, vol-normalised)
        pos = np.isin(idx, hold_arr).astype(float)
        outcome = np.log1p(r_simple[idx]) - np.log1p(mkt_ret)
        paid = np.array([cost[t, j] if j in buys else 0.0 for j in idx])
        v = vol[t, idx] * np.sqrt(step)
        pending = (kc, idx, pos, outcome, np.nan_to_num(paid), v)
        held = new
    return [r for r in out if r is not None], dict(r_cum=cum)


# ----------------------------------------------------------------------------- strategy B
def run_rebound(P, brain, start, end, scale, theta=0.5, hold=10, max_pos=20, sigma=2.5,
                explore=0.0, learn=True, rng=None):
    """Returns trade list (one row per shock event, bought or not)."""
    rng = rng or np.random.default_rng(0)
    dates = P["dates"]; lr = P["lr"].values; elig = P["eligible"].values
    C = P["C"].values; O = P["O"].values; cost = P["cost"].values; vol = P["vol"].values
    shock = P["f"]["shock"].values
    mkt_cum = np.nan_to_num(P["mkt_bench"].values).cumsum()
    T = len(dates)
    t_idx = np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0]
    pending = []          # (exit_t, kc_row, bought, outcome, cost, v)
    open_until = []       # exit times of open positions
    events = []
    for t in t_idx:
        # learn from events whose 10-day window has finished
        if learn and pending:
            done = [e for e in pending if e[0] <= t]
            pending = [e for e in pending if e[0] > t]
            if done:
                kc = np.stack([e[1] for e in done]); pos = np.array([e[2] for e in done])
                oc = np.array([e[3] for e in done]); cz = np.array([e[4] for e in done]); v = np.array([e[5] for e in done])
                brain.learn(kc, pos, oc / v, cz / v)
        open_until = [x for x in open_until if x > t]
        if t + hold + 1 >= T:
            continue
        ev = np.where(elig[t - 1] & (shock[t] < -sigma) & np.isfinite(O[t + 1]))[0] if t > 0 else []
        if len(ev) == 0:
            continue
        X = snapshot(P, FEATURES_B, t, ev, ts_scale=scale)
        # shock-specific features are better compared in absolute terms than ranked within a day
        X[:, 0] = np.clip((shock[t, ev] + 4) / 1.5, -2.5, 2.5)
        kc = brain.kenyon(X)
        _, score = brain.decide(kc)
        if explore:
            score = score + rng.normal(0, explore, len(score))
        want = score > theta
        slots = max_pos - len(open_until)
        buy = np.zeros(len(ev), bool)
        if slots > 0 and want.any():
            cand = np.where(want)[0][np.argsort(-score[want])][:slots]
            buy[cand] = True
        t_exit = t + hold
        for i, j in enumerate(ev):
            entry, exit_ = O[t + 1, j], C[t_exit, j]
            if not np.isfinite(exit_):
                last = C[t + 1:t_exit + 1, j]; last = last[np.isfinite(last)]
                exit_ = last[-1] if len(last) else entry
            r = np.log(exit_ / entry)
            m = mkt_cum[t_exit] - mkt_cum[t]
            c_rt = 2 * np.nan_to_num(cost[t, j], nan=0.05) + P["sell_tax"]
            v = vol[t, j] * np.sqrt(hold)
            events.append(dict(t=int(t), date=dates[t], j=int(j), bought=bool(buy[i]), score=float(score[i]),
                               ret=float(np.expm1(r)), mkt=float(np.expm1(m)), cost=float(c_rt),
                               excess_net=float(np.expm1(r) - np.expm1(m) - c_rt)))
            pending.append((t_exit, kc[i], float(buy[i]), float(r - m), float(c_rt * buy[i]), float(v)))
            if buy[i]:
                open_until.append(t_exit)
    return events


def new_brain(n_features, seed, lr):
    return MushroomBodyTrader(n_features, seed=seed, lr=lr, threshold=0.5)
