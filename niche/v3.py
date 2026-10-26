"""Fly-brain v3: news senses, simple-model senses, a market-direction brain, and 'love'.

Two mushroom bodies:
  * STOCK BRAIN picks stocks. Senses = 10 quant factors + 6 simple-model forecasts (the 'maths
    basics': trend, time-series momentum, Bollinger mean reversion, a rolling pooled OLS forecast,
    EWMA volatility, market beta) + 4 news senses (article volume, tone, bullish and bearish keyword
    share; only where per-company news exists, i.e. MOEX large caps).
  * MARKET BRAIN decides direction each week: LONG (hold the stock brain's top picks) or SHORT
    (MOEX: short the stock brain's bottom picks; elsewhere: short an equal-weight basket of the market,
    standing in for an index future). Senses = market news (volume, tone, bull/bear keyword share) +
    simple models on the market index + S&P 500 trend.

Affects:
  * happiness / pain: as before (loss-averse, 2x), for each brain's own job.
  * LOVE: every week the fly compares its trailing 52-week USD return with the S&P 500's. When it is
    ahead, it feels love in proportion to the margin (capped at 10 points), which reinforces whatever it
    is currently holding and its current market direction. Falling behind triggers nothing extra.

All accounting is in USD so that 'beating the S&P 500' is a like-for-like comparison.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from flybrain.brain import MushroomBodyTrader

FACTORS = ["mom_12_1", "rev_1m", "rev_1w", "high_52w", "vol_60", "max_21", "illiq", "volu_trend", "resid_3m", "zero_days"]
MODELS = ["trend_ma", "tsmom", "boll_z", "ols_fc", "ewma_vol", "beta"]
NEWS = ["news_vol", "news_tone", "news_bull", "news_bear"]
CONTEXT = ["ctx_trend", "ctx_revert"]      # which simple model has been paying lately (same for every stock)
FEATS_STOCK = FACTORS + MODELS + NEWS + CONTEXT
FEATS_MKT = ["m_news_vol", "m_news_tone", "m_news_bull", "m_news_bear", "m_trend", "m_tsmom", "m_rev1m", "m_vol",
             "spx_1m", "spx_trend"]
POSITION_USD, MIN_ADV_USD = 10_000, 100_000
COMMISSION, BASKET_COST, BORROW = 0.0010, 0.0010, 0.03


class LovingBrain(MushroomBodyTrader):
    """Mushroom body plus a direct sensory pathway (like the fly's lateral horn): output neurons also see the
    projection neurons directly, with the same dopamine-gated plasticity. The sparse Kenyon-cell code is good
    at memorising specific situations; the direct pathway is good at simple straight-line rules (the basics)."""
    def __init__(self, n_factors, direct=1.0, **kw):
        super().__init__(n_factors, **kw)
        self.direct = direct
        if direct:
            extra = np.full(self.n_pn, self.w0)
            self.w_long = np.concatenate([self.w_long, extra]); self.w_short = np.concatenate([self.w_short, extra.copy()])

    def kenyon(self, x):
        kc = super().kenyon(x)
        if not self.direct:
            return kc
        pn = np.exp(-0.5 * ((x[:, :, None] - self.centers) / self.sigma) ** 2).reshape(len(x), -1)
        return np.hstack([kc, self.direct * pn.astype(np.float32)])

    def love(self, kc, pos, amount):
        """Tonic reward for being ahead of the S&P 500: strengthens the synapses behind current positions."""
        if amount <= 0 or len(kc) == 0:
            return
        n = max(1, len(kc))
        self.w_long += self.lr * ((amount * (pos > 0)) @ kc) / n
        self.w_short += self.lr * ((amount * (pos < 0)) @ kc) / n
        np.clip(self.w_long, 0.0, self.wmax, out=self.w_long)
        np.clip(self.w_short, 0.0, self.wmax, out=self.w_short)


def rank_gauss(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out = np.zeros_like(x)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0:
        return out
    r = pd.Series(x[ok]).rank().values
    out[ok] = norm.ppf((r - 0.5) / ok.sum())
    return np.clip(out, -2.5, 2.5)


# ------------------------------------------------------------------------------- news senses
def news_block(daily, dates):
    """daily: DataFrame indexed by UTC calendar day with columns all, bull, bear, norm, tone.
    Returns 4 series aligned to trading dates using only news up to the previous calendar day."""
    d = daily.asfreq("D")
    ratio = lambda c: (d[c].fillna(0) / d["norm"].replace(0, np.nan)).fillna(0) * 1e6
    a5 = ratio("all").rolling(5, min_periods=1).sum()
    vol = np.log((a5 + 1) / (a5.rolling(60, min_periods=20).mean() + 1))
    tone = d["tone"].rolling(5, min_periods=1).mean()
    tot = d["all"].fillna(0).rolling(5, min_periods=1).sum() + 1
    bull = d["bull"].fillna(0).rolling(5, min_periods=1).sum() / tot
    bear = d["bear"].fillna(0).rolling(5, min_periods=1).sum() / tot
    out = pd.DataFrame({"vol": vol, "tone": tone, "bull": bull, "bear": bear}).shift(1)   # 1-day lag: no look-ahead
    return out.reindex(pd.DatetimeIndex(dates).normalize(), method="ffill")


# ------------------------------------------------------------------------------- panel prep
def prepare_panel(p, fx_const, fx_daily, half_spread, sell_tax, spx, mkt_news=None, stock_news=None, borrow=BORROW):
    """p: dict Open/High/Low/Close/Volume (dates x tickers, local currency, total-return adjusted).
    fx_const: Series ticker->USD per local unit (for liquidity filter); fx_daily: DataFrame dates x tickers
    of USD per local unit; half_spread: Series; spx: Series of S&P 500 total-return index (USD)."""
    C = p["Close"].where(p["Close"] > 0)
    O, H, L, V = [p[k].reindex_like(C) for k in ("Open", "High", "Low", "Volume")]
    days = C.notna().sum(1) > 0.2 * C.shape[1]
    C, O, H, L, V = [x[days] for x in (C, O, H, L, V)]
    O = O.where(O > 0).fillna(C)
    dates = C.index
    lr = np.log(C).diff()
    lr = lr.where(lr.abs() < 1.0)
    dv = (C * V).mul(fx_const, axis=1)
    adv = dv.rolling(60, min_periods=40).median()
    zero = (lr == 0).astype(float).where(lr.notna()).rolling(60, min_periods=40).mean()
    eligible = (adv >= MIN_ADV_USD) & (zero < 0.3) & (lr.notna().rolling(60).sum() >= 50)
    vol = lr.rolling(60, min_periods=40).std()
    amihud = (lr.abs() / dv.replace(0, np.nan)).rolling(60, min_periods=30).mean()
    cost = (amihud * POSITION_USD).clip(upper=0.05).add(half_spread, axis=1) + COMMISSION
    elig_prev = eligible.shift(1, fill_value=False)
    mkt = lr.where(elig_prev).mean(1)
    mkt_bench = np.log1p(np.expm1(lr).where(elig_prev).mean(1))
    fxl = np.log(fx_daily.reindex(dates).ffill()).diff().fillna(0.0)

    f = {}
    f["mom_12_1"] = lr.shift(21).rolling(231, min_periods=180).sum()
    f["rev_1m"] = lr.rolling(21, min_periods=15).sum()
    f["rev_1w"] = lr.rolling(5, min_periods=4).sum()
    f["high_52w"] = np.log(C / C.rolling(252, min_periods=180).max())
    f["vol_60"] = np.log(vol)
    f["max_21"] = lr.rolling(21, min_periods=15).max()
    f["illiq"] = np.log(amihud)
    f["volu_trend"] = np.log(dv.rolling(20, min_periods=10).mean() / dv.rolling(120, min_periods=60).mean())
    f["resid_3m"] = lr.rolling(63, min_periods=45).sum().sub(mkt.rolling(63, min_periods=45).sum(), axis=0)
    f["zero_days"] = zero
    # --- simple models ("maths basics")
    f["trend_ma"] = np.log(C.rolling(50, min_periods=40).mean() / C.rolling(200, min_periods=150).mean())
    f["tsmom"] = lr.rolling(252, min_periods=180).sum() / (vol * np.sqrt(252))
    f["boll_z"] = (C - C.rolling(20, min_periods=15).mean()) / C.rolling(20, min_periods=15).std()
    f["ewma_vol"] = np.log(np.sqrt((lr ** 2).ewm(alpha=0.06, min_periods=30).mean()))
    mvar = mkt.rolling(120, min_periods=80).var()
    f["beta"] = pd.DataFrame({k: lr[k].rolling(120, min_periods=80).cov(mkt) / mvar for k in lr.columns})
    f["ols_fc"] = ols_forecast(lr, mkt, f, dates)
    # regime context: rolling 3-month average of each basic model's realised weekly rank correlation
    f["ctx_trend"], f["ctx_revert"] = model_payoff(lr, f["trend_ma"], dates), model_payoff(lr, -f["boll_z"], dates)
    # --- stock news (zeros where none)
    for k in NEWS:
        f[k] = pd.DataFrame(0.0, index=dates, columns=C.columns)
    if stock_news:
        for tk, daily in stock_news.items():
            if tk in C.columns and daily is not None and len(daily):
                nb = news_block(daily, dates)
                for k, c in zip(NEWS, ["vol", "tone", "bull", "bear"]):
                    f[k][tk] = nb[c].values
    # --- market senses
    mi = mkt_bench.fillna(0).cumsum()
    mv = mkt.rolling(60, min_periods=40).std()
    m = pd.DataFrame(index=dates)
    m["m_trend"] = mi.rolling(50).mean() - mi.rolling(200).mean()
    m["m_tsmom"] = mkt.rolling(252, min_periods=180).sum() / (mv * np.sqrt(252))
    m["m_rev1m"] = mkt.rolling(21).sum() / (mv * np.sqrt(21))
    m["m_vol"] = np.log(mv)
    spx_lvl = np.log(spx).reindex(dates, method="ffill") if spx is not None else pd.Series(0.0, index=dates)
    m["spx_1m"] = spx_lvl.diff(21)
    m["spx_trend"] = spx_lvl.rolling(50).mean() - spx_lvl.rolling(200).mean()
    if mkt_news is not None and len(mkt_news):
        nb = news_block(mkt_news, dates)
        m["m_news_vol"], m["m_news_tone"], m["m_news_bull"], m["m_news_bear"] = nb["vol"].values, nb["tone"].values, nb["bull"].values, nb["bear"].values
    else:
        for k in ("m_news_vol", "m_news_tone", "m_news_bull", "m_news_bear"):
            m[k] = 0.0
    spx_idx = spx.reindex(dates, method="ffill") if spx is not None else pd.Series(1.0, index=dates)
    return dict(C=C, O=O, lr=lr, mkt=mkt, mkt_bench=mkt_bench, eligible=eligible, vol=vol, cost=cost, sell_tax=sell_tax,
                borrow=borrow, f=f, m=m[FEATS_MKT], dates=dates, tickers=list(C.columns), fxl=fxl, spx=spx_idx)


def ols_forecast(lr, mkt, f, dates, step=5, window=104):
    """Pooled cross-sectional OLS of next-week excess return on 5 classic signals, refit every week
    on the trailing 2 years of *already realised* outcomes. Output is its forecast (a simple model)."""
    regs = ["rev_1w", "rev_1m", "mom_12_1", "vol_60", "high_52w"]
    T, N = lr.shape
    cum = np.nan_to_num(lr.values).cumsum(0)
    mcum = np.nan_to_num(mkt.values).cumsum()
    grid = np.arange(260, T - 1, step)
    Xg, Yg = {}, {}
    for g in grid:
        X = np.stack([rank_gauss(f[k].values[g]) for k in regs], 1)
        Xg[g] = X
        if g + step < T:
            y = (cum[g + step] - cum[g]) - (mcum[g + step] - mcum[g])
            y[~np.isfinite(lr.values[g + 1:g + step + 1]).any(0)] = np.nan
            Yg[g] = y
    out = np.full((T, N), np.nan)
    for i, g in enumerate(grid):
        past = [h for h in grid[max(0, i - window):i] if h + step <= g]          # outcomes known at g
        if len(past) < 26:
            continue
        X = np.vstack([Xg[h] for h in past]); y = np.concatenate([Yg[h] for h in past])
        ok = np.isfinite(y)
        X1 = np.column_stack([np.ones(ok.sum()), X[ok]])
        beta = np.linalg.solve(X1.T @ X1 + 1e-3 * np.eye(X1.shape[1]), X1.T @ y[ok])
        nxt = grid[i + 1] if i + 1 < len(grid) else T
        out[g:nxt] = (np.column_stack([np.ones(N), Xg[g]]) @ beta)[None, :]
    return pd.DataFrame(out, index=dates, columns=lr.columns)


def model_payoff(lr, sig, dates, step=5, window=12):
    """Causal: at day t, the mean over the last `window` weeks of corr(signal at t-5, return t-5 -> t)."""
    T = len(dates); cum = np.nan_to_num(lr.values).cumsum(0); S = sig.values
    ic = np.full(T, np.nan)
    for t in range(step, T):
        a = S[t - step]; b = cum[t] - cum[t - step]
        ok = np.isfinite(a) & np.isfinite(lr.values[t])
        if ok.sum() > 10 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            ic[t] = np.corrcoef(pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank())[0, 1]
    s = pd.Series(ic, index=dates)
    v = s.rolling(window * step, min_periods=step * 4).mean() * 10        # scale ~ +-1
    return pd.DataFrame(np.repeat(v.values[:, None], lr.shape[1], 1), index=dates, columns=lr.columns)


def stock_X(P, t, idx, use_news=True):
    cols = []
    for k in FEATS_STOCK:
        if k in CONTEXT:
            cols.append(np.clip(np.nan_to_num(P["f"][k].values[t, idx]), -2.5, 2.5))
        elif k in NEWS and not use_news:
            cols.append(np.zeros(len(idx)))
        else:
            cols.append(rank_gauss(P["f"][k].values[t, idx]))
    return np.stack(cols, 1)


def mkt_scale(P, end):
    m = P["m"][P["dates"] <= pd.Timestamp(end)]
    return m.mean(), m.std().replace(0, 1.0).fillna(1.0)


def market_X(P, t, scale, use_news=True):
    mu, sd = scale
    x = np.array(((P["m"].iloc[t] - mu) / sd).clip(-2.5, 2.5).fillna(0.0).values, dtype=float)
    if not use_news:
        x[:4] = 0.0
    return x[None, :]


# ------------------------------------------------------------------------------- the strategy
def new_brains(seed, lr):
    sb = LovingBrain(len(FEATS_STOCK), seed=seed, lr=lr, threshold=0.5)
    mb = LovingBrain(len(FEATS_MKT), n_kc=500, seed=seed + 1000, lr=lr, threshold=0.5)
    return sb, mb


def run_v3(P, sb, mb, start, end, mscale, theta_m=1.0, short_stocks=False, K=20, step=5, keep_mult=3,
           learn=True, explore=0.0, use_news=True, love_gain=1.0, love_on=True, rng=None):
    """Weekly loop. Both a LONG book (top picks) and a SHORT book (bottom picks or market basket) are
    tracked every week as shadow portfolios; the market brain's mode decides which one is actually held.
    That makes the null tests exact (any mode sequence can be re-priced)."""
    rng = rng or np.random.default_rng(0)
    dates = P["dates"]; lr = P["lr"].values; elig = P["eligible"].values; cost = P["cost"].values
    vol = P["vol"].values; fxl = P["fxl"].values; spx = P["spx"].values
    cum = np.nan_to_num(lr).cumsum(0); fcum = fxl.cumsum(0)
    mcum = np.nan_to_num(P["mkt_bench"].values).cumsum()
    t_idx = np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0][::step]
    L, S = set(), set()
    mode_prev = "L"
    pend = None
    hist_net, hist_spx = [], []
    zm_mu, zm_var = getattr(mb, "zm_mu", 0.0), getattr(mb, "zm_var", 1.0)   # market brain's usual mood
    recs = []
    for n, t in enumerate(t_idx[:-1]):
        t2 = t_idx[n + 1]
        # ---- feelings about last week: pain/happiness, then love
        if pend is not None and learn:
            kc_s, idx_s, pos_s, out_s, paid_s, v_s, kc_m, pos_m, out_m, paid_m, v_m = pend
            ok = np.isfinite(out_s) & np.isfinite(v_s) & (v_s > 0)
            if ok.any():
                sb.learn(kc_s[ok], pos_s[ok], out_s[ok] / v_s[ok], paid_s[ok] / v_s[ok])
            if np.isfinite(out_m) and v_m > 0:
                mb.learn(kc_m, np.array([pos_m]), np.array([out_m / v_m]), np.array([paid_m / v_m]))
            if love_on and len(hist_net) >= 26:
                k52 = min(52, len(hist_net))
                lead = np.prod(1 + np.array(hist_net[-k52:])) - np.prod(1 + np.array(hist_spx[-k52:]))
                if lead > 0:
                    amt = love_gain * min(lead / 0.10, 1.0)
                    sb.love(kc_s, pos_s, amt); mb.love(kc_m, np.array([pos_m]), amt)
        idx = np.where(elig[t])[0]
        if len(idx) < 12:
            pend = None; continue
        k_t = min(K, len(idx) // 3)
        Xs = stock_X(P, t, idx, use_news)
        kc_s = sb.kenyon(Xs); _, score = sb.decide(kc_s)
        Xm = market_X(P, t, mscale, use_news)
        kc_m = mb.kenyon(Xm); _, zm = mb.decide(kc_m)
        zm = float(zm[0])
        # judge today's market view against the brain's usual view, so a constant bias doesn't decide
        zrel = (zm - zm_mu) / np.sqrt(zm_var + 1e-9)
        zm_mu += 0.02 * (zm - zm_mu); zm_var += 0.02 * ((zm - zm_mu) ** 2 - zm_var)
        mb.zm_mu, mb.zm_var = zm_mu, zm_var
        if explore:
            score = score + rng.normal(0, explore, len(score)); zrel += rng.normal(0, explore)
        mode = "S" if zrel < -theta_m else "L"
        order = idx[np.argsort(-score)]
        rank_top = {j: r for r, j in enumerate(order)}
        rank_bot = {j: r for r, j in enumerate(order[::-1])}
        # long book (always tracked)
        keepL = {j for j in L if j in rank_top and rank_top[j] < keep_mult * k_t}
        buysL = [j for j in order if j not in keepL][:max(k_t - len(keepL), 0)]
        sellsL = L - keepL; newL = keepL | set(buysL)
        # short book
        if short_stocks:
            keepS = {j for j in S if j in rank_bot and rank_bot[j] < keep_mult * k_t}
            buysS = [j for j in order[::-1] if j not in keepS][:max(k_t - len(keepS), 0)]
            sellsS = S - keepS; newS = keepS | set(buysS)
        else:
            newS, buysS, sellsS = set(idx.tolist()), [], set()
        # ---- prices over the coming week, in USD
        rl = np.where(np.isfinite(lr[t + 1:t2 + 1]).any(0), np.expm1(cum[t2] - cum[t]), 0.0)
        fx = np.expm1(fcum[t2] - fcum[t])
        usd_long = (1 + rl) * (1 + fx) - 1
        usd_short = -rl * (1 + fx) - BORROW * step / 252
        Lh = np.array(sorted(newL)); Sh = np.array(sorted(newS))
        retL = float(np.mean(usd_long[Lh])) * len(Lh) / k_t if len(Lh) else 0.0
        c = lambda j, sell: float(np.nan_to_num(cost[t, j], nan=0.05)) + (P["sell_tax"] if sell else 0.0)
        turnL = (sum(c(j, 0) for j in buysL) + sum(c(j, 1) for j in sellsL)) / k_t
        entryL = sum(c(j, 0) for j in newL) / k_t
        exitL = sum(c(j, 1) for j in L) / k_t
        if short_stocks:
            retS = float(np.mean(usd_short[Sh])) * len(Sh) / k_t if len(Sh) else 0.0
            turnS = (sum(c(j, 0) for j in buysS) + sum(c(j, 0) for j in sellsS)) / k_t
            entryS = sum(c(j, 0) for j in newS) / k_t
            exitS = sum(c(j, 0) for j in S) / k_t
        else:
            retS = float(np.mean(usd_short[idx])) + BORROW * step / 252     # index future: no borrow fee
            turnS, entryS, exitS = 0.0, BASKET_COST, BASKET_COST
        if mode == mode_prev:
            paid = turnL if mode == "L" else turnS
        else:
            paid = (entryL + exitS) if mode == "L" else (entryS + exitL)
        net = (retL if mode == "L" else retS) - paid
        spx_ret = float(spx[t2] / spx[t] - 1)
        mkt_ret = float(np.expm1(mcum[t2] - mcum[t]))
        recs.append(dict(date=dates[t], t=int(t), mode=mode, zm=zm, zrel=zrel, net=net, spx=spx_ret, mkt=mkt_ret, k=k_t,
                         retL=retL, retS=retS, turnL=turnL, turnS=turnS, entryL=entryL, entryS=entryS,
                         exitL=exitL, exitS=exitS, nbL=len(buysL), nsL=len(sellsL), nbS=len(buysS), nsS=len(sellsS),
                         elig=idx, Lh=Lh, Sh=Sh))
        hist_net.append(net); hist_spx.append(spx_ret)
        # ---- experiences to feel next week
        pos_s = np.zeros(len(idx))
        if mode == "L":
            pos_s[np.isin(idx, Lh)] = 1.0
        elif short_stocks:
            pos_s[np.isin(idx, Sh)] = -1.0
        out_s = np.log1p(rl[idx]) - np.log1p(mkt_ret)
        paid_s = np.array([c(j, 0) if ((mode == "L" and j in buysL) or (mode == "S" and short_stocks and j in buysS)) else 0.0
                           for j in idx])
        v_s = vol[t, idx] * np.sqrt(step)
        pos_m = 1.0 if mode == "L" else -1.0
        mv = np.nanstd(P["mkt"].values[max(0, t - 60):t]) * np.sqrt(step)
        pend = (kc_s, idx, pos_s, out_s, paid_s, v_s, kc_m, pos_m, np.log1p(mkt_ret), float(paid if mode != mode_prev else 0.0), mv)
        L, S, mode_prev = newL, newS, mode
    return recs


def price_modes(recs, modes):
    """Re-price any mode sequence on the recorded shadow books."""
    net, prev = [], "L"
    for r, m in zip(recs, modes):
        if m == prev:
            paid = r["turnL"] if m == "L" else r["turnS"]
        else:
            paid = (r["entryL"] + r["exitS"]) if m == "L" else (r["entryS"] + r["exitL"])
        net.append((r["retL"] if m == "L" else r["retS"]) - paid); prev = m
    return np.array(net)
