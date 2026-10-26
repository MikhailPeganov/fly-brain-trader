"""Fly-brain v3 experiment: news + simple models + maths school + love, long/short, 5 markets incl. MOEX.

PRE-REGISTERED before any 2024-26 v3 simulation:
  Markets: Tokyo small caps, Vietnam HOSE, Nordic small caps, UK small caps (AIM), MOEX (IMOEX stocks,
           research only: UK sanctions prohibit UK persons from dealing on the Moscow Exchange).
  Model:   stock brain + market brain (niche/v3.py), maths school first (niche/school.py), then real
           pre-training 2017-03..2021 (with exploration), learning through 2022-23, blind test
           2024-01-01..2026-10-23 with online learning. Brain seed 0. Love gain 1.0 (not tuned).
  Tuned on 2022-23 only: theta_m (how bearish the market brain must be before shorting) in {0.75, 1.5},
           pooled across markets, 2 tuning seeds.
  H1 (beats S&P 500): weekly USD net return minus S&P 500 total return; one-sided Newey-West t-test.
  H2a (stock-picking edge): fly vs random stock picks with identical trade counts and the fly's own
           long/short timing (permutation, 2,000 runs).
  H2b (timing edge): fly's long/short timing vs the same timing shifted to random weeks (circular
           shift, 2,000 runs), same stock books.
  Holm correction: 5 tests for H1, 10 for H2. Ablations (news off, school off, love off, never short)
  are secondary and descriptive.
NEWS SCOPE: the GDELT archive throttled this server, so the historical news senses are market-level only
  (per-company news exists only in the live crawler); the stock brain's company-news senses are zero.
"""
import copy, json, os, pickle, sys, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
from niche import data as D, news as NW, school, v3
from flybrain import stats as ST

MARKETS = ["tokyo", "vietnam", "nordic", "aim", "moex"]
LABEL = {"tokyo": "Tokyo small caps", "vietnam": "Vietnam HOSE", "nordic": "Nordic small caps", "aim": "UK small caps (AIM)",
         "moex": "MOEX (Russia)"}
PRE = ("2017-03-01", "2021-12-31"); VAL = ("2022-01-01", "2023-12-31"); TEST = ("2024-01-01", "2026-10-23")
OUT = "results/v3"; CACHE = "data/v3cache"
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
LR = 0.01


def fx_daily(currencies):
    path = f"{CACHE}/fx.pkl"
    fx = pd.read_pickle(path) if os.path.exists(path) else {}
    for c in currencies:
        base = "GBP" if c == "GBp" else c
        if base in fx or base == "USD":
            continue
        s = D._retry(lambda: yf.download(f"{base}=X", start="2014-06-01", end="2026-10-24", progress=False, auto_adjust=False)["Close"])
        fx[base] = 1.0 / pd.Series(np.asarray(s).ravel(), index=s.index)       # USD per 1 unit of local currency
        pd.to_pickle(fx, path)
    return fx


def spx_series():
    s = pd.read_csv("data/daily_SPY.csv", index_col=0, parse_dates=True)["close"]
    return s


def panel(key):
    path = f"{CACHE}/panel_{key}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    p = D.prices(key); u = D.universe(key); sp = D.spreads(key)
    cols = list(p["Close"].columns)
    info = {x["symbol"]: x for x in u}
    fx = fx_daily(sorted({info[c]["currency"] for c in cols}))
    idx = p["Close"].index
    fxd = pd.DataFrame({c: (fx["GBP" if info[c]["currency"] == "GBp" else info[c]["currency"]].reindex(idx, method="ffill")
                            / (100 if info[c]["currency"] == "GBp" else 1)) if info[c]["currency"] != "USD" else 1.0
                        for c in cols}, index=idx).bfill()
    fx_const = pd.Series({c: info[c]["fx_usd"] for c in cols})
    hs = pd.Series({c: sp["spread"].get(c, sp["median"]) / 2 for c in cols})
    P = v3.prepare_panel(p, fx_const, fxd, hs, D.SELL_TAX.get(key, 0.0), spx_series(), mkt_news=NW.load_market(key))
    P["key"] = key
    pickle.dump(P, open(path, "wb"))
    return P


def schooled(seed, log):
    path = f"{CACHE}/school_{seed}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    sb, mb = v3.new_brains(seed, LR)
    rep = school.school(sb, mb, seed=seed, max_rounds=6, school_lr=0.03, log=lambda *a: None)
    pickle.dump((sb, mb, rep), open(path, "wb"))
    log(f"school seed {seed}: " + str({w: (round(v['fly_ic'], 3), round(v['oracle_ic'], 3), v['passed']) for w, v in rep['final_exam'].items()}))
    return sb, mb, rep


def train(P, seed, theta, upto, use_news=True, use_school=True, love_on=True, log=print):
    if use_school:
        sb, mb, _ = schooled(seed, log); sb, mb = copy.deepcopy(sb), copy.deepcopy(mb)
    else:
        sb, mb = v3.new_brains(seed, LR)
    ms = v3.mkt_scale(P, PRE[1])
    kw = dict(theta_m=theta, short_stocks=(P["key"] == "moex"), use_news=use_news, love_on=love_on)
    v3.run_v3(P, sb, mb, *PRE, ms, explore=0.5, rng=np.random.default_rng(seed), **kw)
    if upto > PRE[1]:
        v3.run_v3(P, sb, mb, VAL[0], upto, ms, **kw)
    return sb, mb, ms, kw


# --------------------------------------------------------------------------- tests
def h1(recs):
    ex = np.array([r["net"] - r["spx"] for r in recs])
    t, p, _ = ST.newey_west_t(ex)
    return float(ex.mean() * 52), t, p


def usd_week(P, t, t2):
    lr = P["lr"].values
    cum = np.nan_to_num(lr[t + 1:t2 + 1]).sum(0); fx = P["fxl"].values[t + 1:t2 + 1].sum(0)
    rl = np.where(np.isfinite(lr[t + 1:t2 + 1]).any(0), np.expm1(cum), 0.0)
    return rl, np.expm1(fx)


def null_selection(P, recs, n_perm=2000, seed=0):
    """Random stock books with the fly's per-week trade counts, combined with the fly's own modes."""
    rng = np.random.default_rng(seed); cost = P["cost"].values; tax = P["sell_tax"]; short_stocks = P["key"] == "moex"
    W = len(recs)
    tt = [r["t"] for r in recs] + [recs[-1]["t"] + 5]
    weeks = [usd_week(P, tt[i], min(tt[i + 1], len(P["dates"]) - 1)) for i in range(W)]
    modes = [r["mode"] for r in recs]
    obs = np.mean([r["net"] - r["spx"] for r in recs])
    null = np.empty(n_perm)
    c = lambda t, j, sell: float(np.nan_to_num(cost[t, j], nan=0.05)) + (tax if sell else 0.0)
    for b in range(n_perm):
        L, S = set(), set(); prev = "L"; tot = 0.0
        for w, r in enumerate(recs):
            E = set(r["elig"].tolist()); k = r["k"]; t = r["t"]; rl, fx = weeks[w]
            def step(book, nb, ns, sell_tax):
                forced = {j for j in book if j not in E}; book = book - forced
                extra = max(ns - len(forced), 0)
                drop = set(rng.choice(sorted(book), size=min(extra, len(book)), replace=False).tolist()) if extra and book else set()
                book = book - drop; pool = np.array(sorted(E - book))
                need = max(min(k, len(book) + nb) - len(book), 0)
                buys = rng.choice(pool, size=min(need, len(pool)), replace=False).tolist() if need and len(pool) else []
                sells = forced | drop
                turn = (sum(c(t, j, 0) for j in buys) + sum(c(t, j, sell_tax) for j in sells)) / k
                return book | set(buys), turn, sells
            oldL, oldS = L, S
            L, turnL, _ = step(L, r["nbL"], r["nsL"], 1)
            retL = float(np.mean(((1 + rl) * (1 + fx) - 1)[sorted(L)])) * len(L) / k if L else 0.0
            if short_stocks:
                S, turnS, _ = step(S, r["nbS"], r["nsS"], 0)
                retS = float(np.mean((-rl * (1 + fx) - v3.BORROW * 5 / 252)[sorted(S)])) * len(S) / k if S else 0.0
                entryS = sum(c(t, j, 0) for j in S) / k; exitS = sum(c(t, j, 0) for j in oldS) / k
            else:
                retS, turnS, entryS, exitS = r["retS"], 0.0, v3.BASKET_COST, v3.BASKET_COST
            entryL = sum(c(t, j, 0) for j in L) / k; exitL = sum(c(t, j, 1) for j in oldL) / k
            m = modes[w]
            paid = (turnL if m == "L" else turnS) if m == prev else ((entryL + exitS) if m == "L" else (entryS + exitL))
            tot += (retL if m == "L" else retS) - paid - r["spx"]; prev = m
        null[b] = tot / W
    return float((1 + (null >= obs).sum()) / (n_perm + 1)), null


def null_timing(recs, n_perm=2000, seed=0):
    modes = np.array([r["mode"] for r in recs]); W = len(modes)
    if (modes == "S").sum() == 0 or (modes == "L").sum() == 0:
        return float("nan"), np.array([])
    rng = np.random.default_rng(seed); spx = np.array([r["spx"] for r in recs])
    obs = np.mean(v3.price_modes(recs, modes) - spx)
    null = np.array([np.mean(v3.price_modes(recs, np.roll(modes, rng.integers(8, W - 8))) - spx) for _ in range(n_perm)])
    return float((1 + (null >= obs).sum()) / (n_perm + 1)), null


def holm(p):
    keys = list(p); arr = np.array([1.0 if not np.isfinite(p[k]) else p[k] for k in keys]); o = np.argsort(arr); m = len(arr)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * arr[i])); adj[i] = run
    return {k: float(a) for k, a in zip(keys, adj)}


def summarize(recs):
    net = np.array([r["net"] for r in recs]); spx = np.array([r["spx"] for r in recs])
    yrs = len(recs) / 52
    return dict(weeks=len(recs), cagr_usd=float(np.prod(1 + net) ** (1 / yrs) - 1), spx_cagr=float(np.prod(1 + spx) ** (1 / yrs) - 1),
                total_usd=float(np.prod(1 + net) - 1), spx_total=float(np.prod(1 + spx) - 1),
                vol_ann=float(net.std() * np.sqrt(52)), sharpe=float(net.mean() / (net.std() + 1e-12) * np.sqrt(52)),
                share_short_weeks=float(np.mean([r["mode"] == "S" for r in recs])),
                max_drawdown=float(np.min(np.cumprod(1 + net) / np.maximum.accumulate(np.cumprod(1 + net)) - 1)))


def main():
    t0 = time.time(); logf = open(f"{OUT}/run.log", "a")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log("=== v3 run", time.strftime("%Y-%m-%d %H:%M:%S"))
    Ps = {}
    for k in MARKETS:
        Ps[k] = panel(k)
        nm = Ps[k]["m"]["m_news_vol"].abs().sum() > 0
        log(k, "stocks", len(Ps[k]["tickers"]), "market news", "yes" if nm else "NO")
    for s in (0, 101, 102):
        schooled(s, log)
    out = dict(prereg=__doc__, school={s: pickle.load(open(f"{CACHE}/school_{s}.pkl", "rb"))[2] for s in (0, 101, 102)})

    # ---------------- validation
    grid = []
    for theta in (0.75, 1.5):
        ts = []
        for k in MARKETS:
            for seed in (101, 102):
                sb, mb, ms, kw = train(Ps[k], seed, theta, PRE[1], log=log)
                recs = v3.run_v3(Ps[k], sb, mb, *VAL, ms, **kw)
                ts.append(h1(recs)[1])
        grid.append(dict(theta_m=theta, val_t=float(np.mean(ts)))); log("val", grid[-1])
    theta = max(grid, key=lambda g: g["val_t"])["theta_m"]
    out.update(validation=grid, locked=dict(theta_m=theta, lr=LR, love_gain=1.0), locked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    log("LOCKED theta_m", theta)

    # ---------------- blind test + ablations
    res, series, P1, P2 = {}, {}, {}, {}
    variants = {"full": {}, "no_news": dict(use_news=False), "no_school": dict(use_school=False), "no_love": dict(love_on=False)}
    for k in MARKETS:
        res[k] = {}
        for name, opt in variants.items():
            sb, mb, ms, kw = train(Ps[k], 0, theta, VAL[1], log=log, **opt)
            recs = v3.run_v3(Ps[k], sb, mb, *TEST, ms, **kw)
            ex_ann, t, p = h1(recs)
            res[k][name] = dict(summarize(recs), excess_vs_spx_ann=ex_ann, nw_t=t, p_beats_spx=p)
            if name == "full":
                full = recs
                pa, na = null_selection(Ps[k], recs); pb, nb = null_timing(recs)
                res[k]["full"].update(p_selection=pa, p_timing=pb)
                never = v3.price_modes(recs, ["L"] * len(recs)); spx = np.array([r["spx"] for r in recs])
                res[k]["never_short"] = dict(excess_vs_spx_ann=float((never - spx).mean() * 52),
                                             cagr_usd=float(np.prod(1 + never) ** (52 / len(recs)) - 1))
                series[k] = dict(dates=[str(r["date"].date()) for r in recs], net=[r["net"] for r in recs],
                                 spx=[r["spx"] for r in recs], mode=[r["mode"] for r in recs], null_sel=na.tolist())
            else:
                d = np.array([r["net"] for r in full]) - np.array([r["net"] for r in recs])
                n = min(len(d), len(full))
                tt, pp, _ = ST.newey_west_t(d[:n])
                res[k][name]["full_minus_this_ann"] = float(d.mean() * 52); res[k][name]["p_full_better"] = pp
            log(k, name, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k][name].items()})
    out.update(test=res, holm_h1=holm({k: res[k]["full"]["p_beats_spx"] for k in MARKETS}),
               holm_h2=holm({**{f"{k}_selection": res[k]["full"]["p_selection"] for k in MARKETS},
                             **{f"{k}_timing": res[k]["full"]["p_timing"] for k in MARKETS}}))
    log("Holm H1", out["holm_h1"]); log("Holm H2", out["holm_h2"])
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)
    json.dump(series, open(f"{OUT}/series.json", "w"), default=float)
    log("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
