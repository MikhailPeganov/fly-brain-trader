"""'Maths basics' school: before seeing a real market, the fly is trained on synthetic markets that each
obey one textbook model, and has to pass an exam in each.

  random    : pure random walk with market beta. Nothing is predictable; passing = not imagining a signal.
  momentum  : each stock has a slowly drifting expected return (AR(1) drift). Trend/momentum senses work.
  reversion : prices wobble around a fundamental value (Ornstein-Uhlenbeck). Mean-reversion senses work.
  lowvol    : quieter stocks earn more, and volatility clusters (low-volatility anomaly + GARCH-like vol).
  news      : a weekly hidden shock shows up in bullish/bearish keyword counts and tone, and moves prices
              the following week. News senses work.

Exam = information coefficient (IC): the rank correlation between the fly's stock scores and the next
week's returns on a fresh, unseen episode, compared with the IC of the true hidden signal (the 'oracle').
Pass mark: at least half the oracle's IC (random world: |IC| below 0.02).
"""
import copy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import v3

WORLDS = ["random", "momentum", "reversion", "lowvol", "news"]


def make_world(kind, seed, N=100, T=1300):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2001-01-01", periods=T)
    beta = rng.uniform(0.6, 1.4, N); sig = rng.uniform(0.012, 0.03, N)
    mdrift = np.zeros(T); m = rng.normal(0.0002, 0.009, T)
    mu = np.zeros((T, N)); extra = np.zeros((T, N)); sigt = np.tile(sig, (T, 1)); oracle = np.zeros((T, N))
    weekly = np.arange(T) // 5
    s_stock = rng.normal(size=(weekly.max() + 2, N)); s_mkt = rng.normal(size=weekly.max() + 2)
    if kind == "momentum":
        for t in range(1, T):
            mu[t] = 0.996 * mu[t - 1] + rng.normal(0, 0.0012 * np.sqrt(1 - 0.996 ** 2), N)
            mdrift[t] = 0.995 * mdrift[t - 1] + rng.normal(0, 0.0006 * np.sqrt(1 - 0.995 ** 2))
        oracle = mu.copy()
    elif kind == "reversion":
        x = np.zeros((T, N))
        for t in range(1, T):
            x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.012, N)
        extra = np.vstack([np.zeros(N), np.diff(x, axis=0)])
        oracle = -x
    elif kind == "lowvol":
        h = np.zeros((T, N))
        for t in range(1, T):
            h[t] = 0.97 * h[t - 1] + rng.normal(0, 0.1, N)
        sigt = sig * np.exp(h)
        mu = np.tile(0.0006 - 0.12 * (sig - 0.021), (T, 1))
        oracle = -sigt
    elif kind == "news":
        wk = weekly
        mu = 0.004 * s_stock[np.maximum(wk - 1, 0)] / 5.0          # last week's news moves this week
        mu[wk == 0] = 0
        mdrift = 0.0015 * s_mkt[np.maximum(wk - 1, 0)] / 5.0
        oracle = s_stock[np.maximum(wk - 1, 0)]                    # news already published (last week's)
    r = beta * (m + mdrift)[:, None] + mu + sigt * rng.normal(size=(T, N)) + extra
    C = 20 * np.exp(np.cumsum(r, 0))
    O = np.vstack([C[0], C[:-1]]) * np.exp(rng.normal(0, 0.003, (T, N)))
    H = np.maximum(O, C) * np.exp(np.abs(rng.normal(0, 0.006, (T, N))))
    Lo = np.minimum(O, C) * np.exp(-np.abs(rng.normal(0, 0.006, (T, N))))
    V = np.exp(rng.normal(np.log(2e6), 0.5, (T, N))) / C
    cols = [f"S{i:03d}" for i in range(N)]
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols)
    p = dict(Open=mk(O), High=mk(H), Low=mk(Lo), Close=mk(C), Volume=mk(V))
    # news: daily calendar series
    cal = pd.date_range(dates[0], dates[-1], freq="D")
    cw = np.searchsorted(dates.values, cal.values, side="right") - 1
    cwk = weekly[np.clip(cw, 0, T - 1)]
    stock_news, mkt_news = None, None
    if kind == "news":
        stock_news = {}
        for i, c in enumerate(cols):
            s = s_stock[cwk, i]
            bull = rng.poisson(1.0 * np.exp(0.8 * s)); bear = rng.poisson(1.0 * np.exp(-0.8 * s)); neu = rng.poisson(3, len(cal))
            stock_news[c] = pd.DataFrame({"all": bull + bear + neu, "bull": bull, "bear": bear, "norm": 1e6,
                                          "tone": 1.5 * s + rng.normal(0, 1, len(cal))}, index=cal)
        sm = s_mkt[cwk]
        bull = rng.poisson(20 * np.exp(0.5 * sm)); bear = rng.poisson(20 * np.exp(-0.5 * sm)); neu = rng.poisson(60, len(cal))
        mkt_news = pd.DataFrame({"all": bull + bear + neu, "bull": bull, "bear": bear, "norm": 1e6,
                                 "tone": 1.0 * sm + rng.normal(0, 0.5, len(cal))}, index=cal)
    ones = pd.Series(1.0, index=cols)
    P = v3.prepare_panel(p, ones, mk(np.ones((T, N))), ones * 0.001, 0.0, spx=None,
                         mkt_news=mkt_news, stock_news=stock_news)
    P["oracle"] = pd.DataFrame(oracle, index=dates, columns=cols)
    return P


def exam(P, sb, mb, step=5):
    """IC of the fly's scores vs the oracle's, on this (unseen) episode, without learning."""
    sb2, mb2 = copy.deepcopy(sb), copy.deepcopy(mb)
    dates = P["dates"]; lr = P["lr"].values; cum = np.nan_to_num(lr).cumsum(0); el = P["eligible"].values
    mscale = v3.mkt_scale(P, dates[-1])
    fly_ic, or_ic = [], []
    for t in range(300, len(dates) - step - 1, step):
        idx = np.where(el[t])[0]
        if len(idx) < 20:
            continue
        X = v3.stock_X(P, t, idx)
        kc = sb2.kenyon(X); _, score = sb2.decide(kc)
        fwd = cum[t + step, idx] - cum[t, idx]
        a = spearmanr(score, fwd)[0] if np.std(score) > 0 else 0.0
        o = P["oracle"].values[t, idx]
        b = spearmanr(o, fwd)[0] if np.std(o) > 0 else 0.0
        fly_ic.append(a); or_ic.append(b)
    return float(np.nanmean(fly_ic)), float(np.nanmean(or_ic))


def school(sb, mb, seed=0, max_rounds=6, school_lr=0.03, log=print):
    """Interleaved lessons (every world, every round, shuffled) so later worlds don't overwrite earlier
    ones; a world counts as passed once its exam is passed. Stops when every world has been passed."""
    lr0 = (sb.lr, mb.lr); sb.lr = mb.lr = school_lr
    report = []
    passed = {w: False for w in WORLDS}
    k = 0
    rng = np.random.default_rng(seed)
    for rnd in range(1, max_rounds + 1):
        for w in rng.permutation(WORLDS):
            k += 1
            Ptr = make_world(w, seed * 1000 + k)
            v3.run_v3(Ptr, sb, mb, Ptr["dates"][260], Ptr["dates"][-1], v3.mkt_scale(Ptr, Ptr["dates"][-1]),
                      explore=0.5, love_on=False, rng=np.random.default_rng(k))
            fi, oi = exam(make_world(w, seed * 1000 + 500 + k), sb, mb)
            ok = abs(fi) < 0.02 if w == "random" else fi >= 0.5 * oi
            passed[w] = passed[w] or ok
            report.append(dict(round=rnd, world=w, fly_ic=fi, oracle_ic=oi, passed=ok))
            log(report[-1])
        if all(passed.values()):
            break
    sb.lr, mb.lr = lr0
    final = {}
    for w in WORLDS:
        fi, oi = exam(make_world(w, seed * 1000 + 900 + WORLDS.index(w)), sb, mb)
        final[w] = dict(fly_ic=fi, oracle_ic=oi, pct_of_oracle=(fi / oi if w != "random" and oi else None),
                        passed=(abs(fi) < 0.02) if w == "random" else fi >= 0.5 * oi)
        log(("FINAL", w, final[w]))
    return dict(lessons=report, final_exam=final)
