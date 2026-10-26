"""Niche-market blind test of the pain/happiness fly brain.

PRE-REGISTERED (written before any 2024-26 data was simulated):
  Markets: Vietnam HOSE, Tokyo small caps, Nordic small caps (First North proxy), UK small caps (AIM proxy)
  Strategies: A slow picker, B shock rebound (see niche/strategies.py)
  Periods: pre-train 2016-2021, validation 2022-2023 (choose lr and B's buy threshold, pooled
           across markets), blind test 2024-01-02 .. 2026-10-23 with online learning, brain seed 0.
  Question 1, VIABILITY: does the strategy beat the market of tradeable stocks after all costs?
           One-sided Newey-West t-test on the excess-return series.
  Question 2, MODEL EDGE: does the fly pick better than a random picker following identical rules
           (same number of buys/sells each week for A; same number of buys among each day's
           shocks for B)? Permutation test, net of costs.
  Each question: 8 tests, Holm correction, family-wise alpha = 0.05.
BUG FIX (26 Oct, after first full run): strategy B's market benchmark averaged log returns, which
  understates an equal-weight portfolio; replaced with simple-return equal weight and rerun in full.
DISCLOSURE: while debugging, an early version of strategy A (weekly trading, hold while in top 2K)
  was run once on Vietnam's 2024-26 test period (it lost money). The one change made afterwards,
  letting validation choose a lower trading frequency, is motivated by validation-period costs.
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from niche import strategies as S
from flybrain import stats as ST

MARKETS = ["vietnam", "tokyo", "nordic", "aim"]
LABELS = {"vietnam": "Vietnam HOSE", "tokyo": "Tokyo small caps", "nordic": "Nordic small caps (First North)",
          "aim": "UK small caps (AIM)"}
PRE_END, VAL = "2021-12-31", ("2022-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-10-23")
OUT = "results/niche"
os.makedirs(OUT, exist_ok=True)


def first_date(P):
    ok = P["f"]["mom_12_1"].notna().sum(1) > 20
    return str(P["dates"][ok.values.argmax()].date())


def pretrain_A(P, lr, seed, scale, upto, step=5, keep_mult=2):
    b = S.new_brain(len(S.FEATURES_A), seed, lr)
    rng = np.random.default_rng(seed)
    S.run_picker(P, b, first_date(P), PRE_END, scale, step=step, keep_mult=keep_mult, explore=0.5, rng=rng)
    if upto > PRE_END:
        S.run_picker(P, b, "2022-01-01", upto, scale, step=step, keep_mult=keep_mult)
    return b


def pretrain_B(P, lr, theta, seed, scale, upto):
    b = S.new_brain(len(S.FEATURES_B), seed, lr)
    rng = np.random.default_rng(seed)
    S.run_rebound(P, b, first_date(P), PRE_END, scale, theta=theta, explore=0.7, rng=rng)
    if upto > PRE_END:
        S.run_rebound(P, b, "2022-01-01", upto, scale, theta=theta)
    return b


# ------------------------------------------------------------------ tests
def picker_tests(P, rows, n_perm=2000, seed=0, step=5):
    df = pd.DataFrame(rows)
    ex = (df["net"] - df["mkt"]).values
    t, p_v, _ = ST.newey_west_t(ex)
    obs = ex.mean()
    rng = np.random.default_rng(seed)
    lr = P["lr"].values; cum = np.nan_to_num(lr).cumsum(0); cost = P["cost"].values; tax = P["sell_tax"]
    # weekly simple returns per stock, aligned with each record
    t_of = {d: i for i, d in enumerate(P["dates"])}
    tt = [t_of[d] for d in df["date"]]
    nxt = tt[1:] + [min(tt[-1] + step, len(P["dates"]) - 1)]
    R = [np.expm1(cum[b] - cum[a]) for a, b in zip(tt, nxt)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        held = set(); acc = 0.0
        for w, r in enumerate(rows):
            E = set(r["elig"]); kk = r["k"]
            forced = {j for j in held if j not in E}
            held -= forced
            extra = max(r["n_sell"] - len(forced), 0)
            if extra and held:
                drop = rng.choice(sorted(held), size=min(extra, len(held)), replace=False)
                held -= set(drop.tolist()); sells = forced | set(drop.tolist())
            else:
                sells = forced
            pool = np.array(sorted(E - held))
            nb = max(r["n_hold"] - len(held), 0)
            buys = rng.choice(pool, size=min(nb, len(pool)), replace=False).tolist() if nb and len(pool) else []
            held |= set(buys)
            hold = np.array(sorted(held))
            gross = np.mean(np.nan_to_num(R[w][hold])) if len(hold) else 0.0
            wgt = 1.0 / kk
            c = (sum(cost[tt[w], j] for j in buys) + sum(np.nan_to_num(cost[tt[w], j], nan=0.05) + tax for j in sells)) * wgt
            net = gross * len(hold) / kk - c
            acc += net - r["mkt"]
        null[k] = acc / len(rows)
    p_e = (1 + (null >= obs).sum()) / (n_perm + 1)
    return dict(n_weeks=len(df), excess_ann=float(obs * 252 / step), excess_week=float(obs), nw_t=t, p_viability=p_v,
                p_edge=float(p_e), null_mean_ann=float(null.mean() * 252 / step), null_p95_ann=float(np.percentile(null, 95) * 252 / step),
                net_total=float(np.prod(1 + df["net"]) - 1), mkt_total=float(np.prod(1 + df["mkt"]) - 1),
                gross_excess_ann=float((df["gross"] - df["mkt"]).mean() * 252 / step),
                turnover_per_rebalance=float((df["n_buy"] / df["k"]).mean()), avg_k=float(df["k"].mean()),
                avg_eligible=float(df["n_elig"].mean())), null


def rebound_tests(events, n_perm=5000, seed=0):
    ev = pd.DataFrame(events)
    bought = ev[ev.bought]
    if len(bought) < 10:
        return dict(n_events=len(ev), n_trades=len(bought), note="too few trades"), np.array([])
    obs = bought["excess_net"].mean()
    # viability: monthly average of trades' net excess, NW t-test
    m = bought.groupby(pd.to_datetime(bought["date"]).dt.to_period("M"))["excess_net"].mean()
    t, p_v, _ = ST.newey_west_t(m.values, lags=3)
    # edge: random picks among the same day's shocks, same number of buys per day
    rng = np.random.default_rng(seed)
    groups = [(g["excess_net"].values, int(g["bought"].sum())) for _, g in ev.groupby("t") if g["bought"].any()]
    nb = sum(k for _, k in groups)
    null = np.empty(n_perm)
    for k in range(n_perm):
        s = 0.0
        for x, kk in groups:
            s += x[rng.choice(len(x), kk, replace=False)].sum()
        null[k] = s / nb
    p_e = (1 + (null >= obs).sum()) / (n_perm + 1)
    if null.std() < 1e-12:      # fly bought every shock it saw: no selection to test
        p_e = float("nan")
    return dict(n_events=len(ev), n_trades=len(bought), trades_per_month=float(len(bought) / max(len(m), 1)),
                excess_per_trade=float(obs), gross_excess_per_trade=float((bought["excess_net"] + bought["cost"]).mean()),
                avg_cost=float(bought["cost"].mean()), hit=float((bought["excess_net"] > 0).mean()),
                buy_all_excess=float(ev["excess_net"].mean()), nw_t=t, p_viability=p_v, p_edge=float(p_e),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95))), null


def holm(pvals):
    keys = list(pvals); p = np.array([pvals[k] for k in keys]); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return {k: float(a) for k, a in zip(keys, adj)}


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    Ps = {}
    for k in MARKETS:
        Ps[k] = S.prepare(k)
        el = Ps[k]["eligible"]
        print(k, "stocks", len(Ps[k]["tickers"]), "first date", first_date(Ps[k]),
              "avg tradeable 2024+", round(el[el.index >= "2024-01-01"].sum(1).mean(), 1), flush=True)
    scales = {k: S.ts_scale(Ps[k], PRE_END) for k in MARKETS}
    log = dict(prereg=__doc__, markets=LABELS)

    # ---------- validation (pooled across markets), 2 tuning seeds
    gridA, gridB = [], []
    for lr in (0.01, 0.03):
        # trading frequency: weekly with wide hysteresis, or every 4 weeks
        for step, keep in ((5, 3), (20, 2)):
            sc = []
            for k in MARKETS:
                for seed in (101, 102):
                    b = pretrain_A(Ps[k], lr, seed, scales[k], PRE_END, step, keep)
                    rows, _ = S.run_picker(Ps[k], b, *VAL, scales[k], step=step, keep_mult=keep)
                    df = pd.DataFrame(rows); sc.append(ST.newey_west_t((df.net - df.mkt).values)[0])
            gridA.append(dict(lr=lr, step=step, keep_mult=keep, val_t=float(np.mean(sc)))); print("A", gridA[-1], flush=True)
        for theta in (0.0, 0.5, 1.0):
            sc = []
            for k in MARKETS:
                for seed in (101, 102):
                    b = pretrain_B(Ps[k], lr, theta, seed, scales[k], PRE_END)
                    ev = pd.DataFrame(S.run_rebound(Ps[k], b, *VAL, scales[k], theta=theta))
                    bt = ev[ev.bought] if len(ev) else ev
                    sc.append(bt["excess_net"].mean() / (bt["excess_net"].std() + 1e-9) * np.sqrt(len(bt)) if len(bt) >= 10 else -9)
            gridB.append(dict(lr=lr, theta=theta, val_t=float(np.mean(sc)))); print("B", gridB[-1], flush=True)
    cfgA = max(gridA, key=lambda g: g["val_t"]); cfgB = max(gridB, key=lambda g: g["val_t"])
    log.update(validation_A=gridA, validation_B=gridB, locked_A=cfgA, locked_B=cfgB,
               locked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("LOCKED A", cfgA, "B", cfgB, flush=True)

    # ---------- blind test
    res, series, nulls = {}, {}, {}
    for k in MARKETS:
        P = Ps[k]
        b = pretrain_A(P, cfgA["lr"], 0, scales[k], VAL[1], cfgA["step"], cfgA["keep_mult"])
        rows, _ = S.run_picker(P, b, *TEST, scales[k], step=cfgA["step"], keep_mult=cfgA["keep_mult"], record=True)
        rA, nA = picker_tests(P, rows, step=cfgA["step"])
        b = pretrain_B(P, cfgB["lr"], cfgB["theta"], 0, scales[k], VAL[1])
        evs = S.run_rebound(P, b, *TEST, scales[k], theta=cfgB["theta"])
        rB, nB = rebound_tests(evs)
        res[k] = dict(A=rA, B=rB)
        df = pd.DataFrame(rows)
        series[k] = dict(A=dict(dates=[str(d.date()) for d in df.date], net=df.net.tolist(), mkt=df.mkt.tolist()),
                         B=[dict(date=str(e["date"].date()), x=e["excess_net"], bought=e["bought"]) for e in evs])
        nulls[k] = dict(A=nA.tolist(), B=nB.tolist())
        print(k, "A", {x: round(v, 4) if isinstance(v, float) else v for x, v in rA.items()}, flush=True)
        print(k, "B", {x: round(v, 4) if isinstance(v, float) else v for x, v in rB.items()}, flush=True)
    pv = {f"{k}_{s}": res[k][s].get("p_viability", 1.0) for k in MARKETS for s in "AB"}
    pe = {f"{k}_{s}": (1.0 if not np.isfinite(res[k][s].get("p_edge", 1.0)) else res[k][s].get("p_edge", 1.0))
          for k in MARKETS for s in "AB"}
    log.update(test=res, holm_viability=holm(pv), holm_edge=holm(pe))
    print("Holm viability", log["holm_viability"]); print("Holm edge", log["holm_edge"])
    json.dump(log, open(f"{OUT}/results.json", "w"), indent=1, default=str)
    json.dump(dict(series=series, nulls=nulls), open(f"{OUT}/arrays.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
