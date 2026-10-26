"""'Live day' test: replay Friday 23 Oct 2026 bar by bar (5-minute bars) on the 20 hidden
stocks. The brain uses the hyperparameters locked by experiments/us_blind_test.py (no intraday tuning).

Test A (headline): pre-train on the 40 training stocks over all earlier days in the 60-day
  window; the hidden stocks AND the final day are never seen until the replay.
Test B (more statistical power): pre-train on training stocks for the first 50 days, then
  trade the hidden stocks for the last 10 days with online learning.
"""
import json, os, copy, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from flybrain import pipeline as P, brain as B, sim as S, stats as ST, data, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
OUT = os.path.join(ROOT, "results", "us_blind_test")
BARS_PER_YEAR = 78 * 252


def load():
    syms = data.UNIVERSE
    p = data.panel(syms + [data.MARKET], kind="intraday")
    keep = ~((p["close"].index.hour == 16) & (p["close"].index.minute == 0))
    p = {k: v[keep] for k, v in p.items()}
    mkt = {k: v[[data.MARKET]].iloc[:, 0] for k, v in p.items()}
    p = {k: v[syms] for k, v in p.items()}
    f, fwd, vol = features.intraday_features(p, mkt)
    days = pd.Index(sorted(set(fwd.index.date)))
    return f, fwd, vol, days


def tensors(f, fwd, vol, day_index, pre_days, train_tk):
    dates = np.array(fwd.index.date)
    tmask = np.isin(dates, pre_days)
    names = list(features.INTRADAY_FACTORS)
    fz = {}
    for k in names:
        x = f[k][train_tk][tmask].values.ravel(); x = x[np.isfinite(x)]
        med = np.median(x); iqr = np.subtract(*np.percentile(x, [75, 25]))
        s = iqr / 1.349 if iqr > 0 else np.std(x) + 1e-9
        fz[k] = ((f[k] - med) / s).clip(-3, 3)
    X = features.to_tensor(fz, names)
    vt = float(np.nanmedian(vol[train_tk][tmask].values))
    return X, vt, dates


def sub(X, fwd, vol, dates, days, tk, all_tk):
    m = np.isin(dates, days)
    cols = [all_tk.index(t) for t in tk]
    return dict(X=X[m][:, cols], fwd=fwd.values[m][:, cols], vol=vol.values[m][:, cols],
                idx=fwd.index[m])


def run_block(cfg, f, fwd, vol, days, pre_days, test_days, train_tk, hidden_tk, n_perm):
    all_tk = list(fwd.columns)
    X, vt, dates = tensors(f, fwd, vol, days, pre_days, train_tk)
    pre = sub(X, fwd, vol, dates, pre_days, train_tk, all_tk)
    test = sub(X, fwd, vol, dates, test_days, hidden_tk, all_tk)
    b = B.MushroomBodyTrader(X.shape[2], seed=0, **cfg)
    for _ in range(2):
        S.run(b, pre["X"], pre["fwd"], pre["vol"], P.INTRADAY_COST, vt, explore=0.5)
    r = S.run(b, test["X"], test["fwd"], test["vol"], P.INTRADAY_COST, vt, learn=True)
    m = r["active"].any(1)
    ret = r["port"][m]
    gross = ((r["w"] * np.expm1(np.nan_to_num(test["fwd"]))).sum(1) / np.maximum(r["active"].sum(1), 1))[m]
    bench = S.benchmark(test["fwd"], test["vol"], vt, r["active"])[m]
    # permutation: shift positions within each day so the test respects the intraday structure
    tdates = np.array(test["idx"].date)
    def perm_test(cost, seed):
        rng = np.random.default_rng(seed)
        obs = ST.sharpe(ST._portfolio(r["w"], np.expm1(np.nan_to_num(test["fwd"])), r["active"], cost)[m], BARS_PER_YEAR)
        null = np.empty(n_perm)
        retm = np.expm1(np.nan_to_num(test["fwd"]))
        for i in range(n_perm):
            ws = np.zeros_like(r["w"])
            for dday in test_days:
                rows = np.where(tdates == dday)[0]
                for j in range(r["w"].shape[1]):
                    rr = rows[r["active"][rows, j]]
                    if len(rr) > 10:
                        ws[rr, j] = np.roll(r["w"][rr, j], rng.integers(5, len(rr) - 5))
            null[i] = ST.sharpe(ST._portfolio(ws, retm, r["active"], cost)[m], BARS_PER_YEAR)
        return obs, null, (1 + (null >= obs).sum()) / (n_perm + 1)
    obs, null, p_net = perm_test(P.INTRADAY_COST, 0)
    _, _, p_gross = perm_test(0.0, 1)
    t, p_t, lags = ST.newey_west_t(ret)
    out = dict(
        test_days=[str(x) for x in test_days], n_bars=int(m.sum()),
        n_decisions=int(r["active"].sum()),
        total_return_net=float(np.prod(1 + ret) - 1), total_return_gross=float(np.prod(1 + gross) - 1),
        benchmark_total_return=float(np.prod(1 + bench) - 1),
        sharpe_net_ann=obs, sharpe_gross_ann=ST.sharpe(gross, BARS_PER_YEAR),
        perm_p_net=float(p_net), perm_p_gross=float(p_gross),
        perm_null_p95=float(np.percentile(null, 95)),
        nw_t=t, nw_p=p_t,
        hit=ST.hit_rate_test(r["pos"], test["fwd"], r["active"]),
        avg_abs_position=float(np.abs(r["pos"][r["active"]]).mean()),
        turnover_per_bar=float(np.abs(np.diff(r["w"], axis=0)).sum(1).mean() / len(hidden_tk)),
        alpha=ST.alpha_regression(ret, bench, BARS_PER_YEAR),
    )
    arr = dict(times=[str(x) for x in test["idx"][m]], ret=ret.tolist(), gross=gross.tolist(),
               bench=bench.tolist(), null=null.tolist(),
               happy=np.nanmean(r["happy"], 1)[m].tolist(), pain=np.nanmean(r["pain"], 1)[m].tolist())
    return out, arr


def main():
    path = os.path.join(OUT, "results_daily.json")
    if os.path.exists(path):
        with open(path) as fh:
            cfg = json.load(fh)["locked_config"]
    else:  # same values the daily run locked (printed as LOCKED in its log)
        cfg = json.loads(os.environ["LOCKED_CFG"])
    train_tk, hidden_tk = map(lambda x: list(map(str, x)), P.split_universe())
    f, fwd, vol, days = load()
    live = days[-1]
    print("intraday days:", days[0], "->", live, len(days), "| locked cfg", cfg)
    resA, arrA = run_block(cfg, f, fwd, vol, days, list(days[:-1]), [live], train_tk, hidden_tk, 5000)
    print("A live day", json.dumps({k: v for k, v in resA.items() if k not in ("hit", "alpha")}, default=float))
    resB, arrB = run_block(cfg, f, fwd, vol, days, list(days[:-10]), list(days[-10:]), train_tk, hidden_tk, 1000)
    print("B last 10 days", json.dumps({k: v for k, v in resB.items() if k not in ("hit", "alpha", "test_days")}, default=float))
    with open(os.path.join(OUT, "results_intraday.json"), "w") as fh:
        json.dump(dict(locked_config=cfg, live_day=resA, last_10_days=resB,
                       factors=features.INTRADAY_FACTORS), fh, indent=1, default=float)
    with open(os.path.join(OUT, "intraday_arrays.json"), "w") as fh:
        json.dump(dict(live_day=arrA, last_10_days=arrB), fh)


if __name__ == "__main__":
    main()
