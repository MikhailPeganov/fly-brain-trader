"""Full experiment: tune on validation -> lock -> blind tests -> skill-vs-luck tests -> placebo.

Pre-registered before any test data was touched:
  PRIMARY hypothesis test: circular-shift permutation test of the net-of-cost Sharpe ratio
  on the 20 hidden stocks, 2024-01-02 to 2026-10-22, alpha = 0.05, brain seed 0.
  Everything else is secondary / diagnostic.
"""
import json, os, time, copy, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from flybrain import pipeline as P, brain as B, sim as S, stats as ST, data, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
OUT = os.path.join(ROOT, "results", "us_blind_test")
os.makedirs(OUT, exist_ok=True)
GRID = [dict(lr=lr, threshold=th) for lr in (0.003, 0.01, 0.03) for th in (0.5, 1.0, 1.5)]
EPOCHS, EXPLORE = 2, 0.5
log = {}


def pretrain(cfg, d, seed, epochs=EPOCHS, cost=P.DAILY_COST, vt=P.DAILY_VOL_TARGET):
    b = B.MushroomBodyTrader(d["X"].shape[2], seed=seed, **cfg)
    for _ in range(epochs):
        S.run(b, d["X"], d["fwd"], d["vol"], cost, vt, explore=EXPLORE)
    return b


def evaluate(r, d, cost, periods, bench=None, n_perm=5000, boot=True):
    ret = r["port"][r["active"].any(1)]
    gross = (r["w"] * np.expm1(np.nan_to_num(d["fwd"]))).sum(1) / np.maximum(r["active"].sum(1), 1)
    gross = gross[r["active"].any(1)]
    t, p_t, lags = ST.newey_west_t(ret)
    perm = ST.circular_shift_test(r["w"], d["fwd"], r["active"], cost, n_perm=n_perm, periods=periods)
    perm_g = ST.circular_shift_test(r["w"], d["fwd"], r["active"], 0.0, n_perm=min(n_perm, 2000), periods=periods, seed=1)
    out = dict(
        n_periods=int(len(ret)),
        sharpe_net=ST.sharpe(ret, periods), sharpe_gross=ST.sharpe(gross, periods),
        ann_return_net=float(ret.mean() * periods), ann_vol=float(ret.std() * np.sqrt(periods)),
        total_return_net=float(np.prod(1 + ret) - 1),
        avg_abs_position=float(np.abs(r["pos"][r["active"]]).mean()),
        frac_long=float((r["pos"][r["active"]] > 0).mean()), frac_short=float((r["pos"][r["active"]] < 0).mean()),
        nw_t=t, nw_p=p_t, nw_lags=lags,
        perm_p_net=perm["p"], perm_null_mean=perm["null_mean"], perm_null_p95=perm["null_p95"],
        perm_p_gross=perm_g["p"],
        hit=ST.hit_rate_test(r["pos"], d["fwd"], r["active"]),
    )
    if boot:
        out["bootstrap_p"] = ST.stationary_bootstrap_p(ret, n_boot=3000)
    if bench is not None:
        out["alpha"] = ST.alpha_regression(ret, bench[r["active"].any(1)], periods)
    return out, perm["null"], ret


def main():
    t0 = time.time()
    d = P.daily_tensors()
    train_tk, hidden_tk = P.split_universe()
    log["universe"] = dict(train=list(map(str, train_tk)), hidden=list(map(str, hidden_tk)))
    log["factors"] = features.DAILY_FACTORS

    # ---------------- 1. tune on validation (training stocks only) ---------------
    tr = P.select(d, P.TRAIN, train_tk); va = P.select(d, P.VAL, train_tk)
    grid_res = []
    for cfg in GRID:
        srs = []
        for seed in (101, 102, 103):            # tuning seeds differ from the test seed
            b = pretrain(cfg, tr, seed)
            rv = S.run(b, va["X"], va["fwd"], va["vol"], P.DAILY_COST, P.DAILY_VOL_TARGET)
            srs.append(ST.sharpe(rv["port"], 252))
        grid_res.append(dict(**cfg, val_sharpe_net=float(np.mean(srs)), seeds=srs))
        print("grid", cfg, round(np.mean(srs), 3), flush=True)
    best = max(grid_res, key=lambda g: g["val_sharpe_net"])
    cfg = dict(lr=best["lr"], threshold=best["threshold"])
    log["validation_grid"] = grid_res; log["locked_config"] = cfg
    log["locked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("LOCKED", cfg, flush=True)

    # ---------------- 2. final pre-training on train+val, training stocks -------
    full = P.select(d, (P.TRAIN[0], P.VAL[1]), train_tk)
    brain0 = pretrain(cfg, full, seed=0)

    # ---------------- 3. blind test on hidden stocks -----------------------------
    te_h = P.select(d, P.TEST, hidden_tk)
    te_t = P.select(d, P.TEST, train_tk)
    res = {}
    arrays = {}
    for name, te in [("hidden_stocks", te_h), ("training_stocks", te_t)]:
        b = copy.deepcopy(brain0)
        r = S.run(b, te["X"], te["fwd"], te["vol"], P.DAILY_COST, P.DAILY_VOL_TARGET, learn=True)
        bench = S.benchmark(te["fwd"], te["vol"], P.DAILY_VOL_TARGET, r["active"])
        ev, null, ret = evaluate(r, te, P.DAILY_COST, 252, bench=bench,
                                 n_perm=5000 if name == "hidden_stocks" else 2000)
        ev["benchmark_sharpe"] = ST.sharpe(bench[r["active"].any(1)], 252)
        res[name] = ev
        m = r["active"].any(1)
        arrays[name] = dict(dates=[str(x.date()) for x in te["dates"][m]], ret=ret.tolist(),
                            bench=bench[m].tolist(), null=null.tolist(),
                            happy=np.nanmean(r["happy"], 1)[m].tolist(),
                            pain=np.nanmean(r["pain"], 1)[m].tolist())
        print(name, json.dumps({k: v for k, v in ev.items() if k != "hit"}, default=float)[:400], flush=True)
    # last day of the daily test = the most recent session the model traded
    log["daily_test"] = res

    # robustness: 10 different random brains (connectivity seeds)
    rob = []
    for seed in range(10):
        b = pretrain(cfg, full, seed=seed)
        r = S.run(b, te_h["X"], te_h["fwd"], te_h["vol"], P.DAILY_COST, P.DAILY_VOL_TARGET)
        pr = ST.circular_shift_test(r["w"], te_h["fwd"], r["active"], P.DAILY_COST, n_perm=1000, periods=252)
        rob.append(dict(seed=seed, sharpe_net=pr["observed"], perm_p=pr["p"]))
    log["seed_robustness"] = rob
    print("seeds", [(x["seed"], round(x["sharpe_net"], 2), round(x["perm_p"], 3)) for x in rob], flush=True)

    # ---------------- 4. placebo: same pipeline on returns with random signs -----
    plac = []
    for k in range(40):
        rng = np.random.default_rng(10_000 + k)

        def flip(p, rng=rng):
            c = p["close"]; lr_ = np.log(c).diff()
            sgn = rng.choice([-1.0, 1.0], size=lr_.shape)
            c2 = np.exp((lr_ * sgn).fillna(0).cumsum()) * c.iloc[0]
            c2 = c2.where(c.notna())
            ro, rh, rl = p["open"] / c, p["high"] / c, p["low"] / c
            o2 = c2 * ro
            return dict(close=c2, open=o2, high=np.maximum(c2 * rh, np.maximum(o2, c2)),
                        low=np.minimum(c2 * rl, np.minimum(o2, c2)), volume=p["volume"])
        dp = P.daily_tensors(transform=flip)
        b = pretrain(cfg, P.select(dp, (P.TRAIN[0], P.VAL[1]), train_tk), seed=0)
        th = P.select(dp, P.TEST, hidden_tk)
        r = S.run(b, th["X"], th["fwd"], th["vol"], P.DAILY_COST, P.DAILY_VOL_TARGET)
        pr = ST.circular_shift_test(r["w"], th["fwd"], r["active"], P.DAILY_COST, n_perm=1000, periods=252)
        plac.append(dict(rep=k, sharpe_net=pr["observed"], perm_p=pr["p"]))
    log["placebo"] = dict(reps=plac, false_positive_rate=float(np.mean([x["perm_p"] < 0.05 for x in plac])))
    print("placebo FPR", log["placebo"]["false_positive_rate"], flush=True)

    with open(os.path.join(OUT, "daily_arrays.json"), "w") as f:
        json.dump(arrays, f)
    with open(os.path.join(OUT, "results_daily.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
