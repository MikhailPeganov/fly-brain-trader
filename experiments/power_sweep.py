"""Power check: does the primary test detect a signal that really is there?

The placebo run in `experiments/us_blind_test.py` fixes the FALSE-POSITIVE rate (how often the
permutation test fires on markets with no structure). This fixes the other half: the
FALSE-NEGATIVE rate. Without it, "p = 0.147, no evidence of skill" is uninterpretable,
because a test that can never reject H0 would say exactly the same thing.

Method: rebuild each stock's price series so its log returns follow an AR(1) with a
known coefficient -k (a planted 1-day reversal, which `rev_1d` can in principle learn):

    lr2[t] = lr[t] - k * lr2[t-1]

so E[lr2[t] | lr2[t-1]] = -k * lr2[t-1] exactly. k = 0 reproduces a placebo-like market
(the sign flip leaves no learnable structure), and larger k plants a stronger edge.
The whole pipeline then runs unchanged — pre-train on training stocks, blind test on the
hidden 20, circular-shift permutation test — and we record how often it rejects at 0.05.

Output: results/us_blind_test/results_power.json, a power curve of detection rate vs planted strength.
Read it as: the primary test can detect a reversal edge of size >= k* with power >= 80%;
the observed data is consistent with no edge of that size or larger.
"""
import json, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
from flybrain import pipeline as P, brain as B, sim as S, stats as ST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
OUT = os.path.join(ROOT, "results", "us_blind_test")
os.makedirs(OUT, exist_ok=True)

STRENGTHS = [0.00, 0.02, 0.05, 0.10, 0.20]   # AR(1) reversal coefficient k
REPS = 5                                      # independent markets per strength
N_PERM = 1000
ALPHA = 0.05


def planted(k, seed):
    """Return a transform(panel) -> panel that plants an AR(1) reversal of strength k."""
    def tf(p, k=k, seed=seed):
        rng = np.random.default_rng(seed)
        c = p["close"]
        lr = np.log(c).diff()
        # randomise the innovation order so reps differ but the return distribution matches
        vals = lr.values.copy()
        for j in range(vals.shape[1]):
            col = vals[:, j]
            fin = np.isfinite(col)
            idx = np.where(fin)[0]
            col[idx] = col[rng.permutation(idx)]
            # impose the AR(1): lr2[t] = eps[t] - k * lr2[t-1]
            prev = 0.0
            for i in idx:
                col[i] = col[i] - k * prev
                prev = col[i]
            vals[:, j] = col
        lr2 = lr.copy()
        lr2.iloc[:, :] = vals
        c2 = np.exp(lr2.fillna(0).cumsum()) * c.iloc[0]
        c2 = c2.where(c.notna())
        ro, rh, rl = p["open"] / c, p["high"] / c, p["low"] / c
        o2 = c2 * ro
        return dict(close=c2, open=o2,
                    high=np.maximum(c2 * rh, np.maximum(o2, c2)),
                    low=np.minimum(c2 * rl, np.minimum(o2, c2)),
                    volume=p["volume"])
    return tf


def main():
    t0 = time.time()
    cfg = json.load(open(os.path.join(OUT, "results_daily.json")))["locked_config"]
    print("locked config (unchanged):", cfg, flush=True)
    train_tk, hidden_tk = P.split_universe()
    curve = []
    for k in STRENGTHS:
        reps = []
        for r in range(REPS):
            d = P.daily_tensors(transform=planted(k, seed=50_000 + 100 * int(k * 100) + r))
            full = P.select(d, (P.TRAIN[0], P.VAL[1]), train_tk)
            b = B.MushroomBodyTrader(full["X"].shape[2], seed=0, **cfg)
            for _ in range(2):
                S.run(b, full["X"], full["fwd"], full["vol"], P.DAILY_COST,
                      P.DAILY_VOL_TARGET, explore=0.5)
            te = P.select(d, P.TEST, hidden_tk)
            res = S.run(b, te["X"], te["fwd"], te["vol"], P.DAILY_COST, P.DAILY_VOL_TARGET)
            pr = ST.circular_shift_test(res["w"], te["fwd"], res["active"], P.DAILY_COST,
                                        n_perm=N_PERM, periods=252)
            reps.append(dict(rep=r, sharpe_net=pr["observed"], perm_p=pr["p"]))
            print(f"  k={k:.2f} rep={r} sharpe={pr['observed']:+.2f} p={pr['p']:.3f}", flush=True)
        power = float(np.mean([x["perm_p"] < ALPHA for x in reps]))
        curve.append(dict(strength=k, power=power, reps=reps,
                          mean_sharpe=float(np.mean([x["sharpe_net"] for x in reps]))))
        print(f"k={k:.2f} -> power {power:.0%}", flush=True)

    detectable = next((c["strength"] for c in curve if c["power"] >= 0.8), None)
    out = dict(alpha=ALPHA, n_perm=N_PERM, reps_per_strength=REPS, locked_config=cfg,
               curve=curve, min_detectable_strength_at_80pct_power=detectable)
    with open(os.path.join(OUT, "results_power.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("min detectable k at 80% power:", detectable)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
