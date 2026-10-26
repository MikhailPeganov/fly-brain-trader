"""EXPLORATORY follow-ups on the one promising result (Tokyo, strategy A). Not pre-registered.
1. Seed robustness: 10 differently wired brains, same locked settings.
2. Simple baselines: hold the top 20 by a single quant factor (direction chosen on 2016-23 data),
   same weekly schedule, hysteresis and costs. If one of these matches the fly, the fly isn't
   adding anything beyond a known factor.
3. What the fly tilted toward: average factor score of its holdings vs the tradeable market.
"""
import json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from niche import strategies as S
from experiments import niche_markets as R

P = S.prepare("tokyo")
sc = S.ts_scale(P, R.PRE_END)
cfg = json.load(open("results/niche/results.json"))["locked_A"]
kw = dict(step=cfg["step"], keep_mult=cfg["keep_mult"])
out = {}

# 1. seeds
seeds = []
for seed in range(10):
    b = R.pretrain_A(P, cfg["lr"], seed, sc, R.VAL[1], **kw)
    rows, _ = S.run_picker(P, b, *R.TEST, sc, record=True, **kw)
    r, _ = R.picker_tests(P, rows, n_perm=1000, step=cfg["step"])
    seeds.append(dict(seed=seed, excess_ann=r["excess_ann"], p_edge=r["p_edge"], p_viability=r["p_viability"]))
    print("seed", seeds[-1], flush=True)
out["seeds"] = seeds


# 2. single-factor baselines
class FactorRule:
    def __init__(self, col, sign): self.col, self.sign = col, sign
    def kenyon(self, X): return X
    def decide(self, X): return None, self.sign * X[:, self.col]
    def learn(self, *a, **k): pass


base = []
for i, name in enumerate(S.FEATURES_A):
    if name == "mkt_1m":
        continue
    tr = {}
    for sign in (1, -1):
        rows, _ = S.run_picker(P, FactorRule(i, sign), R.first_date(P), R.VAL[1], sc, **kw)
        df = pd.DataFrame(rows); tr[sign] = (df.net - df.mkt).mean()
    sign = max(tr, key=tr.get)
    rows, _ = S.run_picker(P, FactorRule(i, sign), *R.TEST, sc, **kw)
    df = pd.DataFrame(rows)
    base.append(dict(factor=name, direction="high" if sign > 0 else "low",
                     excess_ann=float((df.net - df.mkt).mean() * 252 / cfg["step"])))
    print("baseline", base[-1], flush=True)
out["baselines"] = base

# 3. tilts of the seed-0 fly
b = R.pretrain_A(P, cfg["lr"], 0, sc, R.VAL[1], **kw)
rows, _ = S.run_picker(P, b, *R.TEST, sc, record=True, **kw)
t_of = {d: i for i, d in enumerate(P["dates"])}
tilt = np.zeros(len(S.FEATURES_A)); n = 0
for r in rows:
    t = t_of[r["date"]]; idx = np.array(r["elig"])
    X = S.snapshot(P, S.FEATURES_A, t, idx, ts_scale=sc)
    held = np.isin(idx, r["held"])
    if held.any():
        tilt += X[held].mean(0) - X.mean(0); n += 1
out["tilts"] = {k: float(v / n) for k, v in zip(S.FEATURES_A, tilt)}
print("tilts", {k: round(v, 2) for k, v in out["tilts"].items()})


# 4. a hand-built mix of the fly's three main tilts: low volatility, low MAX, near 52-week high
class Composite:
    def kenyon(self, X): return X
    def decide(self, X):
        F = S.FEATURES_A
        return None, -X[:, F.index("vol_60")] - X[:, F.index("max_21")] + X[:, F.index("high_52w")]
    def learn(self, *a, **k): pass


rows, _ = S.run_picker(P, Composite(), *R.TEST, sc, record=True, **kw)
out["textbook_composite"], _ = R.picker_tests(P, rows, n_perm=1000, step=cfg["step"])
print("textbook composite", {k: round(v, 4) for k, v in out["textbook_composite"].items()})
json.dump(out, open("results/niche/tokyo_checks.json", "w"), indent=1)
