"""Build the three results pages in docs/ from the JSON in results/.

    python -m experiments.build_reports

Each page is a template in docs/templates/ with the results data injected, wrapped as a standalone
HTML document so it can be opened locally or served with GitHub Pages.
"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, DOCS = os.path.join(ROOT, "results"), os.path.join(ROOT, "docs")

HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
        'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n'
        '</head>\n<body>\n')
TAIL = "\n</body>\n</html>\n"


def load(*p):
    return json.load(open(os.path.join(RES, *p)))


def us_blind_test():
    dl, da = load("us_blind_test", "results_daily.json"), load("us_blind_test", "daily_arrays.json")
    il, ia = load("us_blind_test", "results_intraday.json"), load("us_blind_test", "intraday_arrays.json")
    h = da["hidden_stocks"]
    ret, bench, null = np.array(h["ret"]), np.array(h["bench"]), np.array(h["null"])
    counts, edges = np.histogram(null, bins=40)
    roll = lambda x: [None if not np.isfinite(v) else round(float(v), 4)
                      for v in pd.Series(x).rolling(21, min_periods=5).mean()]
    live = ia["live_day"]
    return dict(
        locked=dl["locked_config"], grid=dl["validation_grid"], universe=dl["universe"],
        daily=dict(dates=h["dates"], fly=np.cumprod(1 + ret).round(5).tolist(), bench=np.cumprod(1 + bench).round(5).tolist(),
                   happy=roll(h["happy"]), pain=roll(h["pain"])),
        hist=dict(counts=counts.tolist(), edges=edges.round(4).tolist(), observed=dl["daily_test"]["hidden_stocks"]["sharpe_net"]),
        live=dict(times=[t[11:16] for t in live["times"]], net=np.cumprod(1 + np.array(live["ret"])).round(6).tolist(),
                  gross=np.cumprod(1 + np.array(live["gross"])).round(6).tolist(),
                  bench=np.cumprod(1 + np.array(live["bench"])).round(6).tolist()),
        tests=dict(hidden=dl["daily_test"]["hidden_stocks"], training=dl["daily_test"]["training_stocks"],
                   live=il["live_day"], last10=il["last_10_days"]),
        seeds=dl["seed_robustness"], placebo=dl["placebo"])


def niche_markets():
    R, A, T = load("niche", "results.json"), load("niche", "arrays.json"), load("niche", "tokyo_checks.json")
    out = dict(test=R["test"], holm_v=R["holm_viability"], holm_e=R["holm_edge"], lockedA=R["locked_A"],
               lockedB=R["locked_B"], valA=R["validation_A"], valB=R["validation_B"], checks=T)
    curves = {}
    for k, v in A["series"].items():
        a = v["A"]; net = np.cumprod(1 + np.array(a["net"])); mkt = np.cumprod(1 + np.array(a["mkt"]))
        curves[k] = dict(dates=a["dates"], rel=np.round(net / mkt, 4).tolist())
    out["curves"] = curves
    null = np.array(A["nulls"]["tokyo"]["A"]) * 252 / R["locked_A"]["step"]
    c, e = np.histogram(null, bins=40)
    out["hist"] = dict(counts=c.tolist(), edges=np.round(e, 4).tolist(), observed=R["test"]["tokyo"]["A"]["excess_ann"])
    return out


def v3():
    R, S = load("v3", "results.json"), load("v3", "series.json")
    out = dict(test=R["test"], h1=R["holm_h1"], h2=R["holm_h2"], locked=R["locked"], validation=R["validation"],
               school={k: v["final_exam"] for k, v in R["school"].items()})
    out["curves"] = {k: dict(dates=v["dates"], fly=np.round(np.cumprod(1 + np.array(v["net"])), 4).tolist(),
                             spx=np.round(np.cumprod(1 + np.array(v["spx"])), 4).tolist(), mode=v["mode"])
                     for k, v in S.items()}
    return out


PAGES = [("us_blind_test", "us-blind-test.html", us_blind_test),
         ("niche_markets", "niche-markets.html", niche_markets),
         ("v3", "v3.html", v3)]

INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fly-Brain Trader</title>
<style>
:root{--bg:#f6f8f7;--panel:#fff;--ink:#121815;--ink2:#4a5550;--rule:#dde3e0;--link:#1f63b8}
@media (prefers-color-scheme:dark){:root{--bg:#121614;--panel:#1a1f1c;--ink:#eef2ef;--ink2:#b8c2bc;--rule:#2c3430;--link:#7fb0ef}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:40px max(16px,env(safe-area-inset-left,0px))}
h1{font-size:34px;margin:0 0 8px}a{color:var(--link)}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:16px;margin-top:14px}
.card p{margin:4px 0 0;color:var(--ink2)}
</style></head><body><main>
<h1>Fly-Brain Trader</h1>
<p>A fruit-fly mushroom-body model that trades stocks and learns from pain, happiness and love, tested blind with hypothesis tests for skill versus luck.</p>
<div class="card"><a href="us-blind-test.html"><b>1 · US large caps blind test</b></a><p>20 hidden stocks, 2024–26, plus a bar-by-bar replay of an unseen day.</p></div>
<div class="card"><a href="niche-markets.html"><b>2 · Niche markets</b></a><p>Two strategies in Vietnam, Tokyo, Nordic and UK small caps, with realistic costs.</p></div>
<div class="card"><a href="v3.html"><b>3 · Version 3: news, maths school, love, MOEX</b></a><p>Five markets in US dollars against the S&amp;P 500.</p></div>
</main></body></html>
"""


def main():
    for key, fname, fn in PAGES:
        tpl = open(os.path.join(DOCS, "templates", f"{key}.html")).read()
        page = tpl.replace("__DATA__", json.dumps(fn(), allow_nan=False))
        open(os.path.join(DOCS, fname), "w").write(HEAD + page + TAIL)
        print("wrote docs/" + fname)
    open(os.path.join(DOCS, "index.html"), "w").write(INDEX)
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
