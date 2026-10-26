"""Universes, 10-year daily prices and snapshot spreads for the four tradeable niche markets.
Everything is cached under data/niche/ so reruns don't hit Yahoo again."""
import json, os, time
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as Q

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "niche")
SEED = 11

MARKETS = {
    # key: (label, screener queries, local market-cap ceiling, sample size)
    "vietnam": ("Vietnam HOSE", [Q("eq", ["exchange", "VSE"])], None, 400),
    "tokyo": ("Tokyo small caps", [Q("eq", ["exchange", "JPX"])], 30e9, 500),
    "nordic": ("First North / Nordic small caps", [Q("eq", ["exchange", "STO"]), Q("eq", ["exchange", "HEL"]),
                                                    Q("eq", ["exchange", "CPH"])], "EUR250m", 400),
    "aim": ("AIM / UK small caps", [Q("eq", ["exchange", "LSE"])], 250e6, 400),
}
# trading frictions per side, on top of the quoted half-spread and price impact
COMMISSION = 0.0010                       # 0.10% per side (typical online broker for these markets)
SELL_TAX = {"vietnam": 0.0010}            # Vietnam: 0.1% tax on sale proceeds; AIM shares are stamp-duty exempt


def _retry(fn, tries=8):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(15 + 10 * i)


def _fx(ccy):
    base = "GBP" if ccy == "GBp" else ccy
    if base == "USD":
        r = 1.0
    else:
        s = _retry(lambda: yf.download(f"{base}USD=X", period="1mo", progress=False, auto_adjust=False)["Close"])
        r = float(np.asarray(s).ravel()[-1])
    return r / 100 if ccy == "GBp" else r


def universe(key):
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f"universe_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    label, queries, cap, n = MARKETS[key]
    allq = []
    for q in queries:
        off = 0
        while True:
            r = _retry(lambda: yf.screen(q, size=250, offset=off, sortField="intradaymarketcap", sortAsc=True))
            qs = r.get("quotes", []); allq += qs; off += len(qs)
            if not qs or off >= r.get("total", 0):
                break
    allq = [x for x in allq if x.get("quoteType", "EQUITY") == "EQUITY" and x.get("marketCap")]
    fx = {c: _fx(c) for c in sorted({x["currency"] for x in allq})}
    if cap == "EUR250m":
        eur = _fx("EUR")
        allq = [x for x in allq if x["marketCap"] * fx[x["currency"]] / eur < 250e6]
    elif cap is not None:
        allq = [x for x in allq if x["marketCap"] < cap]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(allq), size=min(n, len(allq)), replace=False))
    u = [dict(symbol=allq[i]["symbol"], currency=allq[i]["currency"], fx_usd=fx[allq[i]["currency"]]) for i in pick]
    json.dump(u, open(path, "w"))
    return u


def prices(key):
    path = os.path.join(ROOT, f"prices_{key}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    syms = [x["symbol"] for x in universe(key)]
    frames = []
    for i in range(0, len(syms), 50):
        d = _retry(lambda: yf.download(syms[i:i + 50], start="2015-01-01", end="2026-10-24", interval="1d",
                                       group_by="ticker", auto_adjust=True, progress=False, threads=False))
        frames.append(d)
        time.sleep(1)
    d = pd.concat(frames, axis=1)
    out = {f: pd.DataFrame({s: d[s][f] for s in syms if s in d.columns.get_level_values(0)})
           for f in ["Open", "High", "Low", "Close", "Volume"]}
    pd.to_pickle(out, path)
    return out


def spreads(key):
    """Snapshot quoted spread per stock (26 Oct 2026). Missing -> market median."""
    path = os.path.join(ROOT, f"spreads_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    from yfinance.data import YfData
    y, out = YfData(), {}
    syms = [x["symbol"] for x in universe(key)]
    for i in range(0, len(syms), 40):
        try:
            j = _retry(lambda: y.get_raw_json("https://query1.finance.yahoo.com/v7/finance/quote",
                                              params={"symbols": ",".join(syms[i:i + 40])}), tries=4)
        except Exception:
            continue
        for q in j["quoteResponse"]["result"]:
            b, a = q.get("bid"), q.get("ask")
            if b and a and a > b > 0 and (a - b) / ((a + b) / 2) < 0.3:
                out[q["symbol"]] = (a - b) / ((a + b) / 2)
    med = float(np.median(list(out.values()))) if out else 0.01
    res = dict(spread={s: out.get(s, med) for s in syms}, n_quoted=len(out), median=med)
    json.dump(res, open(path, "w"))
    return res


if __name__ == "__main__":
    for k in MARKETS:
        u = universe(k); p = prices(k); s = spreads(k)
        c = p["Close"]
        print(f"{k:8s} universe {len(u)}  with prices {c.notna().any().sum()}  "
              f"dates {c.index[0].date()}..{c.index[-1].date()}  quoted spreads {s['n_quoted']} "
              f"(median {s['median']*1e4:.0f} bps)", flush=True)
