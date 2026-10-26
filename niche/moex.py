"""MOEX (Moscow Exchange) data for RESEARCH BACKTESTS ONLY.

The UK designated the Moscow Exchange under its Russia sanctions in June 2024, so a UK person
cannot trade there. This module only downloads public historical prices to test the model.

Builds the same cached files as niche/data.py so strategies.prepare("moex") works:
  * IMOEX constituents (current), daily TQBR history 2015-2026 from the ISS API
  * renamed companies stitched to their predecessors (T<-TCSG, YDEX<-YNDX, X5<-FIVE, HEAD<-HHRU)
  * total-return adjustment: dividends (registry dates from dohod.ru) added back on the ex-date
    (registry date minus 2 trading days before 31 Jul 2023, minus 1 after the move to T+1)
  * splits / share exchanges: day moves above 3x are treated as corporate actions, not returns
  * live bid/offer snapshot for spreads; RUB->USD from Yahoo
"""
import json, os, re, subprocess, time
import numpy as np
import pandas as pd

from .data import ROOT, _fx

ISS = "https://iss.moex.com/iss"
PREDECESSORS = {"T": ["TCSG"], "YDEX": ["YNDX"], "X5": ["FIVE"], "HEAD": ["HHRU"]}


def _get(url, tries=5):
    for i in range(tries):
        r = subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", url], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
        time.sleep(2 + 3 * i)
    raise RuntimeError(url)


def constituents():
    d = json.loads(_get(f"{ISS}/statistics/engines/stock/markets/index/analytics/IMOEX.json?limit=100&iss.meta=off"))
    a = d["analytics"]; i = a["columns"].index("ticker")
    return sorted({row[i] for row in a["data"]})


def history(secid, start="2015-01-01"):
    rows, cols, off = [], None, 0
    while True:
        d = json.loads(_get(f"{ISS}/history/engines/stock/markets/shares/boards/TQBR/securities/{secid}.json"
                            f"?from={start}&start={off}&iss.meta=off"))
        h = d["history"]; cols = h["columns"]; rows += h["data"]
        cur = d["history.cursor"]["data"][0]
        off = cur[0] + cur[2]
        if off >= cur[1] or not h["data"]:
            break
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["TRADEDATE"])
    close = df["CLOSE"].fillna(df["LEGALCLOSEPRICE"])
    out = pd.DataFrame({"Open": df["OPEN"], "High": df["HIGH"], "Low": df["LOW"], "Close": close,
                        "Value": df["VALUE"]}, ).set_index(df["date"])
    out = out[~out.index.duplicated()]
    for k in ("Open", "High", "Low"):
        out[k] = out[k].fillna(out["Close"])
    return out.dropna(subset=["Close"])


def dividends(secid):
    try:
        h = _get(f"https://www.dohod.ru/ik/analytics/dividend/{secid.lower()}")
    except Exception:
        return []
    out = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        cells = [re.sub("<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if len(cells) >= 4 and re.fullmatch(r"\d\d\.\d\d\.\d{4}", cells[1]) and "прогноз" not in r:
            try:
                out.append((pd.to_datetime(cells[1], dayfirst=True), float(cells[3].replace(",", "."))))
            except ValueError:
                pass
    return out


def stitched(secid):
    parts = [history(p) for p in PREDECESSORS.get(secid, [])] + [history(secid)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(), []
    # chain on returns so the price level is continuous across a ticker change
    df = parts[0]
    for p in parts[1:]:
        p = p[p.index > df.index[-1]]
        if len(p):
            df = pd.concat([df, p])
    return df, sum((dividends(s) for s in PREDECESSORS.get(secid, []) + [secid]), [])


def total_return(df, divs):
    c = df["Close"].astype(float)
    r = c / c.shift(1)
    dates = df.index
    for reg, amt in divs:
        if reg < dates[0] or reg > dates[-1]:
            continue
        pos = dates.searchsorted(reg)                     # registry date (or next trading day)
        lag = 1 if reg >= pd.Timestamp("2023-07-31") else 2
        ex = pos - lag + 1 if pos < len(dates) and dates[pos] == reg else pos - lag + 1
        if 1 <= ex < len(dates):
            r.iloc[ex] = (c.iloc[ex] + amt) / c.iloc[ex - 1]
    # corporate actions: anything beyond a 3x move in a day is a split/share exchange, not a return
    r[(r > 3) | (r < 1 / 3)] = np.nan
    tr = r.fillna(1.0).cumprod() * c.iloc[0]
    ratio = tr / c
    out = pd.DataFrame({k: df[k] * ratio for k in ("Open", "High", "Low")})
    out["Close"] = tr
    out["Volume"] = df["Value"] / tr                       # so Close * Volume = RUB turnover
    bad = r.isna() & (r.index != r.index[0])
    return out, int(bad.sum())


def spreads(secids):
    d = json.loads(_get(f"{ISS}/engines/stock/markets/shares/boards/TQBR/securities.json?iss.only=marketdata"
                        f"&marketdata.columns=SECID,BID,OFFER,LAST&iss.meta=off"))
    m = {row[0]: row for row in d["marketdata"]["data"]}
    out = {}
    for s in secids:
        if s in m and m[s][1] and m[s][2] and m[s][2] > m[s][1] > 0:
            out[s] = (m[s][2] - m[s][1]) / ((m[s][2] + m[s][1]) / 2)
    med = float(np.median(list(out.values()))) if out else 0.001
    return dict(spread={s: out.get(s, med) for s in secids}, n_quoted=len(out), median=med)


def build():
    os.makedirs(ROOT, exist_ok=True)
    secs = constituents()
    rub = _fx("RUB")
    frames, info, log = {}, [], {}
    for s in secs:
        df, divs = stitched(s)
        if len(df) < 250:
            log[s] = "too short"; continue
        adj, n_ca = total_return(df, divs)
        frames[s] = adj
        info.append(dict(symbol=s, currency="RUB", fx_usd=rub))
        log[s] = dict(days=len(adj), first=str(adj.index[0].date()), dividends=len(divs), corp_actions=n_ca)
        print(s, log[s], flush=True)
        time.sleep(0.3)
    idx = sorted(set().union(*[f.index for f in frames.values()]))
    out = {k: pd.DataFrame({s: frames[s][k] for s in frames}).reindex(idx) for k in ("Open", "High", "Low", "Close", "Volume")}
    pd.to_pickle(out, os.path.join(ROOT, "prices_moex.pkl"))
    json.dump(info, open(os.path.join(ROOT, "universe_moex.json"), "w"))
    json.dump(spreads([x["symbol"] for x in info]), open(os.path.join(ROOT, "spreads_moex.json"), "w"))
    json.dump(log, open(os.path.join(ROOT, "moex_build_log.json"), "w"), indent=1)


if __name__ == "__main__":
    build()
