"""Market data download (Yahoo Finance chart API) with on-disk caching."""
import json, os, time, datetime as dt
import urllib.request
import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "MS", "C", "XOM", "CVX", "COP", "SLB", "JNJ", "PFE", "MRK", "ABBV",
    "UNH", "LLY", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE",
    "DIS", "CMCSA", "VZ", "T", "INTC", "CSCO", "ORCL", "IBM", "ADBE", "CRM",
    "QCOM", "TXN", "AMD", "CAT", "DE", "BA", "GE", "HON", "MMM", "UPS",
    "LMT", "RTX", "NEE", "DUK", "SO", "AMT", "SPG", "LIN", "NEM", "F",
]
MARKET = "SPY"


def _fetch(url, retries=6):
    import subprocess
    for i in range(retries):
        out = subprocess.run(["curl", "-s", "-m", "30", "-A", "Mozilla/5.0", "-w", "\n%{http_code}", url],
                             capture_output=True, text=True)
        body, _, code = out.stdout.rpartition("\n")
        if code == "200":
            return json.loads(body)
        if i == retries - 1:
            raise RuntimeError(f"HTTP {code} for {url}")
        time.sleep(2 + 3 * i)


def _parse(js, intraday=False):
    res = js["chart"]["result"][0]
    ts = res.get("timestamp")
    if ts is None:
        return pd.DataFrame()
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]},
                      index=pd.to_datetime(ts, unit="s", utc=True))
    if not intraday and "adjclose" in res["indicators"]:
        adj = np.array(res["indicators"]["adjclose"][0]["adjclose"], dtype=float)
        ratio = adj / df["close"].values
        for k in ["open", "high", "low"]:
            df[k] = df[k] * ratio
        df["close"] = adj
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    df.index = df.index.tz_convert(tz)
    if not intraday:
        df.index = pd.DatetimeIndex(df.index.date)
    return df.dropna(subset=["close"])


def daily(symbol, start="2013-01-01", end="2026-10-24", refresh=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"daily_{symbol}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    p1 = int(dt.datetime.fromisoformat(start).timestamp())
    p2 = int(dt.datetime.fromisoformat(end).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    df = _parse(_fetch(url))
    df.to_csv(path)
    time.sleep(0.4)
    return df


def intraday(symbol, interval="5m", rng="60d", refresh=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"intraday_{interval}_{symbol}.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        return df
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range={rng}&interval={interval}")
    df = _parse(_fetch(url), intraday=True)
    df.to_csv(path)
    time.sleep(0.4)
    return df


def panel(symbols, kind="daily"):
    """Return dict field -> DataFrame (dates x symbols)."""
    frames = {s: (daily(s) if kind == "daily" else intraday(s)) for s in symbols}
    fields = ["open", "high", "low", "close", "volume"]
    out = {f: pd.DataFrame({s: frames[s][f] for s in symbols}) for f in fields}
    return out


if __name__ == "__main__":
    for s in UNIVERSE + [MARKET]:
        d = daily(s)
        i = intraday(s)
        print(f"{s:6s} daily {d.index[0].date()}..{d.index[-1].date()} ({len(d)})  "
              f"5m {len(i)} bars, last {i.index[-1]}")
