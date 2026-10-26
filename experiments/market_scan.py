"""Scan 10 under-followed markets for volatility vs trading cost (no edge-hunting).

For each market: random sample of up to 60 stocks from a small-cap proxy universe,
last 12 months of daily OHLCV, then per stock:
  vol_ann      annualised close-to-close volatility
  spread       live quoted bid-ask spread (one snapshot, 26 Oct 2026) where Yahoo has quotes
  impact_10k   price impact of a $10,000 trade from the Amihud (2002) illiquidity ratio
  zero_days    share of days with zero return (illiquidity, Lesmond et al. 1999)
  adv_usd      median daily traded value in USD
  ratio        daily volatility / round-trip cost (spread + 2x impact) -> moves per unit of cost
"""
import json, re, subprocess, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as Q

warnings.filterwarnings("ignore")
N_SAMPLE, SEED = 60, 7

# (name, screener query or ticker list, small-cap market-cap ceiling in local currency, note)
MARKETS = [
    ("Tokyo Standard/Growth (proxy: JPX < ¥30bn)", [Q("eq", ["exchange", "JPX"])], 30e9),
    ("Nasdaq First North (proxy: Nordic < €250m)", [Q("eq", ["exchange", "STO"]), Q("eq", ["exchange", "HEL"]), Q("eq", ["exchange", "CPH"])], None),
    ("London AIM (proxy: LSE < £250m)", [Q("eq", ["exchange", "LSE"])], 250e6),
    ("Warsaw NewConnect (proxy: WSE < PLN 300m)", [Q("eq", ["exchange", "WSE"])], 300e6),
    ("TSX Venture", [Q("eq", ["exchange", "VAN"])], None),
    ("ASX small caps (< A$300m)", [Q("eq", ["exchange", "ASX"])], 300e6),
    ("Vietnam HOSE", [Q("eq", ["exchange", "VSE"])], None),
    ("Pakistan KSE-100", "PK", None),
    ("Egypt EGX (EGX 30/70-type names)", "EG", None),
    ("Bucharest BVB", [Q("eq", ["exchange", "BVB"])], None),
]
NORDIC_CAP_EUR = 250e6


def _retry(fn, tries=8):
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(15 + 10 * i)   # Yahoo rate limits: back off


def screen_all(queries):
    out = []
    for q in queries:
        off = 0
        while True:
            r = _retry(lambda: yf.screen(q, size=250, offset=off, sortField="intradaymarketcap", sortAsc=True))
            qs = r.get("quotes", [])
            out += qs
            off += len(qs)
            if not qs or off >= r.get("total", 0):
                break
    return out


def pk_tickers():
    h = subprocess.run(["curl", "-s", "-m", "20", "-A", "Mozilla/5.0", "https://en.wikipedia.org/wiki/KSE_100_Index"],
                       capture_output=True, text=True).stdout
    syms = sorted(set(re.findall(r'psx\.com\.pk/[^"]*?/([A-Z]{2,8})"', h)))
    return [dict(symbol=s + ".KA", currency="PKR", marketCap=None) for s in syms]


def fx_to_usd(ccys):
    rates = {"USD": 1.0}
    for c in ccys:
        base = "GBP" if c == "GBp" else c
        if base in rates:
            continue
        s = yf.download(f"{base}USD=X", period="1y", progress=False, auto_adjust=False)["Close"]
        rates[base] = float(np.asarray(s).ravel()[-1])
    # LSE quotes are in pence (GBp), so 1 GBp = GBPUSD / 100
    return {c: rates["GBP"] / 100 if c == "GBp" else rates[c] for c in ccys}


def abdi_ranaldo(df):
    c, h, l = np.log(df["Close"]), np.log(df["High"]), np.log(df["Low"])
    eta = (h + l) / 2
    s2 = 4 * (c - eta) * (c - eta.shift(-1))
    m = s2.groupby(df.index.to_period("M")).mean().clip(lower=0)
    return float(np.sqrt(m).mean())


def live_quotes(symbols):
    """Snapshot quoted spread (ask-bid)/mid via Yahoo's quote endpoint, in batches."""
    from yfinance.data import YfData
    y, out = YfData(), {}
    for i in range(0, len(symbols), 40):
        try:
            j = _retry(lambda: y.get_raw_json("https://query1.finance.yahoo.com/v7/finance/quote",
                                              params={"symbols": ",".join(symbols[i:i + 40])}), tries=3)
        except Exception:
            continue          # no quotes for this batch: spread left blank, impact measure still used
        for q in j["quoteResponse"]["result"]:
            b, a = q.get("bid"), q.get("ask")
            if b and a and a > b > 0 and (a - b) / ((a + b) / 2) < 0.5:
                out[q["symbol"]] = (a - b) / ((a + b) / 2)
    return out


def stock_stats(df, fx):
    df = df.dropna(subset=["Close", "High", "Low"])
    df = df[(df["High"] > 0) & (df["Low"] > 0)]
    if len(df) < 150:
        return None
    r = np.log(df["Close"]).diff().dropna()
    return dict(
        vol_ann=float(r.std() * np.sqrt(252)),
        # Amihud: average |return| per dollar traded, over days with trading (standard definition)
        amihud=float(((r.abs()) / (df["Close"] * df["Volume"] * fx).reindex(r.index).replace(0, np.nan)).mean()),
        zero_days=float((r == 0).mean()),
        adv_usd=float((df["Close"] * df["Volume"]).median() * fx),
    )


def main():
    import os
    os.makedirs("results/market_scan/cache", exist_ok=True)
    rows, per_stock = [], {}
    for k, (name, src, cap) in enumerate(MARKETS):
        cache = f"results/market_scan/cache/{k}.json"
        if os.path.exists(cache):
            c = json.load(open(cache)); rows.append(c["row"]); per_stock[name] = c["stocks"]; continue
        rng = np.random.default_rng(SEED + k)
        if src == "PK":
            univ = pk_tickers()
        elif src == "EG":
            # Yahoo's screener lists Egyptian stocks under ISIN codes with no history,
            # so use common ticker codes of EGX 30 / EGX 70 constituents instead
            eg = ("COMI EAST HRHO TMGH SWDY ETEL ABUK MFPC ORAS FWRY JUFO ESRS AMOC HELI PHDC OCDI CIEB ADIB "
                  "EFIH EMFD ORWE CLHO BTFH SKPC GBCO CCAP ISPH EFID MASR RAYA ARCC EGAL OIH ALCN EGTS PHAR "
                  "MNHD EFIC POUL SUGR ACGC KABO DSCW ATQA ELEC ZMID PRDC CSAG EKHOA MCQE RMDA CIRA SAUD "
                  "AMES UNIT OLFI MPCI NIPH DOMT EGCH ODIN TALM ASCM").split()
            univ = [dict(symbol=t + ".CA", currency="EGP", marketCap=None) for t in eg]
        else:
            univ = [q for q in screen_all(src) if q.get("quoteType", "EQUITY") == "EQUITY"]
        ccys = sorted({q.get("currency") for q in univ if q.get("currency")})
        fx = fx_to_usd(ccys)
        if cap is not None:
            univ = [q for q in univ if q.get("marketCap") and q["marketCap"] < cap]
        if name.startswith("Nasdaq First North"):
            eur = 1 / fx_to_usd(["EUR"])["EUR"]
            univ = [q for q in univ if q.get("marketCap") and q["marketCap"] * fx[q["currency"]] * eur < NORDIC_CAP_EUR]
        univ = [q for q in univ if q.get("marketCap") is None or q["marketCap"] > 0]
        pick = rng.choice(len(univ), size=min(N_SAMPLE, len(univ)), replace=False)
        sample = [univ[i] for i in pick]
        data = _retry(lambda: yf.download([q["symbol"] for q in sample], period="1y", interval="1d",
                                          group_by="ticker", auto_adjust=False, progress=False, threads=False))
        quotes = live_quotes([q["symbol"] for q in sample])
        stats = []
        for q in sample:
            try:
                st = stock_stats(data[q["symbol"]], fx[q["currency"]])
            except Exception:
                st = None
            if st:
                st["symbol"] = q["symbol"]; st["spread"] = quotes.get(q["symbol"], np.nan); stats.append(st)
        d = pd.DataFrame(stats)
        d["impact_10k"] = (d["amihud"] * 10_000).clip(upper=0.5)
        cost = d["spread"].fillna(0) + 2 * d["impact_10k"]
        d["ratio"] = (d["vol_ann"] / np.sqrt(252)) / cost.replace(0, np.nan)
        per_stock[name] = d.to_dict("records")
        rows.append(dict(market=name, universe=len(univ), n=len(d),
                         vol_ann=d["vol_ann"].median(), spread_bps=d["spread"].median() * 1e4,
                         quotes=int(d["spread"].notna().sum()), impact_10k_bps=d["impact_10k"].median() * 1e4,
                         zero_days=d["zero_days"].median(), adv_usd=d["adv_usd"].median(),
                         ratio=d["ratio"].median(),
                         share_tradeable=float(((d["adv_usd"] > 100_000) & (cost < 0.01)).mean())))
        json.dump(dict(row=rows[-1], stocks=per_stock[name]), open(cache, "w"), default=float)
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("results/market_scan/market_scan.csv", index=False)
    json.dump(per_stock, open("results/market_scan/market_scan_stocks.json", "w"), default=float)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
