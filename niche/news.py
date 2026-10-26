"""News senses.

HISTORICAL (for training and the blind test): the GDELT 2.0 DOC archive, which monitors world news
in 65 languages (machine-translated) since 2017. For each query we pull daily:
  all  = number of matching articles          bull = matching articles that also contain bullish keywords
  norm = all articles GDELT monitored that day bear = matching articles that also contain bearish keywords
  tone = average tone of matching articles (negative = gloomy)

LIVE (for using the model day to day): crawler.py scans fresh headlines for the same keyword lists, so
the senses the fly trained on are the senses it gets in live use.
"""
import json, os, random, subprocess, time, urllib.parse
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "news")
API = "https://api.gdeltproject.org/api/v2/doc/doc"
START, END = "20170101000000", "20261025000000"

BULL = "(rally OR surge OR record OR upgrade OR buyback OR dividend OR beats OR rebound OR outperform OR growth)"
BEAR = "(selloff OR plunge OR slump OR downgrade OR sanctions OR lawsuit OR default OR probe OR warning OR crash)"

MARKET_QUERIES = {
    "tokyo": '(Nikkei OR TOPIX OR "Tokyo stocks" OR "Japanese stocks")',
    "vietnam": '("VN-Index" OR "Vietnam stocks" OR "Vietnamese stocks" OR "Ho Chi Minh Stock Exchange")',
    "nordic": '("Stockholm stocks" OR OMXS30 OR "Swedish stocks" OR "Nordic stocks" OR "Helsinki stocks" OR "Copenhagen stocks")',
    "aim": '(FTSE OR "London stocks" OR "UK stocks" OR "British stocks")',
    "moex": '("Moscow Exchange" OR "Russian stocks" OR "MOEX index" OR "Russian market")',
}
# English names GDELT's translation layer uses for IMOEX companies
MOEX_NAMES = {
    "AFKS": "Sistema", "AFLT": "Aeroflot", "ALRS": "Alrosa", "BSPB": '"Bank Saint Petersburg"', "CBOM": '"Credit Bank of Moscow"',
    "CHMF": "Severstal", "ENPG": '"EN+ Group"', "GAZP": "Gazprom", "GMKN": '"Norilsk Nickel"', "HEAD": "HeadHunter",
    "IRAO": '"Inter RAO"', "LKOH": "Lukoil", "MAGN": '"Magnitogorsk Iron"', "MOEX": '"Moscow Exchange"', "MTSS": '"Mobile TeleSystems"',
    "NLMK": "NLMK", "NVTK": "Novatek", "OZON": "Ozon", "PHOR": "PhosAgro", "PLZL": "Polyus", "POSI": '"Positive Technologies"',
    "ROSN": "Rosneft", "RTKM": "Rostelecom", "RUAL": "Rusal", "SBER": "Sberbank", "SNGS": "Surgutneftegas", "TATN": "Tatneft",
    "TRNFP": "Transneft", "VKCO": '"VK Company"', "VTBR": '"VTB Bank"', "X5": '"X5 Retail"', "YDEX": "Yandex", "T": '(Tinkoff OR "T-Bank")',
    "RENI": '"Renaissance Insurance"', "MDMG": '"MD Medical"', "SVCB": "Sovcombank", "UGLD": '"Southern Urals Gold"',
    "FLOT": "Sovcomflot", "RAGR": '"Rusagro"', "CNRU": "Cian", "SBERP": None, "SNGSP": None, "TATNP": None, "DOMRF": '"DOM.RF"',
}


def _call(query, mode):
    url = f"{API}?query={urllib.parse.quote(query)}&mode={mode}&format=json&startdatetime={START}&enddatetime={END}"
    for i in range(40):
        r = subprocess.run(["curl", "-s", "-m", "120", "-A", "Mozilla/5.0", url], capture_output=True, text=True)
        t = r.stdout
        if t.startswith("{"):
            try:
                d = json.loads(t)
                tl = d.get("timeline") or []
                if not tl:
                    return pd.Series(dtype=float), pd.Series(dtype=float)
                s = tl[0]["data"]
                idx = pd.to_datetime([x["date"][:8] for x in s])
                return pd.Series([x["value"] for x in s], index=idx), pd.Series([x.get("norm", 0) for x in s], index=idx)
            except Exception:
                pass
        time.sleep(random.uniform(15, 45))          # shared network: back off and try again
    raise RuntimeError(f"GDELT unavailable for {query}")


def series(key, base_query):
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f"{key}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    a, normv = _call(base_query, "timelinevolraw"); time.sleep(6)
    b, _ = _call(f"{base_query} {BULL}", "timelinevolraw"); time.sleep(6)
    c, _ = _call(f"{base_query} {BEAR}", "timelinevolraw"); time.sleep(6)
    tone, _ = _call(base_query, "timelinetone"); time.sleep(6)
    df = pd.DataFrame({"all": a, "bull": b, "bear": c, "norm": normv, "tone": tone}).sort_index()
    df.to_pickle(path)
    return df


def build(markets=("tokyo", "vietnam", "nordic", "aim", "moex"), companies=True):
    for k in markets:
        df = series(f"market_{k}", MARKET_QUERIES[k])
        print(k, len(df), df.index.min().date() if len(df) else "-", int(df["all"].sum()) if len(df) else 0, flush=True)
    if companies:
        for tk, name in MOEX_NAMES.items():
            if not name:
                continue
            df = series(f"moex_{tk}", name)
            print(tk, len(df), int(df["all"].sum()) if len(df) else 0, flush=True)


def load_market(key):
    p = os.path.join(ROOT, f"market_{key}.pkl")
    return pd.read_pickle(p) if os.path.exists(p) else None


def load_moex_companies():
    out = {}
    for tk, name in MOEX_NAMES.items():
        src = tk if name else {"SBERP": "SBER", "SNGSP": "SNGS", "TATNP": "TATN"}[tk]   # preference shares share news
        p = os.path.join(ROOT, f"moex_{src}.pkl")
        if os.path.exists(p):
            out[tk] = pd.read_pickle(p)
    return out


if __name__ == "__main__":
    build()
