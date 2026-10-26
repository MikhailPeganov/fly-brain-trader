"""Live news crawler: the fly's news senses for day-to-day use.

For each market (and each MOEX company) it pulls the last 24 hours of headlines from
  * the GDELT 2.0 article list (world news, 65 languages, machine-translated), and
  * Yahoo Finance RSS headline feeds for individual tickers (where Yahoo covers them),
then scans each headline for the SAME bullish/bearish keyword lists the fly was trained on
(niche/news.py), so live inputs match training inputs.

Output: per market/company counts of matching headlines, bullish, bearish, and the net balance,
plus the headlines themselves so a human can check what moved the score.
Run:  PYTHONPATH=. python3 -m niche.crawler
"""
import json, re, subprocess, time, urllib.parse, xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .news import BULL, BEAR, MARKET_QUERIES, MOEX_NAMES, API

WORDS_BULL = re.findall(r"[a-z]+", BULL.lower().replace(" or ", " "))
WORDS_BEAR = re.findall(r"[a-z]+", BEAR.lower().replace(" or ", " "))


def _curl(url):
    r = subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", url], capture_output=True, text=True)
    return r.stdout


def classify(title):
    t = title.lower()
    bull = sum(bool(re.search(rf"\b{w}", t)) for w in WORDS_BULL)
    bear = sum(bool(re.search(rf"\b{w}", t)) for w in WORDS_BEAR)
    return bull, bear


def gdelt_headlines(query, tries=8):
    url = f"{API}?query={urllib.parse.quote(query)}&mode=artlist&maxrecords=75&timespan=1d&format=json&sort=datedesc"
    for _ in range(tries):
        t = _curl(url)
        if t.startswith("{"):
            try:
                return [dict(title=a["title"], source=a.get("domain"), url=a["url"], lang=a.get("language"))
                        for a in json.loads(t).get("articles", [])]
            except Exception:
                return []
        time.sleep(20)
    return []


def yahoo_headlines(symbol):
    t = _curl(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(symbol)}")
    try:
        root = ET.fromstring(t)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = it.findtext("title") or ""
        when = it.findtext("pubDate") or ""
        out.append(dict(title=title, source="Yahoo Finance", url=it.findtext("link"), published=when))
    return out


def scan(items):
    rows = []
    for a in items:
        b, s = classify(a["title"])
        rows.append(dict(a, bull=b, bear=s))
    n = len(rows); nb = sum(r["bull"] > 0 for r in rows); ns = sum(r["bear"] > 0 for r in rows)
    return dict(headlines=n, bullish=nb, bearish=ns, net=(nb - ns) / max(n, 1), items=rows)


def crawl(markets=MARKET_QUERIES, companies=MOEX_NAMES, yahoo_symbols=()):
    out = dict(crawled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), markets={}, companies={}, tickers={})
    for k, q in markets.items():
        out["markets"][k] = scan(gdelt_headlines(q)); time.sleep(6)
        print("market", k, {x: out["markets"][k][x] for x in ("headlines", "bullish", "bearish")}, flush=True)
    for tk, name in companies.items():
        if name:
            out["companies"][tk] = scan(gdelt_headlines(name)); time.sleep(6)
    for s in yahoo_symbols:
        out["tickers"][s] = scan(yahoo_headlines(s))
    return out


if __name__ == "__main__":
    res = crawl(yahoo_symbols=("7203.T", "6758.T", "AAPL"))
    json.dump(res, open("results/live_news_snapshot.json", "w"), indent=1, ensure_ascii=False)
    print("saved results/live_news_snapshot.json")
