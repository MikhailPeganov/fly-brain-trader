# Fly-Brain Trader

A trading model based on the fruit fly's mushroom body (the brain region flies use to learn from reward and punishment). It feels **pain** when a trade loses, **happiness** when it wins, and in version 3 **love** when it is beating the S&P 500. Every version is tested **blind** on stocks and dates it never saw, with hypothesis tests that ask whether it has a real strategy or is just lucky.

**Headline result:** across three studies and six markets, the model never beat its benchmark after costs. One weak, repeated hint of real stock-picking skill appeared in Tokyo small caps, but it doesn't survive correction for multiple testing. The value of this project is the testing framework: it catches look-ahead bugs, separates skill from luck, and was itself checked with placebo markets and planted signals.

Results pages, online at **https://mikhailpeganov.github.io/fly-brain-trader/** (or open the files in `docs/` locally):
- `docs/us-blind-test.html`: study 1, US large caps
- `docs/niche-markets.html`: study 2, niche markets
- `docs/v3.html`: study 3, news + maths school + love + MOEX

---

## The three studies

| Study | What was tested | Main result |
|---|---|---|
| **1. US large caps** | 60 US stocks; trained on 40, tested blind on 20 hidden stocks (Jan 2024 – Oct 2026), plus a bar-by-bar replay of an unseen day (23 Oct 2026) | Net Sharpe 0.26; a permutation test found 15% of time-shuffled copies did as well (p = 0.147). Hit rate 50.1% vs 50.0% by chance. **No evidence of skill.** |
| **2. Niche markets** | Two long-only strategies (slow picker, shock rebound) in Vietnam, Tokyo, Nordic and UK small caps, with live spreads, price impact and commissions | 0 of 8 strategy–market pairs beat their market. The Tokyo picker beat random picking (p = 0.028), but **p = 0.23 after Holm correction**. |
| **3. Version 3** | Adds news senses, a maths "school", a long/short brain and love; five markets including MOEX; measured in USD against the S&P 500 | Beat the S&P 500 in 0 of 5 markets (best: MOEX +6.7% a year vs S&P +20.6%). Tokyo picking edge again (p = 0.041), **p = 0.41 after correction**. |

Things the testing caught along the way:
- **A look-ahead bug.** The first live-day replay looked brilliant (p = 0.001) because one feature used the day's own closing price. After the fix, the edge vanished.
- **A biased benchmark.** Averaging log returns understated the market, flattering one strategy. It was fixed and rerun.
- **The tests themselves were validated.** On 40 placebo markets with no signal, 5% came out "significant", as chance predicts. A planted signal was detected.

---

## How the model works

```
quant factors + simple models + news  ──►  projection neurons (population code)
                                            │  each Kenyon cell samples ~7 at random (FlyWire statistics)
                                            ▼
                                     2,000 Kenyon cells, only top 5% active (APL inhibition)
                                            │  plastic synapses
                                            ▼
                          output neurons: "go long" vs "go short"  ──►  position
                                            ▲
         dopamine: happiness (gain) / pain (loss, weighted 2×) / love (ahead of S&P 500)
```

- **Learning:** three-factor dopamine-gated plasticity at the Kenyon-cell → output synapses. There is also weaker "what if" (fictive) learning for trades not taken, and a tonic mood that makes the model cautious after a run of losses.
- **Version 3 adds:**
  - a direct sensory pathway (like the fly's lateral horn);
  - a second "market brain" that decides long vs short;
  - news senses from the GDELT archive (article volume, tone, bullish/bearish keyword share);
  - six simple-model senses (trend, momentum, Bollinger z-score, rolling least-squares forecast, EWMA volatility, beta);
  - a synthetic-market school with exams.

---

## Repository layout

```
fly-brain-trader/
├── README.md                  this file
├── LICENSE                    MIT
├── requirements.txt           Python packages
├── .gitignore                 keeps downloaded data and caches out of git
├── flybrain/                  core model (study 1)
│   ├── brain.py               mushroom-body trader: Kenyon cells, dopamine pain/happiness learning
│   ├── features.py            16 daily + 12 intraday quant factors
│   ├── sim.py                 causal trading simulator
│   ├── stats.py               skill-vs-luck tests: permutation, Newey-West, bootstrap, alpha, hit rate
│   ├── data.py                Yahoo Finance download + cache (US stocks)
│   └── pipeline.py            train/validation/test split and hidden-stock split
├── niche/                     studies 2 and 3
│   ├── data.py                universes, prices and live spreads for the niche markets
│   ├── strategies.py          study 2: slow picker and shock-rebound strategies
│   ├── v3.py                  study 3: stock brain + market brain, news & simple-model senses, love
│   ├── school.py              study 3: synthetic "maths school" and exams
│   ├── news.py                historical news senses from the GDELT archive
│   ├── crawler.py             live news crawler (same keywords as training)
│   └── moex.py                Moscow Exchange data: dividends, splits, ticker changes (research only)
├── experiments/               one script per experiment; run from the repo root
│   ├── us_blind_test.py       study 1: tune, lock, blind test, seeds, placebo
│   ├── live_day_replay.py     study 1: bar-by-bar replay of an unseen day
│   ├── market_scan.py         volatility vs trading cost across 10 markets
│   ├── niche_markets.py       study 2: pre-registered blind test + hypothesis tests
│   ├── tokyo_checks.py        study 2: exploratory checks on the Tokyo result
│   ├── v3_experiment.py       study 3: school, tuning, blind test, ablations
│   ├── power_sweep.py         study 1 add-on: power curve for planted signals of rising strength
│   └── build_reports.py       rebuilds the docs/ pages from results/
├── results/                   saved outputs (JSON + run logs) for every study
│   ├── us_blind_test/
│   ├── market_scan/
│   ├── niche/
│   └── v3/
└── docs/                      results pages (GitHub Pages)
    ├── index.html
    ├── us-blind-test.html
    ├── niche-markets.html
    ├── v3.html
    └── templates/             page templates used by build_reports.py
```

---

## Running it

Requires Python 3.11+ and `curl` (preinstalled on macOS and most Linux).

```bash
git clone https://github.com/MikhailPeganov/fly-brain-trader.git
cd fly-brain-trader
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root. Downloaded data goes into `data/` (git-ignored).

| Step | Command | Time | Downloads |
|---|---|---|---|
| Study 1 data | `python -m flybrain.data` | ~2 min | Yahoo Finance, 61 US tickers |
| Study 1 | `python -m experiments.us_blind_test` | ~10 min | none |
| Live-day replay | `python -m experiments.live_day_replay` | ~2 min | none |
| Power curve (optional) | `python -m experiments.power_sweep` | ~5 min | none |
| Market scan | `python -m experiments.market_scan` | ~15 min | Yahoo screener + quotes |
| Study 2 data | `python -m niche.data` | ~20 min | Yahoo, 4 markets × 235–500 stocks |
| Study 2 | `python -m experiments.niche_markets` | ~7 min | none |
| Tokyo checks | `python -m experiments.tokyo_checks` | ~5 min | none |
| Study 3 data | `python -m niche.moex` then `python -m niche.news` | 30 min – hours | MOEX ISS, dohod.ru, GDELT (rate-limited) |
| Study 3 | `python -m experiments.v3_experiment` | ~15 min | FX rates |
| Rebuild pages | `python -m experiments.build_reports` | seconds | none |
| Live news | `python -m niche.crawler` | ~5 min | GDELT + Yahoo RSS |

Notes:
- **Results will differ slightly.** Yahoo's data changes over time (revisions, delistings), so a rerun won't reproduce the saved numbers exactly. The saved `results/` are the record of the runs described above.
- **The news archive is slow.** GDELT asks for at most one request every 5 seconds, and `niche/news.py` backs off automatically. Market-level news (20 queries) is enough for study 3; company-level news is optional and takes hours.

---

## Limitations

- **Survivorship bias:** stocks are sampled from today's listings, so companies that went bust are missing. This flatters raw returns. The random-picker tests share the same lists, so the edge tests are fairer.
- **Costs** use one snapshot of today's quoted spreads for the whole history, plus estimated price impact and a 0.10% commission.
- **News in study 3 is market-level only** in the backtest. Per-company history was too slow to download, so company-news senses were zero in training and testing.
- **Short-selling** outside MOEX is modelled as shorting an equal-weight basket, standing in for an index future.
- **The maths school** passed 3 of 5 exams (random walk, low volatility, news) and failed momentum and mean reversion.

## Legal and disclaimer

- **Not investment advice.** This is a research project. Nothing here recommends buying or selling anything.
- **MOEX is research only.** The UK designated the Moscow Exchange under its Russia sanctions in June 2024; UK persons can't deal there. `niche/moex.py` only downloads public historical prices for backtesting.
- **Data** comes from Yahoo Finance, the GDELT Project, the MOEX ISS API and dohod.ru, under their own terms. Raw data isn't redistributed in this repository; the scripts download it.

## License

MIT; see `LICENSE`.
