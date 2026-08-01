# Omega_RiskModels

Partner-facing walkthrough for **[Omega Point](https://www.omegapoint.ai/)**: evaluate
**ERM3 Orth Risk** as a **hosted model** you could offer alongside Barra and other
models you already distribute.

This repo is **API-only**. Pre-rendered snapshot panels and server PDFs come from
the RiskModels API. Local / GCS zarr history is out of scope here (can be
permissioned later).

| Layer | What it isolates | What you can trade / study |
|---|---|---|
| Market | broad equity exposure | size a market ETF hedge |
| Sector | incremental sector exposure | industry beta vs market beta |
| Subsector | incremental business-model exposure | refine hedge + peers |
| Residual / idiosyncratic | what remains after the hierarchy | stock-specific or manager-selection sleeve |

**Docs:** [API guide](https://riskmodels.app/docs/api) · [Interactive reference](https://riskmodels.app/api-reference) · [Get an API key](https://riskmodels.app/get-key)

---

## Choose how to run

| Path | Best for | Entry point |
|---|---|---|
| **Notebook** | Colab / Jupyter, narrative session | [`notebooks/RiskModels_Stocks_and_Funds_the partner_Quickstart.ipynb`](notebooks/RiskModels_Stocks_and_Funds_the partner_Quickstart.ipynb) |
| **Script** | Terminal, saved artifacts | [`python/stocks_and_funds_quickstart.py`](python/stocks_and_funds_quickstart.py) |

---

## Setup (once)

```bash
cd Omega_RiskModels
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
```

### Add your API key

1. Get a key at [riskmodels.app/get-key](https://riskmodels.app/get-key).
2. Copy the example env file and paste the key (do **not** commit this file — it is gitignored):

```bash
cp .env.example .env.local
```

3. Edit `.env.local` so it looks like:

```bash
RISKMODELS_API_KEY=rm_your_actual_key_here
```

4. Run the notebook or script from the **repo root** (so `quickstart_connect()` can find `.env.local`).

**Alternatives**

| Method | When to use |
|---|---|
| `export RISKMODELS_API_KEY=...` in the shell | Script / one-off terminal session |
| Colab Secrets named `RISKMODELS_API_KEY` | Google Colab notebook |
| Secure prompt when nothing is set | `quickstart_connect()` will ask; key is not printed |

Optional: uncomment `RISKMODELS_BASE_URL` in `.env.local` only if you need a non-default API host (default is `https://riskmodels.app/api`).

---

## Session spine (API-only)

1. **Part 0 — Product surfaces:** `snapshot_panel` PNGs (`l3_explained_risk_hbar`,
   `hedge_notionals_hbar`, `hedge_depth_retained`) + `get_metrics_snapshot_pdf`
2. **Part I — Stock:** metrics, HR × `$` notional ticket, return attribution
3. **Part II — Fund:** search → composed snapshot → residual persistence → hedge ticket
4. **Part III — Bring your book:** small weight vector via `analyze_portfolio` + portfolio PDF

Not included: zarr-backed historical waterfall (permission later).

---

## Option A — Notebook

Open `notebooks/RiskModels_Stocks_and_Funds_the partner_Quickstart.ipynb`, connect, set
`STOCK_TICKER` / `FUND_QUERY` / `BOOK`, run top to bottom.

---

## Option B — Script

```bash
python python/stocks_and_funds_quickstart.py --no-show
python python/stocks_and_funds_quickstart.py --stock AAPL --fund VTSAX --notional 10000000 --no-show
python python/stocks_and_funds_quickstart.py --skip-fund --skip-book --no-show   # panels + stock only
```

Artifacts under `output/`:

- `{TICKER}_l3_explained_risk_hbar.png` (and sibling panel PNGs)
- `{TICKER}_metrics_snapshot.pdf`
- `{TICKER}_l3_er.png`, `{TICKER}_attribution_{years}y.png`
- fund decomp / attribution / hedge PNGs
- `demo_book_risk_snapshot.pdf`

---

## Layout

```
Omega_RiskModels/
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/RiskModels_Stocks_and_Funds_the partner_Quickstart.ipynb
├── python/stocks_and_funds_quickstart.py
└── output/
```
