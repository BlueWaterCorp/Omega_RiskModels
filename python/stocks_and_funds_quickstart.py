#!/usr/bin/env python3
"""
RiskModels × Omega Point quickstart (CLI twin of the quickstart notebook).

API-only: metrics, funds, pre-rendered snapshot panels, server PDFs.
No local zarr.

  python python/stocks_and_funds_quickstart.py --no-show
  python python/stocks_and_funds_quickstart.py --stock AAPL --fund VTSAX --no-show
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

_PRE_PARSE = argparse.ArgumentParser(add_help=False)
_PRE_PARSE.add_argument("--no-show", action="store_true")
_PRE_ARGS, _ = _PRE_PARSE.parse_known_args()
if _PRE_ARGS.no_show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from riskmodels import RiskModelsClient
from riskmodels.notebook import quickstart_connect

COLORS = {
    "market": "#4F46E5",
    "sector": "#16A34A",
    "subsector": "#0EA5E9",
    "idiosyncratic": "#6B7280",
    "gross": "#111827",
}

DEFAULT_BOOK = {"NVDA": 0.25, "AAPL": 0.25, "MSFT": 0.25, "JPM": 0.25}
PANEL_SLUGS = (
    "l3_explained_risk_hbar",
    "hedge_notionals_hbar",
    "hedge_depth_retained",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RiskModels Orth Risk quickstart for Omega Point (API-only).",
    )
    p.add_argument("--stock", default="NVDA")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--fund", default="AGTHX")
    p.add_argument("--notional", type=float, default=10_000_000)
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--skip-panels", action="store_true", help="Skip snapshot_panel + stock PDF")
    p.add_argument("--skip-stock", action="store_true")
    p.add_argument("--skip-fund", action="store_true")
    p.add_argument("--skip-book", action="store_true")
    return p.parse_args()


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print("(empty)" if frame.empty else frame.to_string())


def save_or_show(fig: plt.Figure, path: Path, *, no_show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()


def api_get(session: Any, base_url: str, path: str, **params: Any) -> tuple[dict, dict]:
    response = session.get(f"{base_url}{path}", params=params, timeout=90)
    if not response.ok:
        raise RuntimeError(f"GET {path} failed ({response.status_code}): {response.text[:500]}")
    headers = {
        "data_as_of": response.headers.get("X-Data-As-Of"),
        "filing_date": response.headers.get("X-Data-Filing-Date"),
        "model_version": response.headers.get("X-Risk-Model-Version"),
        "cost_usd": response.headers.get("X-API-Cost-USD"),
    }
    return response.json(), headers


def first_value(series: pd.Series, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in series.index and pd.notna(series[name]):
            return series[name]
    return default


def pick_col(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def run_panels(client: RiskModelsClient, ticker: str, output_dir: Path) -> None:
    print(f"\n# Part 0 · Product surfaces — {ticker}")
    for slug in PANEL_SLUGS:
        try:
            payload, lineage = client.snapshot_panel("stock", ticker, slug, format="png")
            if not isinstance(payload, (bytes, bytearray)):
                print(f"SKIP {slug}: non-bytes payload")
                continue
            path = output_dir / f"{ticker}_{slug}.png"
            path.write_bytes(payload)
            print(
                f"OK  {slug}  ({len(payload):,} bytes)  "
                f"as_of={getattr(lineage, 'data_as_of', None)}  → {path.name}"
            )
        except Exception as exc:  # noqa: BLE001 — demo resilience
            print(f"SKIP {slug}: {exc}")

    try:
        pdf_bytes, _ = client.get_metrics_snapshot_pdf(ticker)
        path = output_dir / f"{ticker}_metrics_snapshot.pdf"
        path.write_bytes(pdf_bytes)
        print(f"PDF {path.name} ({len(pdf_bytes):,} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP metrics snapshot PDF: {exc}")


def run_stock(
    client: RiskModelsClient,
    ticker: str,
    years: int,
    notional: float,
    output_dir: Path,
    *,
    no_show: bool,
) -> None:
    print(f"\n# Part I · Stock — {ticker}")
    metrics_df = client.get_metrics(ticker, as_dataframe=True)
    if metrics_df.empty:
        raise ValueError(f"No metrics returned for {ticker}.")
    row = metrics_df.iloc[-1]

    er = pd.Series(
        {
            "Market": first_value(row, "l3_market_er", "l3_mkt_er"),
            "Sector": first_value(row, "l3_sector_er", "l3_sec_er"),
            "Subsector": first_value(row, "l3_subsector_er", "l3_sub_er"),
            "Residual": first_value(row, "l3_residual_er", "l3_res_er"),
        },
        dtype="float64",
    )
    hedge = pd.DataFrame(
        [
            {
                "Layer": "Market",
                "ETF": first_value(row, "market_factor_etf", "market_etf", default="SPY"),
                "HR ($ ETF / $1 stock)": first_value(row, "l3_market_hr", "l3_mkt_hr"),
            },
            {
                "Layer": "Sector",
                "ETF": first_value(row, "sector_etf", default="sector ETF"),
                "HR ($ ETF / $1 stock)": first_value(row, "l3_sector_hr", "l3_sec_hr"),
            },
            {
                "Layer": "Subsector",
                "ETF": first_value(row, "subsector_etf", default="subsector ETF"),
                "HR ($ ETF / $1 stock)": first_value(row, "l3_subsector_hr", "l3_sub_hr"),
            },
        ]
    )
    hedge["Notional hedge ($)"] = hedge["HR ($ ETF / $1 stock)"] * notional

    print_frame(
        f"{ticker} snapshot",
        pd.DataFrame(
            {
                "Value": {
                    "Ticker": ticker,
                    "As of": first_value(row, "date", "teo", default="latest"),
                    "Price": first_value(row, "price_close"),
                    "23d annualized volatility": first_value(row, "vol_23d"),
                    "L3 ER total": er.sum(min_count=1),
                    "Ticket notional": notional,
                }
            }
        ),
    )
    hedge_display = hedge.copy()
    hedge_display["HR ($ ETF / $1 stock)"] = hedge_display["HR ($ ETF / $1 stock)"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    hedge_display["Notional hedge ($)"] = hedge_display["Notional hedge ($)"].map(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
    print_frame(f"{ticker} hedge ticket", hedge_display)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = [COLORS["market"], COLORS["sector"], COLORS["subsector"], COLORS["idiosyncratic"]]
    er_plot = er.dropna()
    er_plot.plot(kind="barh", ax=ax, color=colors[: er_plot.size])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(left=0)
    ax.set_xlabel("Share of variance")
    ax.set_title(f"{ticker} — current L3 Orth Risk decomposition")
    ax.grid(axis="x", alpha=0.2)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.1%}" for v in er_plot], padding=4)
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{ticker}_l3_er.png", no_show=no_show)

    stock_history = client.get_ticker_returns(ticker, years=years).copy()
    if stock_history.empty:
        raise ValueError(f"No return history returned for {ticker}.")
    stock_history["date"] = pd.to_datetime(stock_history["date"])
    stock_history = stock_history.sort_values("date").reset_index(drop=True)

    gross_col = pick_col(stock_history, "returns_gross", "gross_return")
    l1_fr = pick_col(stock_history, "l1_factor_return", "l1_fr")
    l2_fr = pick_col(stock_history, "l2_factor_return", "l2_fr")
    l3_fr = pick_col(stock_history, "l3_factor_return", "l3_fr")
    l1_cfr = pick_col(stock_history, "l1_combined_factor_return", "l1_cfr")
    l2_cfr = pick_col(stock_history, "l2_combined_factor_return", "l2_cfr")
    l3_cfr = pick_col(stock_history, "l3_combined_factor_return", "l3_cfr")
    l3_rr = pick_col(stock_history, "l3_residual_return", "l3_rr")
    if gross_col is None or l3_cfr is None:
        raise KeyError(f"Missing gross/L3 fields. Columns: {list(stock_history.columns)}")

    contrib = pd.DataFrame(index=stock_history.index)
    contrib["Market"] = stock_history[l1_fr] if l1_fr else stock_history[l1_cfr]
    contrib["Sector"] = (
        stock_history[l2_fr] if l2_fr else stock_history[l2_cfr] - stock_history[l1_cfr]
    )
    contrib["Subsector"] = (
        stock_history[l3_fr] if l3_fr else stock_history[l3_cfr] - stock_history[l2_cfr]
    )
    contrib["Idiosyncratic"] = (
        stock_history[l3_rr] if l3_rr else stock_history[gross_col] - stock_history[l3_cfr]
    )
    contrib["Gross"] = stock_history[gross_col]
    cumulative = contrib.fillna(0).cumsum()

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, key in [
        ("Market", "market"),
        ("Sector", "sector"),
        ("Subsector", "subsector"),
        ("Idiosyncratic", "idiosyncratic"),
    ]:
        ax.plot(stock_history["date"], cumulative[label], label=label, color=COLORS[key], lw=1.6)
    ax.plot(
        stock_history["date"],
        cumulative["Gross"],
        label="Gross",
        color=COLORS["gross"],
        lw=2.2,
        ls="--",
    )
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"{ticker} — arithmetic cumulative return attribution ({years}y)")
    ax.legend(ncol=5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{ticker}_attribution_{years}y.png", no_show=no_show)

    gap = (
        contrib[["Market", "Sector", "Subsector", "Idiosyncratic"]].sum(axis=1) - contrib["Gross"]
    ).abs()
    print("Mean absolute daily identity gap:", f"{gap.mean():.6%}")


def run_fund(
    session: Any,
    base_url: str,
    fund_query: str,
    notional: float,
    output_dir: Path,
    *,
    no_show: bool,
) -> None:
    print(f"\n# Part II · Fund — {fund_query}")
    search_body, search_headers = api_get(session, base_url, "/funds/search", q=fund_query, limit=10)
    fund_hits = search_body.get("results") or []
    if not fund_hits:
        raise ValueError(f"No fund matched {fund_query!r}.")

    hits = pd.DataFrame(fund_hits)
    show_cols = [
        c
        for c in [
            "ticker",
            "fund_name",
            "equity_style_9box",
            "morningstar_category",
            "net_expense_ratio",
            "latest_report_date",
            "bw_fund_id",
        ]
        if c in hits.columns
    ]
    print_frame(f"Fund search — {fund_query!r}", hits[show_cols].head(10))
    print("Search headers:", {k: v for k, v in search_headers.items() if v})

    ticker_series = hits.get("ticker", pd.Series(index=hits.index, dtype=str))
    exact = hits[ticker_series.astype(str).str.upper() == fund_query.upper()]
    selected = (exact.iloc[0] if not exact.empty else hits.iloc[0]).to_dict()
    fund_id = selected["bw_fund_id"]
    fund_label = str(selected.get("ticker") or selected.get("fund_name") or fund_id)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in fund_label)
    print(f"Selected: {fund_label} | {selected.get('fund_name')} | {fund_id}")

    fund_snapshot, fund_headers = api_get(session, base_url, f"/funds/snapshot/{fund_id}")
    fund_metrics = fund_snapshot.get("metrics") or {}
    fund_returns = fund_metrics.get("returns") or {}
    diagnostics = fund_metrics.get("diagnostics") or {}
    metadata = fund_snapshot.get("_metadata") or fund_metrics.get("_metadata") or {}

    print_frame(
        f"{fund_label} identity",
        pd.DataFrame(
            {
                "Value": {
                    "Fund": fund_snapshot.get("fund_name") or selected.get("fund_name"),
                    "Ticker": fund_snapshot.get("ticker") or selected.get("ticker"),
                    "Style cell": fund_snapshot.get("equity_style_9box")
                    or selected.get("equity_style_9box"),
                    "Holdings report date": fund_snapshot.get("report_date")
                    or fund_headers.get("data_as_of"),
                    "Filing date": fund_snapshot.get("filing_date")
                    or fund_headers.get("filing_date"),
                    "Model version": metadata.get("model_version")
                    or fund_headers.get("model_version"),
                    "API cost for this call": fund_headers.get("cost_usd"),
                }
            }
        ),
    )

    latest = pd.Series(
        {
            "Market": fund_returns.get("market"),
            "Sector": fund_returns.get("sector"),
            "Subsector": fund_returns.get("subsector"),
            "Idiosyncratic": fund_returns.get("idiosyncratic"),
        },
        dtype="float64",
    )
    bars = latest.dropna()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(
        bars.index,
        bars.values,
        color=[COLORS["market"], COLORS["sector"], COLORS["subsector"], COLORS["idiosyncratic"]][
            : len(bars)
        ],
    )
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"{fund_label} — latest holdings-derived return decomposition")
    ax.grid(axis="y", alpha=0.2)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.2%}" for v in bars.values], padding=3)
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{safe}_latest_decomp.png", no_show=no_show)

    history_rows = ((fund_snapshot.get("portfolio_history") or {}).get("rows") or [])
    fund_history = pd.DataFrame(history_rows)
    if fund_history.empty:
        print("No portfolio history in this fund snapshot.")
    else:
        fund_history["teo"] = pd.to_datetime(fund_history["teo"])
        fund_history = fund_history.sort_values("teo").reset_index(drop=True)
        rename = {
            "portfolio_market_return": "Market",
            "portfolio_sector_return": "Sector",
            "portfolio_subsector_return": "Subsector",
            "portfolio_idiosyncratic_return": "Idiosyncratic",
            "portfolio_gross_return": "Gross",
        }
        available = {k: v for k, v in rename.items() if k in fund_history.columns}
        fund_attr = fund_history[list(available)].rename(columns=available).fillna(0).cumsum()
        fig, ax = plt.subplots(figsize=(11, 5))
        for label, key in [
            ("Market", "market"),
            ("Sector", "sector"),
            ("Subsector", "subsector"),
            ("Idiosyncratic", "idiosyncratic"),
        ]:
            if label in fund_attr:
                ax.plot(fund_history["teo"], fund_attr[label], label=label, color=COLORS[key], lw=1.8)
        if "Gross" in fund_attr:
            ax.plot(
                fund_history["teo"],
                fund_attr["Gross"],
                label="Gross",
                color=COLORS["gross"],
                lw=2.2,
                ls="--",
            )
        ax.axhline(0, color="#9CA3AF", linewidth=0.8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f"{fund_label} — arithmetic cumulative fund attribution")
        ax.legend(ncol=5, frameon=False, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_or_show(fig, output_dir / f"{safe}_attribution.png", no_show=no_show)

    print_frame(
        f"{fund_label} diagnostics",
        pd.DataFrame(
            {
                "Value": {
                    "ERM3 universe coverage": diagnostics.get("weight_sum"),
                    "Active holdings": diagnostics.get("n_holdings_active"),
                    "Effective N (HHI)": diagnostics.get("effective_n"),
                    "Top-10 weight": diagnostics.get("top10_weight_sum"),
                }
            }
        ),
    )

    hedge_rows = []
    hedge_block = fund_snapshot.get("hedge") or {}
    for level in ["L1", "L2", "L3"]:
        for leg in hedge_block.get(level) or []:
            hedge_rows.append(
                {"Level": level, "ETF": leg.get("etf"), "HR ($ ETF / $1 fund)": leg.get("hr")}
            )
    fund_hedge = pd.DataFrame(hedge_rows)
    if fund_hedge.empty:
        print("No fund hedge basket.")
        return
    fund_hedge["Notional hedge ($)"] = fund_hedge["HR ($ ETF / $1 fund)"] * notional
    disp = fund_hedge.copy()
    disp["HR ($ ETF / $1 fund)"] = disp["HR ($ ETF / $1 fund)"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    )
    disp["Notional hedge ($)"] = disp["Notional hedge ($)"].map(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
    print_frame(f"{fund_label} hedge ticket", disp)

    plotted = fund_hedge.dropna(subset=["HR ($ ETF / $1 fund)"])
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.32 * len(plotted))))
    labels = plotted["Level"] + " · " + plotted["ETF"].astype(str)
    colors = plotted["Level"].map(
        {"L1": COLORS["market"], "L2": COLORS["sector"], "L3": COLORS["subsector"]}
    )
    ax.barh(labels, plotted["HR ($ ETF / $1 fund)"], color=colors)
    ax.axvline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_xlabel("ETF notional per $1 of fund exposure")
    ax.set_title(f"{fund_label} — latest fund hedge basket")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{safe}_hedge_basket.png", no_show=no_show)


def run_book(client: RiskModelsClient, book: dict[str, float], output_dir: Path) -> None:
    print("\n# Part III · Bring your book")
    print("Book:", book)
    pa = client.analyze_portfolio(book, metrics=["full_metrics", "hedge_ratios"], years=1)
    print_frame(
        "Portfolio L3 ER (weight-mean)",
        pd.DataFrame([pa.portfolio_l3_er_weighted_mean]),
    )
    print_frame(
        "Portfolio hedge ratios",
        pd.DataFrame([pa.portfolio_hedge_ratios]),
    )
    try:
        pdf_bytes, _ = client.post_portfolio_risk_snapshot_pdf(
            book, title="Omega demo book"
        )
        path = output_dir / "demo_book_risk_snapshot.pdf"
        path.write_bytes(pdf_bytes)
        print(f"Book PDF → {path} ({len(pdf_bytes):,} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP book PDF: {exc}")


def main() -> int:
    args = parse_args()
    if args.skip_stock and args.skip_fund and args.skip_book and args.skip_panels:
        print("Nothing to run.", file=sys.stderr)
        return 2

    session, base_url, api_key = quickstart_connect()
    os.environ.setdefault("RISKMODELS_API_KEY", api_key)
    os.environ.setdefault("RISKMODELS_BASE_URL", base_url)
    client = RiskModelsClient.from_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Connected to", base_url)
    print(
        {
            "stock": args.stock,
            "years": args.years,
            "fund": args.fund,
            "notional": args.notional,
            "output_dir": str(args.output_dir),
            "offer": "api-only (panels + PDFs + metrics/funds; no zarr)",
        }
    )

    if not args.skip_panels:
        run_panels(client, args.stock, args.output_dir)
    if not args.skip_stock:
        run_stock(
            client,
            args.stock,
            args.years,
            args.notional,
            args.output_dir,
            no_show=args.no_show,
        )
    if not args.skip_fund:
        run_fund(
            session,
            base_url,
            args.fund,
            args.notional,
            args.output_dir,
            no_show=args.no_show,
        )
    if not args.skip_book:
        run_book(client, DEFAULT_BOOK, args.output_dir)

    print(
        "\nDone. Orth Risk is a hosted-model candidate alongside Barra — "
        "API panels/PDFs today; zarr history later if permissioned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
