#!/usr/bin/env python3
"""
RiskModels API quickstart (CLI twin of the notebook).

API-only: metrics, L* hedge-depth dispatch, funds with the FF2 style layer,
pre-rendered snapshot panels, server PDFs. No local zarr.

The cascade is one sequence: market → sector → subsector → FF2 style → final
residual. Tradeable hedge legs stop at subsector (one ETF each, signed dollars);
FF2 (SMB + HML) carries no ETF hedge — it is the attribution layer that keeps
style tilt out of the stock-selection read.

  python python/stocks_and_funds_quickstart.py --no-show
  python python/stocks_and_funds_quickstart.py --stock AAPL --fund VTSAX --no-show
"""

from __future__ import annotations

import argparse
import contextlib
import io
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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from riskmodels import RiskModelsClient
from riskmodels.notebook import quickstart_connect

# ── One visual language, matched to the server-rendered snapshot panels ──────
NAVY = "#1B2A4A"
LAYER_COLORS = {
    "Market": "#64748B",
    "Sector": "#1D6FA8",
    "Subsector": "#7C3AED",
    "Style (FF2)": "#0D9488",     # attribution only — no ETF hedge
    "L* residual": "#16A34A",
    "Residual": "#16A34A",
    "Final residual": "#16A34A",
    "Gross": NAVY,
}
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": NAVY, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E9F0", "grid.linewidth": 0.8,
    "axes.edgecolor": "#94A3B8", "axes.labelcolor": "#334155",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "legend.frameon": False,
})

DEFAULT_BOOK = {"NVDA": 0.25, "AAPL": 0.25, "MSFT": 0.25, "JPM": 0.25}
PANEL_SLUGS = (
    "l3_explained_risk_hbar",
    "hedge_notionals_hbar",
    "hedge_depth_retained",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RiskModels Orth Risk quickstart (API-only).",
    )
    p.add_argument("--stock", default="NVDA")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--fund", default="AGTHX")
    p.add_argument("--notional", type=float, default=10_000_000)
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--skip-panels", action="store_true", help="Skip snapshot_panel + stock PDF")
    p.add_argument("--skip-stock", action="store_true")
    p.add_argument("--skip-lstar", action="store_true", help="Skip the L* hedge-depth section")
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


def api_post(session: Any, base_url: str, path: str, **body: Any) -> tuple[dict, dict]:
    response = session.post(f"{base_url}{path}", json=body, timeout=90)
    if not response.ok:
        raise RuntimeError(f"POST {path} failed ({response.status_code}): {response.text[:500]}")
    return response.json(), {"cost_usd": response.headers.get("X-API-Cost-USD")}


def first_value(series: pd.Series, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in series.index and pd.notna(series[name]):
            return series[name]
    return default


def pick_col(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def waterfall(ax, steps: dict, total_label: str = "Gross", title: str | None = None) -> None:
    """Stacked waterfall: ordered {layer: contribution}. Bars float at the running
    cumulative, residual layers hatched, total bar in navy, connectors dotted."""
    cum = 0.0
    for i, (lab, v) in enumerate(steps.items()):
        hatch = "//" if "residual" in lab.lower() else None
        ax.bar(i, v, bottom=cum, width=0.62, color=LAYER_COLORS.get(lab, "#94A3B8"),
               hatch=hatch, edgecolor="white", linewidth=0.5)
        ax.annotate(f"{v:+.1%}", (i, cum + v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold",
                    color="#334155" if v >= 0 else "#B45309")
        cum += v
        ax.plot([i + 0.31, i + 0.69], [cum, cum], ls=":", color="#9CA3AF", lw=1)
    n = len(steps)
    ax.bar(n, cum, width=0.62, color=NAVY)
    ax.annotate(f"{cum:+.1%}", (n, max(cum, 0.0)), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=NAVY)
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(list(steps) + [total_label], fontsize=9)
    levels = np.concatenate([[0.0], np.cumsum(list(steps.values()))])
    hi, lo = float(levels.max()), float(min(0.0, levels.min()))
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - (0.12 * span if lo < 0 else 0.0), hi + 0.15 * span)
    ax.axhline(0, color="#9CA3AF", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if title:
        ax.set_title(title, loc="left")


def cum_with_t0(contrib: pd.DataFrame, dates) -> pd.DataFrame:
    """Arithmetic cumulative attribution anchored at 0: prepends a zero row one
    business day before the first date so every line starts at exactly 0."""
    dates = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    cum = contrib.reset_index(drop=True).fillna(0).cumsum()
    cum.index = dates
    t0 = dates.iloc[0] - pd.tseries.offsets.BDay(1)
    return pd.concat([pd.DataFrame(0.0, index=[t0], columns=cum.columns), cum])


def attribution_lines(ax, frame: pd.DataFrame, title: str | None = None) -> None:
    """Cumulative attribution lines (frame indexed by date, anchored at 0).
    Gross heavy navy; layers in the shared palette; dodged end-value labels.
    Ticks are clipped to the data range so the label margin shows no phantom dates."""
    dates = frame.index
    span = float(np.nanmax(frame.values) - np.nanmin(frame.values)) or 1.0
    for col in frame.columns:
        lw, color = (2.4, NAVY) if col == "Gross" else (1.5, LAYER_COLORS.get(col, "#94A3B8"))
        ax.plot(dates, frame[col], color=color, lw=lw)
    y_last = None
    for col, v in frame.iloc[-1].sort_values().items():
        y = v if y_last is None else max(v, y_last + 0.055 * span)
        y_last = y
        color = NAVY if col == "Gross" else LAYER_COLORS.get(col, "#94A3B8")
        ax.annotate(f" {col} {v:+.1%}", (dates[-1], y), color=color,
                    fontsize=8.5, va="center", fontweight="bold")
    ax.axhline(0, color="#9CA3AF", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(dates[0], dates[-1] + (dates[-1] - dates[0]) * 0.24)
    last_num = mdates.date2num(dates[-1])
    ax.set_xticks([t for t in ax.get_xticks() if t <= last_num])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    if title:
        ax.set_title(title, loc="left")


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
            as_of = getattr(lineage, "data_as_of", None)
            note = f"  as_of={as_of}" if as_of else ""
            print(f"OK  {slug}  ({len(payload):,} bytes){note}  → {path.name}")
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
    contrib["Residual"] = (
        stock_history[l3_rr] if l3_rr else stock_history[gross_col] - stock_history[l3_cfr]
    )
    contrib["Gross"] = stock_history[gross_col]

    cumulative = cum_with_t0(contrib, stock_history["date"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    attribution_lines(ax1, cumulative,
                      title=f"{ticker} — cumulative return attribution ({years}y, arithmetic)")
    totals = contrib[["Market", "Sector", "Subsector", "Residual"]].sum()
    waterfall(ax2, totals.to_dict(), total_label="Gross", title="Same window, totalled")
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{ticker}_attribution_{years}y.png", no_show=no_show)

    gap = (
        contrib[["Market", "Sector", "Subsector", "Residual"]].sum(axis=1) - contrib["Gross"]
    ).abs()
    print("Mean absolute daily identity gap:", f"{gap.mean():.6%}")

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
    print_frame(f"{ticker} hedge ticket (legs stop at subsector)", hedge_display)

    # Signed ER bar — a negative layer is information, not error (see L* below)
    er_plot = er.dropna()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = list(er_plot.index)[::-1]
    values = list(er_plot.values)[::-1]
    colors = [LAYER_COLORS.get(k, "#94A3B8") if v >= 0 else "#D97706" for k, v in zip(labels, values)]
    ax.barh(labels, values, color=colors, height=0.58)
    for k, v in zip(labels, values):
        ax.annotate(f"{v:+.1%}", (v, k), ha="left" if v >= 0 else "right",
                    va="center", fontsize=9.5, fontweight="bold",
                    color="#334155" if v >= 0 else "#B45309",
                    xytext=(4 if v >= 0 else -4, 0), textcoords="offset points")
    ax.axvline(0, color="#9CA3AF", lw=0.8)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Share of variance (signed covariance share)")
    ax.set_title(f"{ticker} — current L3 Orth Risk decomposition (layers sum to {er_plot.sum():.0%})", loc="left")
    ax.margins(x=0.14)
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{ticker}_l3_er.png", no_show=no_show)
    if (er_plot < 0).any():
        neg = ", ".join(er_plot[er_plot < 0].index)
        print(f"Negative layer ({neg}) — signed covariance share; the L* section shows the response.")


def run_lstar(
    session: Any,
    base_url: str,
    ticker: str,
    notional: float,
    output_dir: Path,
    *,
    no_show: bool,
) -> None:
    """L*: the API dispatches the shallowest hedge depth (L1/L2/L3) whose marginal
    ER clears a threshold. Hedge legs stop at subsector; FF2 has no ETF hedge and
    is reported as a variance-ratio diagnostic on the L* residual."""
    print(f"\n# Part I·b · L* hedge-depth dispatch — {ticker}")
    lstar_body, _ = api_get(session, base_url, "/lstar", ticker=ticker)
    lstar_df = pd.DataFrame(
        {
            "date": pd.to_datetime(lstar_body["dates"]),
            "level": lstar_body["lstar"],
            "residual_return": lstar_body["residual_return"],
        }
    )
    level_now = lstar_df["level"].dropna().iloc[-1]
    print(
        f"L* today for {ticker}: {level_now}  "
        f"(marginal-ER threshold {lstar_body.get('threshold_used', 0.01):.0%})"
    )

    dec, _ = api_post(session, base_url, "/decompose", ticker=ticker)
    hedge_levels = dec.get("hedge_levels") or {}
    menu_rows = []
    for lvl in ["L1", "L2", "L3"]:
        blk = hedge_levels.get(lvl) or {}
        etfs = blk.get("hedge_etfs") or {}
        legs = []
        for leg_name in ["market", "sector", "subsector"]:
            hr = blk.get(f"{leg_name}_hr")
            etf = etfs.get(leg_name)
            if hr is not None and etf:
                side = "short" if hr < 0 else "long"
                legs.append(f"{side} ${abs(hr) * notional:,.0f} {etf}")
        res_er = blk.get("residual_er")
        menu_rows.append(
            {
                "Depth": lvl + ("  <- L*" if lvl == hedge_levels.get("recommended_level") else ""),
                "Residual ER": f"{res_er:.1%}" if res_er is not None else "—",
                f"Ticket on ${notional:,.0f}": " · ".join(legs) or "—",
            }
        )
    print_frame(f"{ticker} hedge-depth menu", pd.DataFrame(menu_rows).set_index("Depth"))

    style_ev = (dec.get("style") or {}).get("explained_variance")
    ss_ev = (dec.get("stock_specific") or {}).get("explained_variance")
    if style_ev is not None and ss_ev is not None:
        print(
            f"FF2 split of the L* residual: style {style_ev:.1%} + stock-specific {ss_ev:.1%} "
            f"≈ {style_ev + ss_ev:.1%} variance share (variance-ratio convention, L* basis). "
            "No ETF hedge on FF2 — the stock-specific share is the skills-paper skill basis."
        )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [4, 1.1]}
    )
    cum_resid = cum_with_t0(
        lstar_df[["residual_return"]].rename(columns={"residual_return": "L* residual"}),
        lstar_df["date"],
    )
    ax1.plot(cum_resid.index, cum_resid["L* residual"],
             color=LAYER_COLORS["L* residual"], lw=1.8)
    ax1.annotate(
        f" {cum_resid['L* residual'].iloc[-1]:+.1%}",
        (cum_resid.index[-1], cum_resid["L* residual"].iloc[-1]),
        color=LAYER_COLORS["L* residual"], fontsize=9, fontweight="bold", va="center",
    )
    ax1.axhline(0, color="#9CA3AF", lw=0.8)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_title(f"{ticker} — cumulative L* residual — the skills-paper ranking feature (starts at 0)", loc="left")

    level_num = lstar_df["level"].map({"L1": 1, "L2": 2, "L3": 3})
    ax2.fill_between(lstar_df["date"], 0.5, level_num, step="post", color="#1D6FA8", alpha=0.25)
    ax2.step(lstar_df["date"], level_num, where="post", color="#1D6FA8", lw=1.6)
    ax2.set_yticks([1, 2, 3], ["L1", "L2", "L3"])
    ax2.set_ylim(0.5, 3.5)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Dispatched hedge depth", loc="left", fontsize=10)
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{ticker}_lstar.png", no_show=no_show)


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

    # The FF2 style layer: prefer the returns block, fall back to the history rows
    history_rows = ((fund_snapshot.get("portfolio_history") or {}).get("rows") or [])
    style_val = fund_returns.get("style")
    if style_val is None and history_rows:
        hist_tmp = pd.DataFrame(history_rows).dropna(subset=["portfolio_gross_return"])
        if not hist_tmp.empty and "portfolio_style_return" in hist_tmp.columns:
            style_val = hist_tmp.iloc[-1]["portfolio_style_return"]

    steps = {
        "Market": fund_returns.get("market"),
        "Sector": fund_returns.get("sector"),
        "Subsector": fund_returns.get("subsector"),
        "Style (FF2)": style_val,
        "Final residual": fund_returns.get("idiosyncratic"),
    }
    steps = {k: v for k, v in steps.items() if v is not None}
    gross = fund_returns.get("gross")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    waterfall(ax, steps, total_label="Sum",
              title=f"{fund_label} — latest month, holdings-derived return decomposition")
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{safe}_latest_decomp.png", no_show=no_show)
    if gross is not None:
        layer_sum = sum(steps.values())
        print(
            f"Gross {gross:.2%} | layer sum {layer_sum:.2%} | identity residual "
            f"{gross - layer_sum:+.3%} (holdings-aggregation remainder)"
        )

    fund_history = pd.DataFrame(history_rows)
    if fund_history.empty:
        print("No portfolio history in this fund snapshot.")
    else:
        fund_history["teo"] = pd.to_datetime(fund_history["teo"])
        fund_history = fund_history.sort_values("teo").reset_index(drop=True)
        fund_history = fund_history.dropna(subset=["portfolio_gross_return"]).reset_index(drop=True)
        rename = {
            "portfolio_market_return": "Market",
            "portfolio_sector_return": "Sector",
            "portfolio_subsector_return": "Subsector",
            "portfolio_style_return": "Style (FF2)",
            "portfolio_idiosyncratic_return": "Final residual",
            "portfolio_gross_return": "Gross",
        }
        available = {k: v for k, v in rename.items() if k in fund_history.columns}
        fund_attr = fund_history[list(available)].rename(columns=available)

        cum_fund = cum_with_t0(fund_attr, fund_history["teo"])
        fig, ax = plt.subplots(figsize=(11.5, 5))
        attribution_lines(
            ax, cum_fund,
            title=f"{fund_label} — cumulative fund attribution (arithmetic, anchored at 0)",
        )
        fig.tight_layout()
        save_or_show(fig, output_dir / f"{safe}_attribution.png", no_show=no_show)

        if "Final residual" in fund_attr:
            resid = fund_attr["Final residual"].dropna()
            persistence = {
                "Observed months": float(len(resid)),
                "Positive final-residual months": (resid > 0).mean() if len(resid) else np.nan,
                "Cumulative final residual (selection)": resid.sum(),
            }
            if "Style (FF2)" in fund_attr:
                persistence["Cumulative style (FF2) contribution"] = fund_attr["Style (FF2)"].sum()
            print_frame(f"{fund_label} final-residual persistence", pd.DataFrame({"Value": persistence}))

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
    print_frame(f"{fund_label} hedge ticket (basket stops at subsector)", disp)

    plotted = fund_hedge.dropna(subset=["HR ($ ETF / $1 fund)"]).copy()
    top = plotted.reindex(
        plotted["HR ($ ETF / $1 fund)"].abs().sort_values(ascending=False).index
    ).head(12)
    rest = plotted.drop(top.index)
    top = top.iloc[::-1]

    level_color = {"L1": LAYER_COLORS["Market"], "L2": LAYER_COLORS["Sector"], "L3": LAYER_COLORS["Subsector"]}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = (top["Level"] + " · " + top["ETF"].astype(str)).tolist()
    values = top["HR ($ ETF / $1 fund)"].tolist()
    colors = top["Level"].map(level_color).tolist()
    if not rest.empty:
        labels = [f"Other ({len(rest)} legs, net)"] + labels
        values = [rest["HR ($ ETF / $1 fund)"].sum()] + values
        colors = ["#CBD5E1"] + colors
    ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(0, color="#9CA3AF", lw=0.8)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ETF notional per $1 of fund exposure")
    ax.set_title(f"{fund_label} — largest hedge legs (top 12 of {len(plotted)} by |HR|)", loc="left")
    fig.tight_layout()
    save_or_show(fig, output_dir / f"{safe}_hedge_basket.png", no_show=no_show)


def run_book(
    client: RiskModelsClient, book: dict[str, float], output_dir: Path, *, no_show: bool
) -> None:
    print("\n# Part III · Bring your book")
    print("Book:", book)
    pa = client.analyze_portfolio(book, metrics=["full_metrics", "hedge_ratios"], years=1)
    er_map = dict(pa.portfolio_l3_er_weighted_mean or {})
    print_frame("Portfolio L3 ER (weight-mean)", pd.DataFrame([er_map]))
    er_steps = {
        "Market": er_map.get("l3_market_er"),
        "Sector": er_map.get("l3_sector_er"),
        "Subsector": er_map.get("l3_subsector_er"),
        "Residual": er_map.get("l3_residual_er"),
    }
    er_steps = {k: float(v) for k, v in er_steps.items() if v is not None and pd.notna(v)}
    if er_steps:
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        waterfall(ax, er_steps, total_label="Total",
                  title="Demo book — weight-mean L3 explained risk")
        fig.tight_layout()
        save_or_show(fig, output_dir / "demo_book_l3_er.png", no_show=no_show)
    print_frame(
        "Portfolio hedge ratios",
        pd.DataFrame([pa.portfolio_hedge_ratios]),
    )
    try:
        pdf_bytes, _ = client.post_portfolio_risk_snapshot_pdf(
            book, title="Demo book"
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

    # Connect quietly when the key is already discoverable; interactive prompt otherwise.
    env_ready = bool(os.environ.get("RISKMODELS_API_KEY")) or any(
        (base / name).exists()
        for base in (Path.cwd(), Path.cwd().parent)
        for name in (".env.local", ".env")
    )
    with contextlib.redirect_stderr(io.StringIO()) if env_ready else contextlib.nullcontext():
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
            "offer": "api-only (panels + PDFs + metrics/L*/funds; no zarr)",
        }
    )

    if not args.skip_stock:
        run_stock(
            client,
            args.stock,
            args.years,
            args.notional,
            args.output_dir,
            no_show=args.no_show,
        )
    if not args.skip_stock and not args.skip_lstar:
        try:
            run_lstar(
                session,
                base_url,
                args.stock,
                args.notional,
                args.output_dir,
                no_show=args.no_show,
            )
        except Exception as exc:  # noqa: BLE001 — demo resilience
            print(f"SKIP L* section: {exc}")
    if not args.skip_panels:
        run_panels(client, args.stock, args.output_dir)
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
        run_book(client, DEFAULT_BOOK, args.output_dir, no_show=args.no_show)

    print(
        "\nDone. Orth Risk is a hosted-model candidate alongside Barra — signed "
        "hierarchical ER, ETF hedge tickets (legs stop at subsector), L* depth "
        "dispatch, and the FF2/final-residual fund view. Zarr history later if permissioned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
