#!/usr/bin/env python3
"""Stock buy/sell signals — plain English, one screen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from liquidity_map.auto_trader import evaluate_and_trade, get_paper_portfolio, load_trade_state
from liquidity_map.chart import ChartLines, build_chart
from liquidity_map.data import auto_interval, fetch_bars, is_crypto, resolve_ticker
from liquidity_map.model import (
    CRYPTO_TRAIL_PCT,
    LONG_TREND_MA,
    STRICTNESS_LABELS,
    backtest,
    build_config,
    live_advice,
    long_trend_status,
    scan_trades,
)
from liquidity_map.paper_broker import get_position_info

RANGES = {
    "1 Day": "1d",
    "1 Week": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
}

CRYPTO_PICKS = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
}


@st.cache_data(show_spinner=False)
def _load(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_bars(ticker, period=period, interval=interval)


@st.cache_data(show_spinner=False)
def _load_trend(ticker: str) -> pd.DataFrame:
    return fetch_bars(ticker, period="1y", interval="1d")


def _banner(advice) -> None:
    if advice.action == "buy":
        st.success(f"**BUY** — {advice.reason}")
    elif advice.action == "sell":
        st.error(f"**SELL** — {advice.reason}")
    else:
        st.info(f"**WAIT** — {advice.reason}")


def main() -> None:
    st.set_page_config(page_title="Stock Signals", layout="wide")
    st.title("Should I buy or sell?")

    with st.sidebar:
        st.caption("Stocks or crypto")
        pick = st.selectbox(
            "Quick pick",
            ["SPY", "Bitcoin (BTC)", "Ethereum (ETH)", "Custom symbol…"],
        )
        if pick == "Custom symbol…":
            ticker = st.text_input("Symbol", "SPY").strip().upper()
        elif pick == "SPY":
            ticker = "SPY"
            st.caption("Symbol: **SPY**")
        else:
            ticker = CRYPTO_PICKS[pick]
            st.caption(f"Symbol: **{ticker}**")
        period = RANGES[st.selectbox("How far back to look", list(RANGES.keys()), index=3)]

        st.divider()
        strictness = st.slider(
            "How picky should signals be?",
            min_value=1,
            max_value=5,
            value=3,
            help="Left = more trades. Right = fewer trades, but cleaner setups.",
        )
        st.caption(STRICTNESS_LABELS[strictness])

        st.divider()
        st.markdown(
            """
**Rules (plain English)**
- **Buy** only in a long-term uptrend (above 200-day average)
- **Buy** on a bounce off support when short-term trend is up
- **Sell** at profit target or if price drops from its high
- **Crypto** uses a wider stop ({:.1f}% vs stocks)
            """.format(CRYPTO_TRAIL_PCT)
        )

    if not ticker:
        st.warning("Type a stock symbol above.")
        return

    try:
        df = _load(ticker, period, auto_interval(period))
        trend_df = _load_trend(ticker)
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        return

    crypto = is_crypto(ticker)
    yahoo = resolve_ticker(ticker)
    if crypto:
        st.caption("Crypto — trades 24/7 · wider trail stop")
    elif yahoo != ticker:
        st.caption(f"Loading data as {yahoo}")

    cfg = build_config(ticker, period, len(df), strictness)
    markers = scan_trades(df, cfg, ticker=ticker)
    stats = backtest(df, cfg, ticker=ticker)
    trend_label, _, trend_ma = long_trend_status(trend_df)

    state = load_trade_state()
    portfolio = get_paper_portfolio(state, ticker)
    last = float(df["close"].iloc[-1])
    position = get_position_info(portfolio, ticker, last)
    in_pos = position is not None and position.qty > 0
    peak = float(state.peak_prices.get(ticker, position.avg_price if position else last))

    advice = live_advice(
        df,
        in_position=in_pos,
        entry_price=position.avg_price if position else None,
        peak_price=peak if in_pos else None,
        cfg=cfg,
        ticker=ticker,
        trend_df=trend_df,
    )

    _banner(advice)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price now", f"${advice.price:.2f}")
    c2.metric(f"Long trend ({LONG_TREND_MA}d)", trend_label, help=f"Average ${trend_ma:,.2f}")
    c3.metric("Busiest price", f"${advice.poc:.2f}")
    c4.metric("Support → Profit", f"${advice.val:.0f} → ${advice.vah:.0f}")
    if stats.trades:
        c5.metric("Past trades won", f"{stats.win_rate:.0f}%")
    else:
        c5.metric("Past trades", "None")

    st.plotly_chart(
        build_chart(
            df,
            ticker,
            markers,
            advice,
            ChartLines(
                entry=position.avg_price if in_pos else None,
                target=advice.target,
                stop=advice.stop,
            ),
        ),
        width="stretch",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("What the lines mean")
        trail_note = f"{cfg.trail_pct:.1f}% below the high"
        if crypto:
            trail_note += " (wider for crypto)"
        st.markdown(
            f"""
- **Orange — Busiest price:** where most trading happened
- **Gray — Support floor / Profit ceiling:** cheap vs expensive zone
- **Green — Take profit here**
- **Red — Sell if drops here:** {trail_note}
- **Long trend {trend_label}:** buys only when **Up** (above {LONG_TREND_MA}-day average)
            """
        )

    with right:
        st.subheader("Past trades")
        if not markers:
            st.write("No trades in this range. Try looser pickiness (1–2) or a longer date range.")
        else:
            closes = {pd.Timestamp(r.datetime): float(r.close) for r in df.itertuples(index=False)}
            buys = [m for m in markers if m.action == "buy"]
            sells = [m for m in markers if m.action == "sell"]
            rows = []
            for b, s in zip(buys, sells):
                bp, sp = closes.get(pd.Timestamp(b.datetime)), closes.get(pd.Timestamp(s.datetime))
                ret = f"{(sp - bp) / bp * 100:+.1f}%" if bp and sp and bp > 0 else "—"
                rows.append({"Buy": b.label, "Sell": s.label, "Made": ret})
            if len(buys) > len(sells):
                rows.append({"Buy": buys[-1].label, "Sell": "still open", "Made": "—"})
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption(
                f"{stats.trades} completed trades · avg {stats.avg_return_pct:+.1f}% · "
                f"total {stats.total_return_pct:+.1f}%"
            )

    with st.expander("Test a paper trade"):
        if st.button("Check now"):
            r = evaluate_and_trade(ticker, df)
            st.write(f"**{r.action.upper()}** — {r.message}")


if __name__ == "__main__":
    main()