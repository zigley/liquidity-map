#!/usr/bin/env python3
"""Stock buy/sell signals — plain English, one screen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from liquidity_map.auto_trader import evaluate_and_trade, get_paper_portfolio, load_trade_state
from liquidity_map.chart import ChartLines, build_chart
from liquidity_map.data import auto_interval, fetch_bars, resolve_ticker
from liquidity_map.model import (
    STRICTNESS_LABELS,
    adapt_config,
    backtest,
    config_for_strictness,
    live_advice,
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


@st.cache_data(show_spinner=False)
def _load(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_bars(ticker, period=period, interval=interval)


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
        ticker = st.text_input("Stock symbol", "SPY").strip().upper()
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
**In plain English**
- **Buy** when price bounces off a busy support level and trend is up
- **Sell** at your profit target, or if price drops from its high
- **Wait** otherwise — no guesswork
            """
        )

    if not ticker:
        st.warning("Type a stock symbol above.")
        return

    try:
        df = _load(ticker, period, auto_interval(period))
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        return

    yahoo = resolve_ticker(ticker)
    if yahoo != ticker:
        st.caption(f"Loading data as {yahoo}")

    cfg = adapt_config(period, len(df), config_for_strictness(strictness))
    markers = scan_trades(df, cfg)
    stats = backtest(df, cfg)

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
    )

    _banner(advice)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price now", f"${advice.price:.2f}")
    c2.metric("Busiest price", f"${advice.poc:.2f}", help="Where the most shares traded")
    c3.metric("Support → Profit", f"${advice.val:.0f} → ${advice.vah:.0f}")
    if stats.trades:
        c4.metric(
            "Past trades won",
            f"{stats.win_rate:.0f}%",
            help=f"{stats.trades} buy-then-sell trades in this range",
        )
    else:
        c4.metric("Past trades", "None in range")

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
        st.markdown(
            f"""
- **Orange — Busiest price:** where most volume traded
- **Gray — Support floor / Profit ceiling:** cheap zone vs expensive zone
- **Green — Take profit here:** sell when price reaches this (if you're in)
- **Red — Sell if drops here:** protects gains ({cfg.trail_pct:.1f}% below the high)
- **B1, S1…** — past buys and sells this model would have made
            """
        )

    with right:
        st.subheader("Past trades")
        if not markers:
            st.write("No trades matched your pickiness level in this date range. Try **Loose** (1–2) or a longer range.")
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
                f"At this pickiness: {stats.trades} completed trades, "
                f"average {stats.avg_return_pct:+.1f}% each"
            )

    with st.expander("Test a paper trade"):
        st.caption("Runs the same BUY / SELL / WAIT rules against your fake portfolio.")
        if st.button("Check now"):
            r = evaluate_and_trade(ticker, df)
            st.write(f"**{r.action.upper()}** — {r.message}")


if __name__ == "__main__":
    main()