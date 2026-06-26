#!/usr/bin/env python3
"""Stock buy/sell signals — one model, one screen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from liquidity_map.auto_trader import evaluate_and_trade, get_paper_portfolio, load_trade_state
from liquidity_map.chart import ChartLines, build_chart
from liquidity_map.data import auto_interval, fetch_bars, resolve_ticker
from liquidity_map.model import DEFAULT_CONFIG, adapt_config, backtest, live_advice, scan_trades
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


def _action_banner(advice) -> None:
    colors = {"buy": "green", "sell": "red", "wait": "gray"}
    icons = {"buy": "🟢 BUY", "sell": "🔴 SELL", "wait": "⚪ WAIT"}
    st.markdown(
        f":{colors[advice.action]}[**{icons[advice.action]}** — {advice.reason}]"
    )


def main() -> None:
    st.set_page_config(page_title="Stock Signals", layout="wide")
    st.title("Stock Buy / Sell Signals")

    with st.sidebar:
        ticker = st.text_input("Ticker", "SPY").strip().upper()
        period = RANGES[st.selectbox("Range", list(RANGES.keys()), index=3)]
        interval = auto_interval(period)

        st.divider()
        st.markdown(
            """
**The model**
1. **Buy** — uptrend + bounce off VAL or POC
2. **Sell** — trail stop or VAH target
3. **Wait** — anything else
            """
        )

    if not ticker:
        st.warning("Enter a ticker.")
        return

    try:
        df = _load(ticker, period, interval)
    except Exception as exc:
        st.error(str(exc))
        return

    if resolve_ticker(ticker) != ticker:
        st.caption(f"Data via {resolve_ticker(ticker)}")

    cfg = adapt_config(period, len(df), DEFAULT_CONFIG)
    markers = scan_trades(df, cfg)

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

    _action_banner(advice)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${advice.price:.2f}")
    c2.metric("POC", f"${advice.poc:.2f}")
    c3.metric("VAL → VAH", f"${advice.val:.0f} – ${advice.vah:.0f}")
    stats = backtest(df, cfg)
    c4.metric("Backtest win %", f"{stats.win_rate:.0f}%" if stats.trades else "—")

    lines = ChartLines(
        entry=position.avg_price if in_pos else None,
        target=advice.target,
        stop=advice.stop,
    )
    st.plotly_chart(build_chart(df, ticker, markers, advice, lines), width="stretch")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("How to trade this")
        st.markdown(
            f"""
| Step | Rule |
|------|------|
| **Buy** | Price above MA, bounces off **VAL** or **POC** with rejection wick |
| **Target** | **VAH** (green line) — take profit |
| **Stop** | **Trail** (red line) — {cfg.trail_pct}% below peak |
| **Flat** | Wait for next **B** marker |
            """
        )

    with col_b:
        st.subheader("Past trades")
        if not markers:
            st.caption("No trades in this range.")
        else:
            closes = {pd.Timestamp(r.datetime): float(r.close) for r in df.itertuples(index=False)}
            rows = []
            buys = [m for m in markers if m.action == "buy"]
            sells = [m for m in markers if m.action == "sell"]
            for b, s in zip(buys, sells):
                bp, sp = closes.get(pd.Timestamp(b.datetime)), closes.get(pd.Timestamp(s.datetime))
                ret = f"{(sp - bp) / bp * 100:+.2f}%" if bp and sp and bp > 0 else "—"
                rows.append({"Buy": b.label, "Sell": s.label, "Return": ret, "Exit": s.reason[:40]})
            if len(buys) > len(sells) and buys:
                rows.append({"Buy": buys[-1].label, "Sell": "OPEN", "Return": "—", "Exit": "—"})
            st.dataframe(rows, width="stretch", hide_index=True)
            if stats.trades:
                st.caption(
                    f"{stats.trades} round trips · avg {stats.avg_return_pct:+.2f}% · "
                    f"total {stats.total_return_pct:+.2f}%"
                )

    with st.expander("Paper trade check"):
        if st.button("Run check now"):
            r = evaluate_and_trade(ticker, df)
            st.write(f"**{r.action.upper()}** — {r.message}")


if __name__ == "__main__":
    main()