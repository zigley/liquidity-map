#!/usr/bin/env python3
"""Streamlit liquidity map — simplified view."""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from liquidity_map.auto_trader import (
    TradeConfig,
    evaluate_and_trade,
    get_actionable_signal,
    get_paper_portfolio,
    load_trade_config,
    load_trade_state,
)
from liquidity_map.chart import ChartOptions, PriceLine, build_chart
from liquidity_map.data import auto_interval, fetch_bars
from liquidity_map.paper_broker import get_position_info, portfolio_value
from liquidity_map.profile import build_volume_profile
from liquidity_map.signals import detect_liquidity_signals

PERIOD_OPTIONS = {
    "1 Week": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
}

INTERVAL_OPTIONS = {
    "Auto": None,
    "Daily": "1d",
    "1 hour": "1h",
    "15 min": "15m",
}


def main() -> None:
    st.set_page_config(page_title="Liquidity Map", page_icon="📊", layout="wide")
    st.title("Liquidity Map")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
        period = PERIOD_OPTIONS[st.selectbox("Range", list(PERIOD_OPTIONS.keys()), index=2)]
        interval = INTERVAL_OPTIONS[st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=0)]
        interval = interval or auto_interval(period)

        st.divider()
        show_profile = st.checkbox("Volume profile", value=True)
        show_poc = st.checkbox("POC / value area lines", value=True)
        show_signals = st.checkbox("Buy & sell markers", value=True)
        show_heatmap = st.checkbox("Liquidity heatmap", value=False)

        st.divider()
        show_price_line = st.checkbox("Custom price line", value=False)
        price_line_value = 0.0
        if show_price_line:
            price_line_value = st.number_input("Price ($)", min_value=0.0, step=0.01, format="%.2f")

        st.divider()
        auto_trade = st.checkbox("Paper auto-trade", value=False)

    if not ticker:
        st.warning("Enter a ticker.")
        return

    try:
        df = fetch_bars(ticker, period=period, interval=interval)
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        return

    profile = build_volume_profile(df, n_bins=80)
    last_price = float(df["close"].iloc[-1])
    signals = detect_liquidity_signals(df, profile) if show_signals else []

    trade_state = load_trade_state()
    portfolio = get_paper_portfolio(trade_state, symbol=ticker)
    position = get_position_info(portfolio, ticker, last_price)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last", f"${last_price:.2f}")
    c2.metric("POC", f"${profile.poc_price:.2f}")
    c3.metric("Range", f"${profile.val_price:.2f} – ${profile.vah_price:.2f}")
    if position:
        c4.metric("Position", f"{position.qty:.2f} sh", delta=f"${position.pnl:+.0f}")
    else:
        c4.metric("Signals", f"{sum(1 for s in signals if s.side=='buy')} buy / {sum(1 for s in signals if s.side=='sell')} sell")

    price_lines: tuple[PriceLine, ...] = ()
    if show_price_line and price_line_value > 0:
        price_lines = (PriceLine(price=price_line_value, label=f"${price_line_value:.2f}"),)

    fig = build_chart(
        df,
        ticker,
        ChartOptions(
            show_profile=show_profile,
            show_heatmap=show_heatmap,
            show_poc_lines=show_poc,
            show_signals=show_signals,
            price_lines=price_lines,
            position=position,
            show_position_line=position is not None,
        ),
    )
    st.plotly_chart(fig, width="stretch")

    if show_signals and signals:
        with st.expander(f"Signal list ({len(signals)})"):
            st.dataframe(
                [{"Side": s.side.upper(), "Price": f"${s.price:.2f}", "Reason": s.reason} for s in signals[-15:]],
                width="stretch",
                hide_index=True,
            )

    if auto_trade:
        cfg = load_trade_config()
        st.caption("Paper trading — enable **Signal only** in `.env` with `AUTO_TRADE_DRY_RUN=true` to preview without trading.")
        if st.button("Check signal now"):
            result = evaluate_and_trade(ticker, df, config=cfg)
            st.write(f"**{result.action.upper()}** — {result.message}")

    st.caption("▲ green = buy at support · ▼ red = sell at resistance · Hover markers for details")


if __name__ == "__main__":
    main()