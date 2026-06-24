#!/usr/bin/env python3
"""Streamlit liquidity map for stock charts."""

from __future__ import annotations

import streamlit as st

from liquidity_map.chart import ChartOptions, build_chart
from liquidity_map.data import (
    auto_interval,
    fetch_bars,
    fetch_quote,
    load_env_credentials,
    login_robinhood,
)
from liquidity_map.liquidity_score import quote_rating
from liquidity_map.profile import build_volume_profile
from liquidity_map.signals import detect_liquidity_signals

PERIOD_OPTIONS = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
}

INTERVAL_OPTIONS = {
    "Auto": None,
    "1 min": "1m",
    "5 min": "5m",
    "15 min": "15m",
    "1 hour": "1h",
    "1 day": "1d",
}


def main() -> None:
    st.set_page_config(page_title="Liquidity Map", page_icon="📊", layout="wide")
    st.title("Liquidity Map")
    st.caption("See where liquidity clusters on the price chart — volume profile, heatmap, and live spread.")

    with st.sidebar:
        st.header("Chart")
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
        period_label = st.selectbox("Range", list(PERIOD_OPTIONS.keys()), index=3)
        period = PERIOD_OPTIONS[period_label]
        interval_label = st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=0)
        interval = INTERVAL_OPTIONS[interval_label] or auto_interval(period)

        st.header("Liquidity overlays")
        show_profile = st.checkbox("Volume profile", value=True)
        show_heatmap = st.checkbox("Time × price heatmap", value=True)
        show_poc_lines = st.checkbox("POC + value area lines", value=True)
        show_hvn_lvn = st.checkbox("HVN / LVN shading", value=False)
        show_signals = st.checkbox("Buy / sell signals", value=True)
        n_bins = st.slider("Price bins", min_value=50, max_value=200, value=100, step=10)

        st.header("Live spread (optional)")
        use_robinhood = st.checkbox("Use Robinhood bid/ask", value=False)
        rh_logged_in = st.session_state.get("rh_logged_in", False)

        if use_robinhood:
            env_user, env_pass = load_env_credentials()
            username = st.text_input("Robinhood email", value=env_user or "")
            password = st.text_input("Robinhood password", type="password", value=env_pass or "")
            if st.button("Login to Robinhood"):
                with st.spinner("Logging in..."):
                    ok = login_robinhood(username, password)
                st.session_state["rh_logged_in"] = ok
                rh_logged_in = ok
                if ok:
                    st.success("Connected")
                else:
                    st.error("Login failed — check credentials")

    if not ticker:
        st.warning("Enter a ticker symbol.")
        return

    try:
        with st.spinner(f"Loading {ticker}..."):
            df = fetch_bars(ticker, period=period, interval=interval)
    except Exception as exc:
        st.error(f"Could not load data for {ticker}: {exc}")
        return

    quote = None
    if use_robinhood and rh_logged_in:
        try:
            quote = fetch_quote(ticker)
        except Exception as exc:
            st.sidebar.warning(f"Quote fetch failed: {exc}")

    profile = build_volume_profile(df, n_bins=n_bins)
    signals = detect_liquidity_signals(df, profile) if show_signals else []
    buy_count = sum(1 for s in signals if s.side == "buy")
    sell_count = sum(1 for s in signals if s.side == "sell")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("POC", f"${profile.poc_price:.2f}")
    col2.metric("Value Area High", f"${profile.vah_price:.2f}")
    col3.metric("Value Area Low", f"${profile.val_price:.2f}")
    if quote:
        rating = quote_rating(quote)
        spread_display = f"{rating.spread_pct:.3f}%" if rating.spread_pct is not None else "n/a"
        col4.metric("Spread", spread_display, delta=rating.label, delta_color="off")
    elif show_signals:
        col4.metric("Signals", f"{buy_count} buy / {sell_count} sell")
    else:
        col4.metric("Bars", str(len(df)))

    fig = build_chart(
        df,
        ticker,
        ChartOptions(
            show_profile=show_profile,
            show_heatmap=show_heatmap,
            show_poc_lines=show_poc_lines,
            show_hvn_lvn=show_hvn_lvn,
            show_signals=show_signals,
            n_bins=n_bins,
            quote=quote,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_signals and signals:
        st.subheader("Liquidity signals")
        signal_rows = [
            {
                "Time": s.datetime,
                "Side": s.side.upper(),
                "Price": f"${s.price:.2f}",
                "Reason": s.reason,
                "Strength": s.strength,
            }
            for s in signals
        ]
        st.dataframe(signal_rows, use_container_width=True, hide_index=True)

    with st.expander("How to read this chart"):
        st.markdown(
            """
            - **Volume profile (right):** Wider bars = more shares traded at that price (historically liquid).
            - **Heatmap (behind candles):** Brighter zones = liquidity clustered at that time and price.
            - **POC:** Point of Control — the price with the highest traded volume.
            - **VAH / VAL:** Value Area High/Low — bounds containing ~70% of volume around the POC.
            - **Spread badge:** Tight spread = easier to buy/sell near the quoted price right now.
            - **Green triangles (buy):** Bullish bounce off VAL, POC, or HVN support; POC reclaim; LVN→HVN breakout.
            - **Red triangles (sell):** Bearish rejection at VAH, POC, or HVN resistance; POC loss; LVN→HVN breakdown.
            """
        )


if __name__ == "__main__":
    main()