#!/usr/bin/env python3
"""Streamlit liquidity map for stock charts."""

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
from liquidity_map.paper_broker import get_position_info, get_position_qty, portfolio_value
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
    st.caption("Volume profile, liquidity heatmap, signals, and paper auto-trade — powered by yfinance.")

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
        show_signals = st.checkbox("Buy / sell signals on chart", value=True)
        show_signal_price_lines = st.checkbox("Lines at buy/sell prices", value=True)
        st.caption("Green ▲ BUY at support · Red ▼ SELL at resistance")
        n_bins = st.slider("Price bins", min_value=50, max_value=200, value=100, step=10)

        st.header("Price line")
        show_price_line = st.checkbox("Draw price line", value=False)
        price_line_value = st.number_input("Price ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        price_line_label = st.text_input("Label", value="Target")
        show_position_line = st.checkbox("Position line (avg entry)", value=True)

        st.header("Paper auto-trade")
        auto_trade_enabled = st.checkbox("Enable auto-trade", value=False)
        trade_cfg = load_trade_config()
        if auto_trade_enabled:
            st.caption("Simulated trades at last close — no broker login needed.")
            signal_only = st.checkbox("Signal only (no paper trades)", value=trade_cfg.dry_run)
            trade_amount = st.number_input("USD per buy", min_value=10.0, max_value=10000.0, value=trade_cfg.trade_amount_usd, step=10.0)
            min_strength = st.slider("Min signal strength", 1, 3, trade_cfg.min_strength)
            max_daily = st.number_input("Max trades per day", min_value=1, max_value=20, value=trade_cfg.max_daily_trades)
            poll_seconds = st.selectbox("Check interval", [30, 60, 120, 300], index=1)
            trade_config = TradeConfig(
                dry_run=signal_only,
                trade_amount_usd=trade_amount,
                min_strength=min_strength,
                max_daily_trades=max_daily,
            )
        else:
            trade_config = None
            poll_seconds = 60

    if not ticker:
        st.warning("Enter a ticker symbol.")
        return

    try:
        with st.spinner(f"Loading {ticker}..."):
            df = fetch_bars(ticker, period=period, interval=interval)
    except Exception as exc:
        st.error(f"Could not load data for {ticker}: {exc}")
        return

    profile = build_volume_profile(df, n_bins=n_bins)
    signals = detect_liquidity_signals(df, profile) if show_signals else []
    buy_count = sum(1 for s in signals if s.side == "buy")
    sell_count = sum(1 for s in signals if s.side == "sell")
    last_price = float(df["close"].iloc[-1])

    if show_price_line and price_line_value <= 0:
        price_line_value = last_price

    price_lines: list[PriceLine] = []
    if show_price_line and price_line_value > 0:
        price_lines.append(PriceLine(price=price_line_value, label=f"{price_line_label} ${price_line_value:.2f}"))

    quick_cols = st.columns(4)
    if quick_cols[0].button("Line @ Last"):
        price_lines.append(PriceLine(price=last_price, label=f"Last ${last_price:.2f}", color="#a78bfa"))
    if quick_cols[1].button("Line @ POC"):
        price_lines.append(PriceLine(price=profile.poc_price, label=f"POC ${profile.poc_price:.2f}", color="#f59e0b"))
    if quick_cols[2].button("Line @ VAH"):
        price_lines.append(PriceLine(price=profile.vah_price, label=f"VAH ${profile.vah_price:.2f}", color="#60a5fa"))
    if quick_cols[3].button("Line @ VAL"):
        price_lines.append(PriceLine(price=profile.val_price, label=f"VAL ${profile.val_price:.2f}", color="#60a5fa"))

    trade_state = load_trade_state()
    portfolio = get_paper_portfolio(trade_state, symbol=ticker)
    pos_qty = get_position_qty(portfolio, ticker)
    position = get_position_info(portfolio, ticker, last_price)
    equity = portfolio_value(portfolio, {ticker: last_price})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("POC", f"${profile.poc_price:.2f}")
    col2.metric("Value Area High", f"${profile.vah_price:.2f}")
    col3.metric("Value Area Low", f"${profile.val_price:.2f}")
    if position:
        pnl_delta = f"${position.pnl:+,.2f} ({position.pnl_pct:+.1f}%)"
        col4.metric("Position", f"${position.avg_price:.2f}", delta=pnl_delta, delta_color="normal")
    elif auto_trade_enabled:
        col4.metric("Paper equity", f"${equity:,.2f}", delta=f"{pos_qty:.2f} {ticker} shares")
    elif show_signals:
        col4.metric("Signals", f"{buy_count} buy / {sell_count} sell")
    else:
        col4.metric("Last", f"${last_price:.2f}")

    fig = build_chart(
        df,
        ticker,
        ChartOptions(
            show_profile=show_profile,
            show_heatmap=show_heatmap,
            show_poc_lines=show_poc_lines,
            show_hvn_lvn=show_hvn_lvn,
            show_signals=show_signals,
            show_signal_price_lines=show_signal_price_lines,
            show_paper_trades=auto_trade_enabled,
            n_bins=n_bins,
            quote=None,
            paper_trades=trade_state.trade_log if auto_trade_enabled else None,
            price_lines=tuple(price_lines),
            position=position,
            show_position_line=show_position_line,
        ),
    )
    st.plotly_chart(fig, width="stretch")

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
        st.dataframe(signal_rows, width="stretch", hide_index=True)

    actionable = get_actionable_signal(df, profile) if show_signals else None
    if actionable:
        st.info(f"Latest actionable signal: **{actionable.side.upper()}** — {actionable.reason} (strength {actionable.strength})")

    if auto_trade_enabled:
        _render_auto_trade(ticker, df, trade_config, poll_seconds, period, interval)

    with st.expander("How to read this chart"):
        st.markdown(
            """
            - **Volume profile (right):** Wider bars = more shares traded at that price (historically liquid).
            - **Heatmap (behind candles):** Brighter zones = liquidity clustered at that time and price.
            - **POC / VAH / VAL:** Key liquidity levels from the volume profile.
            - **Green triangles (buy):** Bullish bounce off liquid support; POC reclaim; LVN→HVN breakout.
            - **Red triangles (sell):** Bearish rejection at liquid resistance; POC loss; LVN→HVN breakdown.
            - **Paper auto-trade:** Simulates trades on liquidity signals — no broker account required.
            """
        )


def _make_auto_trade_loop(poll_seconds: int):
    @st.fragment(run_every=timedelta(seconds=poll_seconds))
    def _loop(ticker: str, period: str, interval: str, config: TradeConfig) -> None:
        df = fetch_bars(ticker, period=period, interval=interval)
        result = evaluate_and_trade(ticker, df, config=config)
        st.session_state["last_trade_result"] = {
            "action": result.action,
            "message": result.message,
            "signal": result.signal.reason if result.signal else None,
        }

    return _loop


def _render_auto_trade(
    ticker: str,
    df,
    config: TradeConfig | None,
    poll_seconds: int,
    period: str,
    interval: str,
) -> None:
    st.subheader("Paper auto-trade")
    if config is None:
        return

    st.caption(f"Paper mode | Checks every {poll_seconds}s during market hours")

    if st.button("Run trade check now"):
        with st.spinner("Evaluating signal..."):
            result = evaluate_and_trade(ticker, df, config=config)
        st.session_state["last_trade_result"] = {
            "action": result.action,
            "message": result.message,
            "signal": result.signal.reason if result.signal else None,
        }

    if st.checkbox("Auto-poll (background)", value=False):
        _make_auto_trade_loop(poll_seconds)(ticker, period, interval, config)

    last = st.session_state.get("last_trade_result")
    if last:
        st.write(f"Last check: **{last['action'].upper()}** — {last['message']}")
        if last.get("signal"):
            st.caption(f"Signal: {last['signal']}")

    state = load_trade_state()
    st.write(f"Paper cash: **${state.paper_cash:,.2f}**")
    if state.trade_log:
        st.markdown("**Recent paper trades**")
        st.dataframe(list(reversed(state.trade_log[-10:])), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()