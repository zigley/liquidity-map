#!/usr/bin/env python3
"""Streamlit liquidity map — chart and backtest."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from liquidity_map.auto_trader import (
    evaluate_and_trade,
    get_paper_portfolio,
    load_trade_config,
    load_trade_state,
)
from liquidity_map.backtest import BacktestResult, run_backtest
from liquidity_map.chart import ChartOptions, PriceLine, build_chart
from liquidity_map.data import auto_interval, fetch_bars
from liquidity_map.paper_broker import get_position_info
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


@st.cache_data(show_spinner=False)
def cached_backtest(
    df_key: str,
    rolling_window: int,
    min_strength: int,
    use_rolling_profile: bool,
    h5: bool,
    h10: bool,
    h20: bool,
) -> BacktestResult:
    df = pd.read_json(df_key)
    df["datetime"] = pd.to_datetime(df["datetime"])
    horizons: list[int] = []
    if h5:
        horizons.append(5)
    if h10:
        horizons.append(10)
    if h20:
        horizons.append(20)
    if not horizons:
        horizons = [10]
    return run_backtest(
        df,
        rolling_window=rolling_window,
        min_strength=min_strength,
        horizons=tuple(horizons),
        use_rolling_profile=use_rolling_profile,
    )


def _df_cache_key(df: pd.DataFrame) -> str:
    return df.to_json(date_format="iso")


def render_chart_tab(
    df: pd.DataFrame,
    ticker: str,
    *,
    simple_view: bool,
    show_profile: bool,
    show_poc: bool,
    show_value_area: bool,
    show_signals: bool,
    min_signal_strength: int,
    show_heatmap: bool,
    show_volume: bool,
    show_price_line: bool,
    price_line_value: float,
    auto_trade: bool,
) -> None:
    profile = build_volume_profile(df, n_bins=80)
    last_price = float(df["close"].iloc[-1])
    all_signals = detect_liquidity_signals(df, profile) if show_signals else []
    signals = [s for s in all_signals if s.strength >= min_signal_strength] if show_signals else []

    trade_state = load_trade_state()
    portfolio = get_paper_portfolio(trade_state, symbol=ticker)
    position = get_position_info(portfolio, ticker, last_price)

    c1, c2, c3 = st.columns(3)
    c1.metric("Last price", f"${last_price:.2f}")
    c2.metric("POC", f"${profile.poc_price:.2f}", help="Price with the most traded volume")
    if position:
        c3.metric("Your position", f"{position.qty:.2f} shares", delta=f"${position.pnl:+.0f}")
    elif show_signals:
        c3.metric("Signals", f"{sum(1 for s in signals if s.side == 'buy')} buy · {sum(1 for s in signals if s.side == 'sell')} sell")
    else:
        c3.metric("Value area", f"${profile.val_price:.0f} – ${profile.vah_price:.0f}")

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
            show_value_area=show_value_area,
            show_signals=show_signals,
            min_signal_strength=min_signal_strength,
            show_volume=show_volume,
            price_lines=price_lines,
            position=position,
            show_position_line=position is not None,
        ),
    )
    st.plotly_chart(fig, width="stretch")

    if simple_view:
        st.info(
            "**How to read this:** The gray bars on the right show where volume traded at each price. "
            "The orange line is **POC** — the busiest price. "
            "▲ green = buy at support · ▼ red = sell at resistance. Hover any marker for details."
        )
    elif show_signals and signals:
        with st.expander(f"Signal list ({len(signals)})"):
            st.dataframe(
                [{"Side": s.side.upper(), "Price": f"${s.price:.2f}", "Strength": s.strength, "Reason": s.reason} for s in signals[-15:]],
                width="stretch",
                hide_index=True,
            )

    if auto_trade:
        cfg = load_trade_config()
        st.caption("Paper trading — set `AUTO_TRADE_DRY_RUN=true` in `.env` to preview signals without trading.")
        if st.button("Check signal now"):
            result = evaluate_and_trade(ticker, df, config=cfg)
            st.write(f"**{result.action.upper()}** — {result.message}")


def render_backtest_tab(df: pd.DataFrame, ticker: str) -> None:
    st.caption(
        "Walk-forward test: each signal uses only **past** bars to build the profile (no look-ahead). "
        "Entry at signal close; exit at close N bars later."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rolling_window = st.number_input("Profile lookback (bars)", min_value=10, max_value=252, value=60, step=5)
    with c2:
        bt_min_strength = st.slider("Min strength", 1, 3, 2, key="bt_strength")
    with c3:
        use_rolling = st.checkbox("Rolling profile (recommended)", value=True)
    with c4:
        st.write("Hold periods (bars)")
        h5 = st.checkbox("5 bars", value=True, key="h5")
        h10 = st.checkbox("10 bars", value=True, key="h10")
        h20 = st.checkbox("20 bars", value=True, key="h20")

    with st.spinner("Running backtest…"):
        result = cached_backtest(
            _df_cache_key(df),
            rolling_window,
            bt_min_strength,
            use_rolling,
            h5,
            h10,
            h20,
        )

    n = len(result.outcomes)
    n_buy = sum(1 for o in result.outcomes if o.side == "buy")
    n_sell = n - n_buy

    m1, m2, m3 = st.columns(3)
    m1.metric("Signals tested", n)
    m2.metric("Buys", n_buy)
    m3.metric("Sells", n_sell)

    if n == 0:
        st.warning("Not enough data for this settings combo. Try a longer range or lower min strength.")
        return

    primary_h = result.horizons[len(result.horizons) // 2] if result.horizons else 10
    primary = next((s for s in result.overall if s.horizon == primary_h), result.overall[0])

    s1, s2, s3 = st.columns(3)
    s1.metric(f"Win rate @ {primary.horizon} bars", f"{primary.win_rate:.1f}%")
    s2.metric(f"Avg return @ {primary.horizon} bars", f"{primary.avg_return_pct:+.2f}%")
    s3.metric(f"Median return @ {primary.horizon} bars", f"{primary.median_return_pct:+.2f}%")

    st.subheader("Results by hold period")

    def _stats_table(label: str, rows: tuple) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Group": label,
                    "Bars": s.horizon,
                    "Trades": s.n,
                    "Win %": round(s.win_rate, 1),
                    "Avg %": round(s.avg_return_pct, 2),
                    "Median %": round(s.median_return_pct, 2),
                }
                for s in rows
                if s.n > 0
            ]
        )

    tables = []
    for label, rows in [("All", result.overall), ("Buy", result.buy), ("Sell", result.sell)]:
        t = _stats_table(label, rows)
        if not t.empty:
            tables.append(t)
    if tables:
        st.dataframe(pd.concat(tables, ignore_index=True), width="stretch", hide_index=True)

    chart_df = pd.DataFrame(
        [{"Bars": s.horizon, "Win %": s.win_rate, "Avg return %": s.avg_return_pct} for s in result.overall if s.n > 0]
    )
    if not chart_df.empty:
        st.bar_chart(chart_df.set_index("Bars")[["Win %", "Avg return %"]])

    with st.expander(f"Trade log ({n} signals)"):
        log_rows = []
        for o in result.outcomes:
            row = {
                "Date": pd.Timestamp(o.datetime).strftime("%Y-%m-%d %H:%M"),
                "Side": o.side.upper(),
                "Entry": f"${o.entry_price:.2f}",
                "Strength": o.strength,
                "Reason": o.reason,
            }
            for h in result.horizons:
                if h in o.returns:
                    row[f"{h}bar %"] = f"{o.returns[h]:+.2f}%"
            log_rows.append(row)
        st.dataframe(log_rows, width="stretch", hide_index=True)

    if not use_rolling:
        st.warning(
            "Full-period profile mode includes future volume in the POC — historical results look better than live trading would."
        )


def main() -> None:
    st.set_page_config(page_title="Liquidity Map", page_icon="📊", layout="wide")
    st.title("Liquidity Map")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
        period = PERIOD_OPTIONS[st.selectbox("Range", list(PERIOD_OPTIONS.keys()), index=2)]
        interval = INTERVAL_OPTIONS[st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=0)]
        interval = interval or auto_interval(period)

        st.divider()
        simple_view = st.toggle("Simple view", value=True, help="Candles, volume profile, POC line, and strong signals only.")

        if simple_view:
            show_profile = True
            show_poc = True
            show_value_area = False
            show_signals = True
            min_signal_strength = 2
            show_heatmap = False
            show_volume = False
            show_price_line = False
            price_line_value = 0.0
            auto_trade = False
        else:
            with st.expander("Chart layers", expanded=True):
                show_profile = st.checkbox("Volume profile", value=True)
                show_poc = st.checkbox("POC / value area lines", value=True)
                show_value_area = show_poc
                show_signals = st.checkbox("Buy & sell markers", value=True)
                show_heatmap = st.checkbox("Liquidity heatmap", value=False)
                show_volume = st.checkbox("Volume bars", value=True)
                min_signal_strength = st.slider("Min signal strength", 1, 3, 1)

            show_price_line = st.checkbox("Custom price line", value=False)
            price_line_value = 0.0
            if show_price_line:
                price_line_value = st.number_input("Price ($)", min_value=0.0, step=0.01, format="%.2f")

            auto_trade = st.checkbox("Paper auto-trade", value=False)

    if not ticker:
        st.warning("Enter a ticker.")
        return

    try:
        df = fetch_bars(ticker, period=period, interval=interval)
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        return

    chart_tab, backtest_tab = st.tabs(["Chart", "Backtest"])

    with chart_tab:
        render_chart_tab(
            df,
            ticker,
            simple_view=simple_view,
            show_profile=show_profile,
            show_poc=show_poc,
            show_value_area=show_value_area,
            show_signals=show_signals,
            min_signal_strength=min_signal_strength,
            show_heatmap=show_heatmap,
            show_volume=show_volume,
            show_price_line=show_price_line,
            price_line_value=price_line_value,
            auto_trade=auto_trade,
        )

    with backtest_tab:
        render_backtest_tab(df, ticker)


if __name__ == "__main__":
    main()