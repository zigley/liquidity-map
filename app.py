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
from liquidity_map.backtest import BacktestResult, compare_backtest, run_backtest
from liquidity_map.chart import ChartOptions, PriceLine, build_chart
from liquidity_map.data import auto_interval, fetch_bars, resolve_ticker
from liquidity_map.paper_broker import get_position_info
from liquidity_map.profile import build_volume_profile
from liquidity_map.signals import (
    DEFAULT_SIGNAL_CONFIG,
    LEGACY_SIGNAL_CONFIG,
    SignalConfig,
    detect_liquidity_signals,
)

PERIOD_OPTIONS = {
    "1 Day": "1d",
    "1 Week": "5d",
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
    "Daily": "1d",
}


def _adapt_config_for_range(config: SignalConfig, period: str, n_bars: int) -> SignalConfig:
    """Shorter MA and cooldown for intraday (1-day) charts."""
    if period != "1d":
        return config
    ma = min(20, max(8, n_bars // 4))
    return SignalConfig(
        require_trend_filter=config.require_trend_filter,
        require_rejection_wick=config.require_rejection_wick,
        trend_ma_period=ma,
        volume_spike_pct=config.volume_spike_pct,
        min_volume_pct=config.min_volume_pct,
        min_confluence=config.min_confluence,
        cooldown_bars=max(2, config.cooldown_bars // 2),
        edge_only=config.edge_only,
        wick_ratio=config.wick_ratio,
    )


def _build_signal_config(
    *,
    require_trend: bool,
    require_wick: bool,
    min_confluence: int,
    edge_only: bool,
    legacy_mode: bool = False,
) -> SignalConfig:
    if legacy_mode:
        return SignalConfig(
            require_trend_filter=False,
            require_rejection_wick=False,
            min_confluence=1,
            edge_only=False,
            volume_spike_pct=0.8,
        )
    return SignalConfig(
        require_trend_filter=require_trend,
        require_rejection_wick=require_wick,
        min_confluence=min_confluence,
        edge_only=edge_only,
    )


@st.cache_data(show_spinner=False)
def cached_backtest(
    df: pd.DataFrame,
    rolling_window: int,
    use_rolling_profile: bool,
    require_trend: bool,
    require_wick: bool,
    min_confluence: int,
    edge_only: bool,
    legacy_mode: bool,
    h5: bool,
    h10: bool,
    h20: bool,
) -> BacktestResult:
    horizons: list[int] = []
    if h5:
        horizons.append(5)
    if h10:
        horizons.append(10)
    if h20:
        horizons.append(20)
    if not horizons:
        horizons = [10]
    config = _build_signal_config(
        require_trend=require_trend,
        require_wick=require_wick,
        min_confluence=min_confluence,
        edge_only=edge_only,
        legacy_mode=legacy_mode,
    )
    return run_backtest(
        df.copy(),
        rolling_window=rolling_window,
        signal_config=config,
        horizons=tuple(horizons),
        use_rolling_profile=use_rolling_profile,
    )


@st.cache_data(show_spinner=False)
def cached_compare(
    df: pd.DataFrame,
    rolling_window: int,
    use_rolling_profile: bool,
    h5: bool,
    h10: bool,
    h20: bool,
) -> tuple[BacktestResult, BacktestResult]:
    horizons: list[int] = []
    if h5:
        horizons.append(5)
    if h10:
        horizons.append(10)
    if h20:
        horizons.append(20)
    if not horizons:
        horizons = [10]
    return compare_backtest(
        df.copy(),
        rolling_window=rolling_window,
        horizons=tuple(horizons),
        use_rolling_profile=use_rolling_profile,
    )


def _primary_stats(result: BacktestResult) -> tuple[int, object]:
    primary_h = result.horizons[len(result.horizons) // 2] if result.horizons else 10
    primary = next((s for s in result.overall if s.horizon == primary_h), result.overall[0])
    return primary_h, primary


def render_chart_tab(
    df: pd.DataFrame,
    ticker: str,
    *,
    simple_view: bool,
    signal_config: SignalConfig,
    show_profile: bool,
    show_poc: bool,
    show_value_area: bool,
    show_signals: bool,
    show_heatmap: bool,
    show_volume: bool,
    show_price_line: bool,
    price_line_value: float,
    auto_trade: bool,
) -> None:
    profile = build_volume_profile(df, n_bins=80)
    last_price = float(df["close"].iloc[-1])
    all_signals = detect_liquidity_signals(df, profile, config=signal_config) if show_signals else []

    trade_state = load_trade_state()
    portfolio = get_paper_portfolio(trade_state, symbol=ticker)
    position = get_position_info(portfolio, ticker, last_price)

    c1, c2, c3 = st.columns(3)
    c1.metric("Last price", f"${last_price:.2f}")
    c2.metric("POC", f"${profile.poc_price:.2f}", help="Price with the most traded volume")
    if position:
        c3.metric("Your position", f"{position.qty:.2f} shares", delta=f"${position.pnl:+.0f}")
    elif show_signals:
        c3.metric(
            "Signals",
            f"{sum(1 for s in all_signals if s.side == 'buy')} buy · {sum(1 for s in all_signals if s.side == 'sell')} sell",
        )
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
            min_confluence=signal_config.min_confluence,
            signal_config=signal_config,
            show_volume=show_volume,
            price_lines=price_lines,
            position=position,
            show_position_line=position is not None,
        ),
    )
    st.plotly_chart(fig, width="stretch")

    if simple_view:
        st.info(
            "**Filtered signals** need confluence score ≥ 3 (trend, wick, edge, volume spike, HVN). "
            "Gray bars = volume profile · Orange = POC · Hover markers for score breakdown."
        )
    elif show_signals and all_signals:
        with st.expander(f"Signal list ({len(all_signals)})"):
            st.dataframe(
                [
                    {
                        "Side": s.side.upper(),
                        "Price": f"${s.price:.2f}",
                        "Score": f"{s.confluence}/5",
                        "Tags": ", ".join(s.tags),
                        "Reason": s.reason,
                    }
                    for s in all_signals[-15:]
                ],
                width="stretch",
                hide_index=True,
            )

    if auto_trade:
        cfg = load_trade_config()
        st.caption("Paper trading — set `AUTO_TRADE_DRY_RUN=true` in `.env` to preview signals without trading.")
        if st.button("Check signal now"):
            result = evaluate_and_trade(ticker, df, config=cfg)
            st.write(f"**{result.action.upper()}** — {result.message}")


def _render_backtest_results(result: BacktestResult, label: str | None = None) -> None:
    n = len(result.outcomes)
    if label:
        st.markdown(f"**{label}**")

    if n == 0:
        st.warning("No signals matched these rules.")
        return

    primary_h, primary = _primary_stats(result)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trades", n)
    m2.metric(f"Win rate @ {primary_h} bars", f"{primary.win_rate:.1f}%")
    m3.metric(f"Avg return @ {primary_h} bars", f"{primary.avg_return_pct:+.2f}%")
    m4.metric(f"Median @ {primary_h} bars", f"{primary.median_return_pct:+.2f}%")

    table = pd.DataFrame(
        [
            {
                "Group": "All",
                "Bars": s.horizon,
                "Trades": s.n,
                "Win %": round(s.win_rate, 1),
                "Avg %": round(s.avg_return_pct, 2),
            }
            for s in result.overall
            if s.n > 0
        ]
    )
    if not table.empty:
        st.dataframe(table, width="stretch", hide_index=True)


def render_backtest_tab(df: pd.DataFrame, ticker: str, signal_config: SignalConfig) -> None:
    st.caption(
        "Walk-forward test using past bars only. Signals require confluence from: "
        "**edge level · trend · wick · volume spike · HVN**."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        default_lookback = min(60, max(15, len(df) // 2))
        rolling_window = st.number_input(
            "Profile lookback (bars)",
            min_value=10,
            max_value=252,
            value=default_lookback,
            step=5,
        )
    with c2:
        use_rolling = st.checkbox("Rolling profile (recommended)", value=True)
    with c3:
        compare_modes = st.checkbox("Compare strict vs legacy", value=True)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        bt_min_confluence = st.slider("Min confluence", 1, 5, signal_config.min_confluence, key="bt_conf")
    with c5:
        bt_trend = st.checkbox("Require trend", value=signal_config.require_trend_filter, key="bt_trend")
    with c6:
        bt_wick = st.checkbox("Require wick", value=signal_config.require_rejection_wick, key="bt_wick")
    with c7:
        st.write("Hold periods")
        h5 = st.checkbox("5", value=True, key="h5")
        h10 = st.checkbox("10", value=True, key="h10")
        h20 = st.checkbox("20", value=True, key="h20")

    with st.spinner("Running backtest…"):
        if compare_modes:
            strict, legacy = cached_compare(df, rolling_window, use_rolling, h5, h10, h20)
            col_a, col_b = st.columns(2)
            with col_a:
                _render_backtest_results(strict, "Strict (trend + wick + score ≥ 3)")
            with col_b:
                _render_backtest_results(legacy, "Legacy (old rules)")
        else:
            result = cached_backtest(
                df,
                rolling_window,
                use_rolling,
                bt_trend,
                bt_wick,
                bt_min_confluence,
                signal_config.edge_only,
                legacy_mode=False,
                h5=h5,
                h10=h10,
                h20=h20,
            )
            _render_backtest_results(result)

            with st.expander(f"Trade log ({len(result.outcomes)})"):
                log_rows = []
                for o in result.outcomes:
                    row = {
                        "Date": pd.Timestamp(o.datetime).strftime("%Y-%m-%d %H:%M"),
                        "Side": o.side.upper(),
                        "Entry": f"${o.entry_price:.2f}",
                        "Score": o.confluence,
                        "Reason": o.reason,
                    }
                    for h in result.horizons:
                        if h in o.returns:
                            row[f"{h}bar %"] = f"{o.returns[h]:+.2f}%"
                    log_rows.append(row)
                st.dataframe(log_rows, width="stretch", hide_index=True)

    if not use_rolling:
        st.warning("Full-period profile includes future volume — results look better than live trading would.")


def main() -> None:
    st.set_page_config(page_title="Liquidity Map", page_icon="📊", layout="wide")
    st.title("Liquidity Map")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
        period = PERIOD_OPTIONS[st.selectbox("Range", list(PERIOD_OPTIONS.keys()), index=3)]
        interval = INTERVAL_OPTIONS[st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=0)]
        interval = interval or auto_interval(period)

        st.divider()
        st.subheader("Signal filters")
        simple_view = st.toggle("Simple view", value=True)

        if simple_view:
            signal_config = DEFAULT_SIGNAL_CONFIG
            show_profile = True
            show_poc = True
            show_value_area = False
            show_signals = True
            show_heatmap = False
            show_volume = False
            show_price_line = False
            price_line_value = 0.0
            auto_trade = False
            st.caption("Score ≥ 3 from trend, wick, edge, volume, HVN · VAL/VAH/POC only")
        else:
            require_trend = st.checkbox("Require trend (50 MA)", value=True)
            require_wick = st.checkbox("Require rejection wick", value=True)
            min_confluence = st.slider("Min confluence score", 1, 5, 3)
            edge_only = st.checkbox("Edge levels only (VAL/VAH/POC)", value=True)
            signal_config = _build_signal_config(
                require_trend=require_trend,
                require_wick=require_wick,
                min_confluence=min_confluence,
                edge_only=edge_only,
            )

            with st.expander("Chart layers", expanded=False):
                show_profile = st.checkbox("Volume profile", value=True)
                show_poc = st.checkbox("POC / value area lines", value=True)
                show_value_area = show_poc
                show_signals = st.checkbox("Buy & sell markers", value=True)
                show_heatmap = st.checkbox("Liquidity heatmap", value=False)
                show_volume = st.checkbox("Volume bars", value=True)

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

    yahoo_symbol = resolve_ticker(ticker)
    if yahoo_symbol != ticker:
        st.caption(f"Using Yahoo symbol **{yahoo_symbol}** for index volume data.")

    signal_config = _adapt_config_for_range(signal_config, period, len(df))
    if period == "1d":
        st.caption(f"Intraday session · {len(df)} bars · auto interval **{interval}**")

    chart_tab, backtest_tab = st.tabs(["Chart", "Backtest"])

    with chart_tab:
        render_chart_tab(
            df,
            ticker,
            simple_view=simple_view,
            signal_config=signal_config,
            show_profile=show_profile,
            show_poc=show_poc,
            show_value_area=show_value_area,
            show_signals=show_signals,
            show_heatmap=show_heatmap,
            show_volume=show_volume,
            show_price_line=show_price_line,
            price_line_value=price_line_value,
            auto_trade=auto_trade,
        )

    with backtest_tab:
        render_backtest_tab(df, ticker, signal_config)


if __name__ == "__main__":
    main()