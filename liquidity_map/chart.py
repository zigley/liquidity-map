"""Plotly chart builder — simplified, readable liquidity map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from liquidity_map.data import Quote
from liquidity_map.paper_broker import PositionInfo
from liquidity_map.heatmap import build_liquidity_heatmap
from liquidity_map.profile import VolumeProfile, build_volume_profile
from liquidity_map.signals import (
    DEFAULT_SIGNAL_CONFIG,
    LiquiditySignal,
    SignalConfig,
    detect_liquidity_signals,
)


@dataclass(frozen=True)
class PriceLine:
    price: float
    label: str = ""
    color: str = "#a78bfa"
    dash: str = "dash"


@dataclass(frozen=True)
class ChartOptions:
    show_profile: bool = True
    show_heatmap: bool = False
    show_poc_lines: bool = True
    show_value_area: bool = True
    show_signals: bool = True
    min_confluence: int = 3
    signal_config: SignalConfig = DEFAULT_SIGNAL_CONFIG
    show_volume: bool = True
    n_bins: int = 80
    price_lines: tuple[PriceLine, ...] = ()
    position: PositionInfo | None = None
    show_position_line: bool = True


def build_chart(df: pd.DataFrame, ticker: str, options: ChartOptions) -> go.Figure:
    profile = build_volume_profile(df, n_bins=options.n_bins)

    profile_width = 0.2 if options.show_profile else 0.0
    main_width = 1.0 - profile_width
    n_cols = 2 if options.show_profile else 1
    n_rows = 2 if options.show_volume else 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[main_width, profile_width] if options.show_profile else [1.0],
        row_heights=[0.75, 0.25] if options.show_volume else [1.0],
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        specs=[
            [{"type": "xy"}, {"type": "xy"}] if options.show_profile else [{"type": "xy"}],
        ]
        + (
            [[{"type": "xy"}, {"type": "xy"}] if options.show_profile else [{"type": "xy"}]]
            if options.show_volume
            else []
        ),
    )

    x_vals = df["datetime"]

    if options.show_heatmap:
        heatmap_matrix, _, y_centers, _ = build_liquidity_heatmap(df, n_bins=options.n_bins)
        fig.add_trace(
            go.Heatmap(
                x=x_vals,
                y=y_centers,
                z=heatmap_matrix,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(59,130,246,0.25)"]],
                showscale=False,
                hoverinfo="skip",
                name="Liquidity",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            increasing_fillcolor="#16a34a",
            decreasing_fillcolor="#dc2626",
        ),
        row=1,
        col=1,
    )

    if options.show_poc_lines:
        _add_level(fig, profile.poc_price, "POC (most traded)", "#f59e0b", "solid", width=2)
        if options.show_value_area:
            _add_level(fig, profile.vah_price, "VAH", "#93c5fd", "dot")
            _add_level(fig, profile.val_price, "VAL", "#93c5fd", "dot")

    for line in options.price_lines:
        _add_level(fig, line.price, line.label or f"${line.price:.0f}", line.color, line.dash)

    if options.show_position_line and options.position:
        pos = options.position
        _add_level(fig, pos.avg_price, f"Entry ${pos.avg_price:.2f}", "#38bdf8", "solid", width=2.5)

    if options.show_signals:
        _add_signal_markers(
            fig,
            df,
            profile,
            config=options.signal_config,
            min_confluence=options.min_confluence,
        )

    if options.show_volume:
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=df["volume"],
                name="Volume",
                marker_color="rgba(148,163,184,0.5)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if options.show_profile:
        max_vol = float(profile.volumes.max()) or 1.0
        fig.add_trace(
            go.Bar(
                x=profile.volumes / max_vol,
                y=profile.bin_centers,
                orientation="h",
                name="Profile",
                marker_color="rgba(148,163,184,0.45)",
                showlegend=False,
                hovertemplate="Price $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(text=f"{ticker}", font=dict(size=24)),
        template="plotly_white",
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        height=680 if options.show_volume else 620,
        margin=dict(l=64, r=120, t=48, b=36),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=12)),
        font=dict(size=14, color="#1e293b"),
        hovermode="x unified",
        bargap=0.15,
    )

    fig.update_yaxes(title_text="Price ($)", gridcolor="#e2e8f0", row=1, col=1)
    if options.show_volume:
        fig.update_yaxes(title_text="Volume", gridcolor="#e2e8f0", row=2, col=1)
        fig.update_xaxes(gridcolor="#e2e8f0", row=2, col=1)
    if options.show_profile:
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)

    return fig


def _add_level(
    fig: go.Figure,
    price: float,
    label: str,
    color: str,
    dash: str,
    width: float = 1.5,
) -> None:
    fig.add_hline(
        y=price,
        line=dict(color=color, width=width, dash=dash),
        annotation_text=label,
        annotation_position="right",
        annotation_font=dict(size=12, color=color),
        row=1,
        col=1,
    )


def _bar_lookup(df: pd.DataFrame) -> dict:
    return {pd.Timestamp(row.datetime): row for row in df.itertuples(index=False)}


def _add_signal_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    profile: VolumeProfile,
    *,
    config: SignalConfig = DEFAULT_SIGNAL_CONFIG,
    min_confluence: int = 3,
) -> None:
    cfg = SignalConfig(
        require_trend_filter=config.require_trend_filter,
        require_rejection_wick=config.require_rejection_wick,
        trend_ma_period=config.trend_ma_period,
        volume_spike_pct=config.volume_spike_pct,
        min_volume_pct=config.min_volume_pct,
        min_confluence=min_confluence,
        cooldown_bars=config.cooldown_bars,
        edge_only=config.edge_only,
        wick_ratio=config.wick_ratio,
    )
    signals = detect_liquidity_signals(df, profile, config=cfg)
    bars = _bar_lookup(df)
    buys = [s for s in signals if s.side == "buy"]
    sells = [s for s in signals if s.side == "sell"]

    def _y(signal: LiquiditySignal, field: str) -> float:
        bar = bars.get(pd.Timestamp(signal.datetime))
        return float(getattr(bar, field)) if bar else signal.price

    if buys:
        fig.add_trace(
            go.Scatter(
                x=[s.datetime for s in buys],
                y=[_y(s, "low") for s in buys],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=16, color="#16a34a", line=dict(width=1.5, color="#fff")),
                customdata=[[s.confluence, s.reason] for s in buys],
                hovertemplate="<b>Buy</b> $%{y:.2f}<br>Score %{customdata[0]}/5<br>%{customdata[1]}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if sells:
        fig.add_trace(
            go.Scatter(
                x=[s.datetime for s in sells],
                y=[_y(s, "high") for s in sells],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=16, color="#dc2626", line=dict(width=1.5, color="#fff")),
                customdata=[[s.confluence, s.reason] for s in sells],
                hovertemplate="<b>Sell</b> $%{y:.2f}<br>Score %{customdata[0]}/5<br>%{customdata[1]}<extra></extra>",
            ),
            row=1,
            col=1,
        )