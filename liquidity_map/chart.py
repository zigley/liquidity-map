"""Plotly chart builder: candles, volume, profile, heatmap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from liquidity_map.data import Quote
from liquidity_map.heatmap import build_liquidity_heatmap
from liquidity_map.liquidity_score import SpreadRating, quote_rating
from liquidity_map.profile import VolumeProfile, build_volume_profile
from liquidity_map.signals import LiquiditySignal, detect_liquidity_signals


@dataclass(frozen=True)
class ChartOptions:
    show_profile: bool = True
    show_heatmap: bool = True
    show_poc_lines: bool = True
    show_hvn_lvn: bool = False
    show_signals: bool = True
    n_bins: int = 100
    quote: Quote | None = None


def _max_profile_volume(volumes: np.ndarray) -> float:
    return float(volumes.max()) if len(volumes) and volumes.max() > 0 else 1.0


def build_chart(df: pd.DataFrame, ticker: str, options: ChartOptions) -> go.Figure:
    profile = build_volume_profile(df, n_bins=options.n_bins)
    heatmap_matrix, x_labels, y_centers, _ = build_liquidity_heatmap(df, n_bins=options.n_bins)

    profile_width = 0.22 if options.show_profile else 0.0
    main_width = 1.0 - profile_width

    fig = make_subplots(
        rows=2,
        cols=2 if options.show_profile else 1,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[main_width, profile_width] if options.show_profile else [1.0],
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
        horizontal_spacing=0.02,
        specs=[
            [{"type": "xy"}, {"type": "xy"}] if options.show_profile else [{"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}] if options.show_profile else [{"type": "xy"}],
        ],
    )

    x_vals = df["datetime"]

    if options.show_heatmap:
        fig.add_trace(
            go.Heatmap(
                x=x_vals,
                y=y_centers,
                z=heatmap_matrix,
                colorscale=[
                    [0.0, "rgba(15,23,42,0)"],
                    [0.2, "rgba(59,130,246,0.15)"],
                    [0.5, "rgba(59,130,246,0.35)"],
                    [0.8, "rgba(168,85,247,0.55)"],
                    [1.0, "rgba(236,72,153,0.75)"],
                ],
                showscale=False,
                hovertemplate="Time: %{x}<br>Price: %{y:.2f}<br>Density: %{z:.2f}<extra></extra>",
                name="Liquidity heatmap",
            ),
            row=1,
            col=1,
        )

    if options.show_hvn_lvn:
        _add_zone_rects(fig, profile, df["datetime"].iloc[0], df["datetime"].iloc[-1], profile.hvn_mask, "rgba(34,197,94,0.12)", "HVN")
        _add_zone_rects(fig, profile, df["datetime"].iloc[0], df["datetime"].iloc[-1], profile.lvn_mask, "rgba(239,68,68,0.10)", "LVN")

    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    if options.show_poc_lines:
        for price, label, color, dash in (
            (profile.poc_price, "POC", "#f59e0b", "solid"),
            (profile.vah_price, "VAH", "#60a5fa", "dot"),
            (profile.val_price, "VAL", "#60a5fa", "dot"),
        ):
            fig.add_hline(
                y=price,
                line=dict(color=color, width=1.5, dash=dash),
                annotation_text=f"{label} {price:.2f}",
                annotation_position="right",
                row=1,
                col=1,
            )

    if options.quote and options.quote.last:
        rating = quote_rating(options.quote)
        fig.add_hline(
            y=options.quote.last,
            line=dict(color=rating.color, width=2, dash="dash"),
            annotation_text=f"Last {options.quote.last:.2f}",
            annotation_position="left",
            row=1,
            col=1,
        )

    if options.show_signals:
        _add_signal_markers(fig, df, profile)

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=df["volume"],
            name="Volume",
            marker_color="rgba(100,116,139,0.55)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if options.show_profile:
        max_vol = _max_profile_volume(profile.volumes)
        norm_vol = profile.volumes / max_vol
        fig.add_trace(
            go.Bar(
                x=norm_vol,
                y=profile.bin_centers,
                orientation="h",
                name="Volume profile",
                marker=dict(
                    color=profile.volumes,
                    colorscale="Viridis",
                    showscale=False,
                ),
                hovertemplate="Price: %{y:.2f}<br>Volume: %{customdata:,.0f}<extra></extra>",
                customdata=profile.volumes,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[0] * len(profile.bin_centers),
                y=profile.bin_centers,
                mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=2,
        )

    title_suffix = ""
    if options.quote:
        rating = quote_rating(options.quote)
        spread_txt = f"{rating.spread_pct:.3f}%" if rating.spread_pct is not None else "n/a"
        bid = f"{options.quote.bid:.2f}" if options.quote.bid else "—"
        ask = f"{options.quote.ask:.2f}" if options.quote.ask else "—"
        title_suffix = f" | Spread {spread_txt} ({rating.label}) | Bid {bid} / Ask {ask}"

    fig.update_layout(
        title=dict(text=f"{ticker} Liquidity Map{title_suffix}", x=0.01, xanchor="left"),
        template="plotly_dark",
        height=780,
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    if options.show_profile:
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
        fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=2, col=2)

    return fig


def _add_signal_markers(fig: go.Figure, df: pd.DataFrame, profile: VolumeProfile) -> None:
    signals = detect_liquidity_signals(df, profile)
    buys = [s for s in signals if s.side == "buy"]
    sells = [s for s in signals if s.side == "sell"]

    if buys:
        fig.add_trace(
            go.Scatter(
                x=[s.datetime for s in buys],
                y=[s.price for s in buys],
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=[8 + s.strength * 2 for s in buys],
                    color="#22c55e",
                    line=dict(width=1, color="#14532d"),
                ),
                customdata=[s.reason for s in buys],
                hovertemplate="BUY<br>%{x}<br>%{y:.2f}<br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if sells:
        fig.add_trace(
            go.Scatter(
                x=[s.datetime for s in sells],
                y=[s.price for s in sells],
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=[8 + s.strength * 2 for s in sells],
                    color="#ef4444",
                    line=dict(width=1, color="#7f1d1d"),
                ),
                customdata=[s.reason for s in sells],
                hovertemplate="SELL<br>%{x}<br>%{y:.2f}<br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def _add_zone_rects(
    fig: go.Figure,
    profile: VolumeProfile,
    x0,
    x1,
    mask: np.ndarray,
    fillcolor: str,
    name: str,
) -> None:
    in_zone = False
    start_idx = 0
    for i, active in enumerate(mask):
        if active and not in_zone:
            in_zone = True
            start_idx = i
        elif not active and in_zone:
            in_zone = False
            _add_rect(fig, profile, start_idx, i - 1, x0, x1, fillcolor, name)
    if in_zone:
        _add_rect(fig, profile, start_idx, len(mask) - 1, x0, x1, fillcolor, name)


def _add_rect(
    fig: go.Figure,
    profile: VolumeProfile,
    start_idx: int,
    end_idx: int,
    x0,
    x1,
    fillcolor: str,
    name: str,
) -> None:
    y0 = float(profile.bin_edges[start_idx])
    y1 = float(profile.bin_edges[min(end_idx + 1, len(profile.bin_edges) - 1)])
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        fillcolor=fillcolor,
        line=dict(width=0),
        layer="below",
        row=1,
        col=1,
    )