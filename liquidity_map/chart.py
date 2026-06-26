"""Clean chart for the unified trading model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from liquidity_map.model import TradeAdvice, TradeMarker
from liquidity_map.profile import build_volume_profile


@dataclass(frozen=True)
class ChartLines:
    target: float | None = None
    stop: float | None = None
    entry: float | None = None


def build_chart(
    df: pd.DataFrame,
    ticker: str,
    markers: list[TradeMarker],
    advice: TradeAdvice,
    lines: ChartLines | None = None,
) -> go.Figure:
    profile = build_volume_profile(df, n_bins=80)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.04,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    x = df["datetime"]
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    max_vol = float(profile.volumes.max()) or 1.0
    fig.add_trace(
        go.Bar(
            x=profile.volumes / max_vol,
            y=profile.bin_centers,
            orientation="h",
            marker_color="rgba(148,163,184,0.4)",
            showlegend=False,
            hovertemplate="Price $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _hline(fig, profile.poc_price, "Busiest price", "#f59e0b", "solid", 2)
    _hline(fig, profile.val_price, "Support floor", "#94a3b8", "dot")
    _hline(fig, profile.vah_price, "Profit ceiling", "#94a3b8", "dot")

    ln = lines or ChartLines()
    if ln.entry:
        _hline(fig, ln.entry, "You bought here", "#38bdf8", "solid", 2)
    if ln.target:
        _hline(fig, ln.target, "Take profit here", "#22c55e", "solid", 1.5)
    if ln.stop:
        _hline(fig, ln.stop, "Sell if drops here", "#ef4444", "dot", 1.5)

    bars = {pd.Timestamp(r.datetime): r for r in df.itertuples(index=False)}
    buys = [m for m in markers if m.action == "buy"]
    sells = [m for m in markers if m.action == "sell"]

    if buys:
        fig.add_trace(
            go.Scatter(
                x=[m.datetime for m in buys],
                y=[float(bars[pd.Timestamp(m.datetime)].low) for m in buys],
                mode="markers+text",
                name="Buy",
                text=[m.label for m in buys],
                textposition="bottom center",
                marker=dict(symbol="triangle-up", size=14, color="#16a34a"),
                customdata=[m.reason for m in buys],
                hovertemplate="<b>%{text} Buy</b><br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if sells:
        fig.add_trace(
            go.Scatter(
                x=[m.datetime for m in sells],
                y=[float(bars[pd.Timestamp(m.datetime)].high) for m in sells],
                mode="markers+text",
                name="Sell",
                text=[m.label for m in sells],
                textposition="top center",
                marker=dict(symbol="triangle-down", size=14, color="#dc2626"),
                customdata=[m.reason for m in sells],
                hovertemplate="<b>%{text} Sell</b><br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=ticker,
        template="plotly_white",
        height=600,
        margin=dict(l=56, r=100, t=40, b=36),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)
    return fig


def _hline(fig: go.Figure, price: float, label: str, color: str, dash: str, width: float = 1.5) -> None:
    fig.add_hline(
        y=price,
        line=dict(color=color, width=width, dash=dash),
        annotation_text=label,
        annotation_position="right",
        annotation_font=dict(size=11, color=color),
        row=1,
        col=1,
    )