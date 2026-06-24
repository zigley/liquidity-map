"""Plotly chart builder: candles, volume, profile, heatmap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from liquidity_map.data import Quote
from liquidity_map.paper_broker import PositionInfo
from liquidity_map.heatmap import build_liquidity_heatmap
from liquidity_map.liquidity_score import SpreadRating, quote_rating
from liquidity_map.profile import VolumeProfile, build_volume_profile
from liquidity_map.signals import LiquiditySignal, detect_liquidity_signals


@dataclass(frozen=True)
class PriceLine:
    price: float
    label: str = ""
    color: str = "#e879f9"
    dash: str = "dash"


@dataclass(frozen=True)
class ChartOptions:
    show_profile: bool = True
    show_heatmap: bool = True
    show_poc_lines: bool = True
    show_hvn_lvn: bool = False
    show_signals: bool = True
    show_signal_price_lines: bool = True
    show_paper_trades: bool = True
    n_bins: int = 100
    quote: Quote | None = None
    paper_trades: list[dict] | None = None
    price_lines: tuple[PriceLine, ...] = ()
    position: PositionInfo | None = None
    show_position_line: bool = True


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

    for line in options.price_lines:
        _add_price_line(fig, line)

    if options.show_position_line and options.position:
        _add_position_line(fig, options.position)

    signal_bars: dict = {}
    if options.show_signals:
        signal_bars = _add_signal_markers(fig, df, profile, options.show_signal_price_lines)

    volume_colors = []
    for ts in x_vals:
        key = pd.Timestamp(ts)
        side = signal_bars.get(key)
        if side == "buy":
            volume_colors.append("#22c55e")
        elif side == "sell":
            volume_colors.append("#ef4444")
        else:
            volume_colors.append("rgba(100,116,139,0.55)")

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=df["volume"],
            name="Volume",
            marker_color=volume_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if options.show_paper_trades and options.paper_trades:
        _add_paper_trade_markers(fig, df, options.paper_trades)

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


def _bar_lookup(df: pd.DataFrame) -> dict:
    return {pd.Timestamp(row.datetime): row for row in df.itertuples(index=False)}


def _signal_y(signal: LiquiditySignal, bars: dict, side: str) -> float:
    bar = bars.get(pd.Timestamp(signal.datetime))
    if bar is None:
        return signal.price
    return float(bar.low) if side == "buy" else float(bar.high)


def _add_position_line(fig: go.Figure, pos: PositionInfo) -> None:
    pnl_sign = "+" if pos.pnl >= 0 else ""
    pnl_color = "#22c55e" if pos.pnl >= 0 else "#ef4444"
    label = (
        f"Position {pos.qty:.2f} {pos.symbol} @ ${pos.avg_price:.2f} "
        f"({pnl_sign}${pos.pnl:.2f} / {pnl_sign}{pos.pnl_pct:.1f}%)"
    )
    fig.add_hline(
        y=pos.avg_price,
        line=dict(color="#38bdf8", width=2.5, dash="solid"),
        annotation_text=label,
        annotation_position="right",
        annotation_font=dict(color="#38bdf8", size=12),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=pos.market_price,
        line=dict(color=pnl_color, width=1.5, dash="dashdot"),
        annotation_text=f"Mark ${pos.market_price:.2f}",
        annotation_position="left",
        annotation_font=dict(color=pnl_color, size=11),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=1,
        xref="paper",
        y0=min(pos.avg_price, pos.market_price),
        y1=max(pos.avg_price, pos.market_price),
        fillcolor="rgba(56,189,248,0.12)" if pos.pnl >= 0 else "rgba(239,68,68,0.12)",
        line=dict(width=0),
        layer="below",
        row=1,
        col=1,
    )


def _add_price_line(fig: go.Figure, line: PriceLine) -> None:
    label = line.label or f"${line.price:.2f}"
    fig.add_hline(
        y=line.price,
        line=dict(color=line.color, width=2, dash=line.dash),
        annotation_text=label,
        annotation_position="right",
        annotation_font=dict(color=line.color, size=12),
        row=1,
        col=1,
    )


def _add_signal_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    profile: VolumeProfile,
    draw_price_lines: bool = True,
) -> dict:
    """Draw buy/sell markers on the chart. Returns {datetime: side} for volume coloring."""
    signals = detect_liquidity_signals(df, profile)
    bars = _bar_lookup(df)
    buys = [s for s in signals if s.side == "buy"]
    sells = [s for s in signals if s.side == "sell"]
    signal_bars: dict = {}

    for s in signals:
        signal_bars[pd.Timestamp(s.datetime)] = s.side

    if buys:
        buy_x = [s.datetime for s in buys]
        buy_y = [_signal_y(s, bars, "buy") for s in buys]
        buy_text = ["BUY"] * len(buys)
        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers+text",
                name="Buy signals",
                text=buy_text,
                textposition="bottom center",
                textfont=dict(size=11, color="#4ade80", family="Arial Black"),
                marker=dict(
                    symbol="triangle-up",
                    size=[14 + s.strength * 4 for s in buys],
                    color="#22c55e",
                    line=dict(width=2, color="#ffffff"),
                ),
                customdata=[s.reason for s in buys],
                hovertemplate="<b>BUY</b><br>%{x}<br>$%{y:.2f}<br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        for x in buy_x:
            fig.add_vline(x=x, line=dict(color="rgba(34,197,94,0.35)", width=1, dash="dot"), row=1, col=1)
        if draw_price_lines:
            for price in sorted({round(y, 2) for y in buy_y}):
                fig.add_hline(
                    y=price,
                    line=dict(color="rgba(34,197,94,0.45)", width=1.5, dash="dot"),
                    annotation_text=f"Buy ${price:.2f}",
                    annotation_position="left",
                    annotation_font=dict(color="#4ade80", size=10),
                    row=1,
                    col=1,
                )

    if sells:
        sell_x = [s.datetime for s in sells]
        sell_y = [_signal_y(s, bars, "sell") for s in sells]
        sell_text = ["SELL"] * len(sells)
        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="markers+text",
                name="Sell signals",
                text=sell_text,
                textposition="top center",
                textfont=dict(size=11, color="#f87171", family="Arial Black"),
                marker=dict(
                    symbol="triangle-down",
                    size=[14 + s.strength * 4 for s in sells],
                    color="#ef4444",
                    line=dict(width=2, color="#ffffff"),
                ),
                customdata=[s.reason for s in sells],
                hovertemplate="<b>SELL</b><br>%{x}<br>$%{y:.2f}<br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        for x in sell_x:
            fig.add_vline(x=x, line=dict(color="rgba(239,68,68,0.35)", width=1, dash="dot"), row=1, col=1)
        if draw_price_lines:
            for price in sorted({round(y, 2) for y in sell_y}, reverse=True):
                fig.add_hline(
                    y=price,
                    line=dict(color="rgba(239,68,68,0.45)", width=1.5, dash="dot"),
                    annotation_text=f"Sell ${price:.2f}",
                    annotation_position="left",
                    annotation_font=dict(color="#f87171", size=10),
                    row=1,
                    col=1,
                )

    return signal_bars


def _add_paper_trade_markers(fig: go.Figure, df: pd.DataFrame, trades: list[dict]) -> None:
    """Overlay executed paper trades as diamond markers."""
    bars = _bar_lookup(df)
    executed_buys = [t for t in trades if t.get("action") == "buy"]
    executed_sells = [t for t in trades if t.get("action") == "sell"]

    def _match_x(trade: dict):
        ts = pd.Timestamp(trade.get("timestamp", ""))
        for bar_ts, bar in bars.items():
            if abs((bar_ts - ts).total_seconds()) < 86400:
                return bar_ts, float(trade.get("price", bar.close))
        return None, float(trade.get("price", 0))

    if executed_buys:
        xs, ys, labels = [], [], []
        for t in executed_buys:
            x, y = _match_x(t)
            if x is not None:
                xs.append(x)
                ys.append(y)
                labels.append("FILLED BUY")
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    name="Paper buys",
                    text=labels,
                    textposition="middle right",
                    textfont=dict(size=10, color="#86efac"),
                    marker=dict(symbol="diamond", size=12, color="#16a34a", line=dict(width=2, color="#fff")),
                    hovertemplate="<b>Paper BUY</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if executed_sells:
        xs, ys, labels = [], [], []
        for t in executed_sells:
            x, y = _match_x(t)
            if x is not None:
                xs.append(x)
                ys.append(y)
                labels.append("FILLED SELL")
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    name="Paper sells",
                    text=labels,
                    textposition="middle left",
                    textfont=dict(size=10, color="#fca5a5"),
                    marker=dict(symbol="diamond", size=12, color="#dc2626", line=dict(width=2, color="#fff")),
                    hovertemplate="<b>Paper SELL</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
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