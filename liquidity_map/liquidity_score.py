"""Live and bar-level liquidity scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from liquidity_map.data import Quote


@dataclass(frozen=True)
class SpreadRating:
    label: str
    color: str
    spread_pct: float | None


def rate_spread(spread_pct: float | None) -> SpreadRating:
    if spread_pct is None:
        return SpreadRating(label="Unknown", color="#9ca3af", spread_pct=None)
    if spread_pct < 0.05:
        return SpreadRating(label="Liquid", color="#22c55e", spread_pct=spread_pct)
    if spread_pct < 0.20:
        return SpreadRating(label="Moderate", color="#f59e0b", spread_pct=spread_pct)
    return SpreadRating(label="Illiquid", color="#ef4444", spread_pct=spread_pct)


def quote_rating(quote: Quote) -> SpreadRating:
    return rate_spread(quote.spread_pct)


def bar_liquidity_density(df: pd.DataFrame) -> pd.Series:
    """Per-bar proxy: volume divided by bar range (higher = more liquid activity)."""
    span = (df["high"] - df["low"]).clip(lower=1e-9)
    return df["volume"] / span