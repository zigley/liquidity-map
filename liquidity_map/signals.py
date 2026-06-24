"""Buy/sell signals where price action aligns with liquidity zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from liquidity_map.profile import VolumeProfile

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class LiquiditySignal:
    datetime: object
    price: float
    side: Side
    reason: str
    strength: int


def _bin_width(profile: VolumeProfile) -> float:
    if len(profile.bin_edges) < 2:
        return 0.01
    return float(profile.bin_edges[1] - profile.bin_edges[0])


def _tolerance(profile: VolumeProfile) -> float:
    return _bin_width(profile) * 1.5


def _near(price: float, level: float, tol: float) -> bool:
    return abs(price - level) <= tol


def _bin_index(price: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, price, side="right") - 1)
    return max(0, min(idx, len(edges) - 2))


def _liquidity_rank(profile: VolumeProfile, bin_idx: int) -> float:
    vol = profile.volumes[bin_idx]
    total = profile.volumes.sum()
    if total <= 0:
        return 0.0
    return float(vol / total)


def _is_high_liquidity(profile: VolumeProfile, bin_idx: int) -> bool:
    return bool(profile.hvn_mask[bin_idx]) or _liquidity_rank(profile, bin_idx) >= 0.012


def _is_low_liquidity(profile: VolumeProfile, bin_idx: int) -> bool:
    return bool(profile.lvn_mask[bin_idx])


def detect_liquidity_signals(
    df: pd.DataFrame,
    profile: VolumeProfile,
    min_volume_pct: float = 0.8,
    cooldown_bars: int = 3,
) -> list[LiquiditySignal]:
    """
    Emit buy/sell markers when candles interact with liquid zones.

    Buy: bullish rejection off VAL, POC, or HVN support.
    Sell: bearish rejection at VAH, POC, or HVN resistance.
    """
    if len(df) < 3:
        return []

    tol = _tolerance(profile)
    median_vol = float(df["volume"].median()) if df["volume"].median() > 0 else 0.0
    signals: list[LiquiditySignal] = []
    last_buy_idx = -cooldown_bars
    last_sell_idx = -cooldown_bars

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        vol = float(row["volume"])
        if median_vol > 0 and vol < median_vol * min_volume_pct:
            continue

        bullish = c > o
        bearish = c < o
        low_bin = _bin_index(l, profile.bin_edges)
        high_bin = _bin_index(h, profile.bin_edges)
        close_bin = _bin_index(c, profile.bin_edges)

        support_levels = (
            ("VAL", profile.val_price),
            ("POC", profile.poc_price),
        )
        resistance_levels = (
            ("VAH", profile.vah_price),
            ("POC", profile.poc_price),
        )

        # --- Buy: bounce from liquid support ---
        if bullish and i - last_buy_idx >= cooldown_bars:
            buy: LiquiditySignal | None = None
            for label, level in support_levels:
                if _near(l, level, tol) and c > level:
                    strength = 2 if label == "POC" else 1
                    if _is_high_liquidity(profile, low_bin):
                        strength += 1
                    buy = LiquiditySignal(
                        datetime=row["datetime"],
                        price=l - tol * 0.25,
                        side="buy",
                        reason=f"Bounce off {label} ({level:.2f})",
                        strength=min(strength, 3),
                    )
                    break
            if buy is None and float(prev["close"]) < profile.poc_price <= c and _near(l, profile.poc_price, tol * 1.2):
                buy = LiquiditySignal(
                    datetime=row["datetime"],
                    price=l - tol * 0.25,
                    side="buy",
                    reason="POC reclaim",
                    strength=3,
                )
            elif buy is None and _is_low_liquidity(profile, low_bin) and _is_high_liquidity(profile, close_bin) and c > float(prev["close"]):
                buy = LiquiditySignal(
                    datetime=row["datetime"],
                    price=l - tol * 0.25,
                    side="buy",
                    reason="LVN → HVN breakout",
                    strength=2,
                )
            elif buy is None and _is_high_liquidity(profile, low_bin) and c > l + (h - l) * 0.45:
                center = float(profile.bin_centers[low_bin])
                buy = LiquiditySignal(
                    datetime=row["datetime"],
                    price=l - tol * 0.25,
                    side="buy",
                    reason=f"HVN support ({center:.2f})",
                    strength=2,
                )
            if buy is not None:
                signals.append(buy)
                last_buy_idx = i

        # --- Sell: rejection at liquid resistance ---
        if bearish and i - last_sell_idx >= cooldown_bars:
            sell: LiquiditySignal | None = None
            for label, level in resistance_levels:
                if _near(h, level, tol) and c < level:
                    strength = 2 if label == "POC" else 1
                    if _is_high_liquidity(profile, high_bin):
                        strength += 1
                    sell = LiquiditySignal(
                        datetime=row["datetime"],
                        price=h + tol * 0.25,
                        side="sell",
                        reason=f"Reject at {label} ({level:.2f})",
                        strength=min(strength, 3),
                    )
                    break
            if sell is None and float(prev["close"]) > profile.poc_price >= c and _near(h, profile.poc_price, tol * 1.2):
                sell = LiquiditySignal(
                    datetime=row["datetime"],
                    price=h + tol * 0.25,
                    side="sell",
                    reason="POC loss",
                    strength=3,
                )
            elif sell is None and _is_low_liquidity(profile, high_bin) and _is_high_liquidity(profile, close_bin) and c < float(prev["close"]):
                sell = LiquiditySignal(
                    datetime=row["datetime"],
                    price=h + tol * 0.25,
                    side="sell",
                    reason="LVN → HVN breakdown",
                    strength=2,
                )
            elif sell is None and _is_high_liquidity(profile, high_bin) and c < h - (h - l) * 0.45:
                center = float(profile.bin_centers[high_bin])
                sell = LiquiditySignal(
                    datetime=row["datetime"],
                    price=h + tol * 0.25,
                    side="sell",
                    reason=f"HVN resistance ({center:.2f})",
                    strength=2,
                )
            if sell is not None:
                signals.append(sell)
                last_sell_idx = i

    return signals