"""
Unified buy/sell model for stocks.

One strategy, three ideas:
  1. WHERE  — enter at high-volume support (VAL / POC)
  2. TREND  — only buy when price is above the moving average
  3. EXIT   — sell on trail stop or VAH target (never wait for POC reject)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from liquidity_map.profile import VolumeProfile, build_volume_profile

Action = Literal["buy", "sell", "wait"]
Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class ModelConfig:
    trend_ma: int = 50
    profile_lookback: int = 60
    trail_pct: float = 1.5
    trail_pct_after_vah: float = 1.0
    min_gain_for_trail: float = 0.5
    min_vah_gain_pct: float = 0.8
    wick_ratio: float = 0.30
    volume_mult: float = 0.85
    level_tolerance: float = 2.0
    cooldown_bars: int = 3


DEFAULT_CONFIG = ModelConfig()


@dataclass(frozen=True)
class TradeMarker:
    datetime: object
    action: Side
    price: float
    label: str
    reason: str


@dataclass(frozen=True)
class TradeAdvice:
    action: Action
    reason: str
    price: float
    target: float | None = None
    stop: float | None = None
    entry: float | None = None
    poc: float = 0.0
    val: float = 0.0
    vah: float = 0.0


@dataclass(frozen=True)
class BacktestStats:
    trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float


def _ma_period(df: pd.DataFrame, cfg: ModelConfig) -> int:
    return min(cfg.trend_ma, max(10, len(df) // 5))


def _tol(profile: VolumeProfile, mult: float = 2.0) -> float:
    edges = profile.bin_edges
    if len(edges) < 2:
        return max(abs(profile.poc_price) * 0.003, 0.01)
    return float(edges[1] - edges[0]) * mult


def _touches_support(low: float, close: float, level: float, tol: float) -> bool:
    return abs(low - level) <= tol or abs(close - level) <= tol * 0.75


def _bullish_rejection(o: float, h: float, l: float, c: float, ratio: float) -> bool:
    rng = h - l
    return rng > 0 and (min(o, c) - l) / rng >= ratio and c > o


def _trend_up(close: float, ma: float) -> bool:
    return np.isnan(ma) or close > ma


def _entry_trigger(
    o: float,
    h: float,
    l: float,
    c: float,
    vol: float,
    profile: VolumeProfile,
    ma: float,
    median_vol: float,
    cfg: ModelConfig,
) -> tuple[bool, str]:
    if not _trend_up(c, ma):
        return False, ""
    if median_vol > 0 and vol < median_vol * cfg.volume_mult:
        return False, ""

    tol = _tol(profile, cfg.level_tolerance)
    bullish = c >= o
    rng = h - l
    wick_ok = rng <= 0 or (min(o, c) - l) / rng >= cfg.wick_ratio or c > o

    for name, level in (("VAL", profile.val_price), ("POC", profile.poc_price)):
        if _touches_support(l, c, level, tol) and c >= level * 0.999 and bullish and wick_ok:
            return True, f"Bounce off {name} ${level:.2f} (uptrend)"

    return False, ""


def _exit_levels(entry: float, peak: float, profile: VolumeProfile, cfg: ModelConfig) -> tuple[float | None, float | None]:
    vah = profile.vah_price
    target = vah if vah >= entry * (1 + cfg.min_vah_gain_pct / 100) else None

    trail_pct = cfg.trail_pct_after_vah if peak >= vah >= entry else cfg.trail_pct
    stop = None
    if peak >= entry * (1 + cfg.min_gain_for_trail / 100):
        stop = peak * (1 - trail_pct / 100)
    return target, stop


def _exit_trigger(
    h: float,
    c: float,
    entry: float,
    peak: float,
    profile: VolumeProfile,
    cfg: ModelConfig,
) -> tuple[bool, str]:
    target, stop = _exit_levels(entry, peak, profile, cfg)
    if target is not None and h >= target:
        return True, f"Target VAH ${target:.2f}"
    if stop is not None and c <= stop:
        return True, f"Trail stop ${stop:.2f} ({cfg.trail_pct}% below peak ${peak:.2f})"
    return False, ""


def scan_trades(df: pd.DataFrame, cfg: ModelConfig = DEFAULT_CONFIG) -> list[TradeMarker]:
    """Walk-forward history: buy at support, sell on trail/target."""
    if len(df) < 30:
        return []

    lookback = min(cfg.profile_lookback, max(20, len(df) // 2))
    ma_len = _ma_period(df, cfg)
    warmup = max(ma_len, 20)
    median_vol = float(df["volume"].median()) or 0.0
    ma_series = df["close"].rolling(ma_len, min_periods=ma_len).mean()

    markers: list[TradeMarker] = []
    in_trade = False
    entry = peak = 0.0
    buy_n = sell_n = 0
    last_buy_idx = -cfg.cooldown_bars

    for i in range(warmup, len(df)):
        window = df.iloc[max(0, i - lookback + 1) : i + 1]
        profile = build_volume_profile(window)
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        vol = float(row["volume"])
        ma = float(ma_series.iloc[i])

        if not in_trade:
            if i - last_buy_idx < cfg.cooldown_bars:
                continue
            hit, reason = _entry_trigger(o, h, l, c, vol, profile, ma, median_vol, cfg)
            if hit:
                buy_n += 1
                markers.append(
                    TradeMarker(
                        datetime=row["datetime"],
                        action="buy",
                        price=c,
                        label=f"B{buy_n}",
                        reason=reason,
                    )
                )
                in_trade = True
                entry = peak = c
                last_buy_idx = i
        else:
            peak = max(peak, h)
            hit, reason = _exit_trigger(h, c, entry, peak, profile, cfg)
            if hit:
                sell_n += 1
                markers.append(
                    TradeMarker(
                        datetime=row["datetime"],
                        action="sell",
                        price=c,
                        label=f"S{sell_n}",
                        reason=reason,
                    )
                )
                in_trade = False

    return markers


def live_advice(
    df: pd.DataFrame,
    *,
    in_position: bool = False,
    entry_price: float | None = None,
    peak_price: float | None = None,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> TradeAdvice:
    """What to do right now on the latest bar."""
    if df.empty:
        return TradeAdvice("wait", "No data", 0.0)

    lookback = min(cfg.profile_lookback, max(20, len(df) // 2))
    window = df.iloc[-lookback:]
    profile = build_volume_profile(window)
    row = df.iloc[-1]
    price = float(row["close"])
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), price
    vol = float(row["volume"])
    ma_len = _ma_period(df, cfg)
    ma = float(df["close"].rolling(ma_len, min_periods=ma_len).mean().iloc[-1])
    median_vol = float(df["volume"].median()) or 0.0

    base = TradeAdvice(
        action="wait",
        reason="",
        price=price,
        poc=profile.poc_price,
        val=profile.val_price,
        vah=profile.vah_price,
    )

    if in_position and entry_price:
        peak = max(peak_price or entry_price, h, price)
        target, stop = _exit_levels(entry_price, peak, profile, cfg)
        hit, reason = _exit_trigger(h, c, entry_price, peak, profile, cfg)
        if hit:
            return TradeAdvice(
                action="sell",
                reason=reason,
                price=price,
                target=target,
                stop=stop,
                entry=entry_price,
                poc=profile.poc_price,
                val=profile.val_price,
                vah=profile.vah_price,
            )
        wait = f"Holding — target ${target:.2f}" if target else "Holding"
        if stop:
            wait += f", stop ${stop:.2f}"
        return TradeAdvice(
            action="wait",
            reason=wait,
            price=price,
            target=target,
            stop=stop,
            entry=entry_price,
            poc=profile.poc_price,
            val=profile.val_price,
            vah=profile.vah_price,
        )

    hit, reason = _entry_trigger(o, h, l, c, vol, profile, ma, median_vol, cfg)
    if hit:
        return TradeAdvice(
            action="buy",
            reason=reason,
            price=price,
            target=profile.vah_price if profile.vah_price > price else None,
            stop=None,
            poc=profile.poc_price,
            val=profile.val_price,
            vah=profile.vah_price,
        )

    if not _trend_up(c, ma):
        return TradeAdvice(
            action="wait",
            reason=f"Below {ma_len}-bar MA — wait for uptrend",
            price=price,
            poc=profile.poc_price,
            val=profile.val_price,
            vah=profile.vah_price,
        )

    return TradeAdvice(
        action="wait",
        reason=f"Watch VAL ${profile.val_price:.2f} or POC ${profile.poc_price:.2f} for bounce",
        price=price,
        poc=profile.poc_price,
        val=profile.val_price,
        vah=profile.vah_price,
    )


def backtest(df: pd.DataFrame, cfg: ModelConfig = DEFAULT_CONFIG) -> BacktestStats:
    """Round-trip stats from walk-forward scan."""
    markers = scan_trades(df, cfg)
    buys = [m for m in markers if m.action == "buy"]
    sells = [m for m in markers if m.action == "sell"]
    n = min(len(buys), len(sells))
    if n == 0:
        return BacktestStats(0, 0.0, 0.0, 0.0)

    closes = {pd.Timestamp(r.datetime): float(r.close) for r in df.itertuples(index=False)}
    returns: list[float] = []
    for b, s in zip(buys[:n], sells[:n]):
        bp = closes.get(pd.Timestamp(b.datetime))
        sp = closes.get(pd.Timestamp(s.datetime))
        if bp and sp and bp > 0:
            returns.append((sp - bp) / bp * 100)

    if not returns:
        return BacktestStats(0, 0.0, 0.0, 0.0)

    wins = sum(1 for r in returns if r > 0)
    return BacktestStats(
        trades=n,
        win_rate=wins / len(returns) * 100,
        avg_return_pct=sum(returns) / len(returns),
        total_return_pct=sum(returns),
    )


def adapt_config(period: str, n_bars: int, cfg: ModelConfig = DEFAULT_CONFIG) -> ModelConfig:
    if period != "1d":
        return cfg
    return ModelConfig(
        trend_ma=min(20, max(8, n_bars // 4)),
        profile_lookback=min(40, max(15, n_bars // 2)),
        trail_pct=cfg.trail_pct,
        trail_pct_after_vah=cfg.trail_pct_after_vah,
        min_gain_for_trail=cfg.min_gain_for_trail,
        min_vah_gain_pct=cfg.min_vah_gain_pct,
        wick_ratio=cfg.wick_ratio,
        volume_mult=cfg.volume_mult,
        cooldown_bars=max(2, cfg.cooldown_bars // 2),
    )