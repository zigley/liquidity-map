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

from liquidity_map.data import is_crypto
from liquidity_map.profile import VolumeProfile, build_volume_profile

LONG_TREND_MA = 200
CRYPTO_TRAIL_PCT = 2.75
CRYPTO_TRAIL_AFTER_VAH_PCT = 2.0

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

STRICTNESS_LABELS: dict[int, str] = {
    1: "Loose — more buy signals, less picky",
    2: "Relaxed — a few extra trades",
    3: "Balanced — default",
    4: "Careful — only cleaner setups",
    5: "Strict — fewest trades, highest bar",
}


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


def _long_trend_period(available_bars: int) -> int:
    if available_bars >= LONG_TREND_MA:
        return LONG_TREND_MA
    if available_bars >= 50:
        return 50
    return 0


def _long_trend_ok(df: pd.DataFrame, idx: int) -> tuple[bool, int]:
    """True when price is above the long-term (200-day) average."""
    period = _long_trend_period(idx + 1)
    if period == 0:
        return True, 0
    window = df["close"].iloc[: idx + 1]
    ma = float(window.rolling(period, min_periods=period).mean().iloc[-1])
    close = float(window.iloc[-1])
    return _trend_up(close, ma), period


def apply_asset_tuning(cfg: ModelConfig, ticker: str) -> ModelConfig:
    """Wider trail stops for volatile crypto."""
    if not is_crypto(ticker):
        return cfg
    return ModelConfig(
        trend_ma=cfg.trend_ma,
        profile_lookback=cfg.profile_lookback,
        trail_pct=max(cfg.trail_pct, CRYPTO_TRAIL_PCT),
        trail_pct_after_vah=max(cfg.trail_pct_after_vah, CRYPTO_TRAIL_AFTER_VAH_PCT),
        min_gain_for_trail=cfg.min_gain_for_trail,
        min_vah_gain_pct=cfg.min_vah_gain_pct,
        wick_ratio=cfg.wick_ratio,
        volume_mult=cfg.volume_mult,
        level_tolerance=cfg.level_tolerance,
        cooldown_bars=cfg.cooldown_bars,
    )


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

    spots = (
        ("the support floor", profile.val_price),
        ("the busiest price", profile.poc_price),
    )
    for label, level in spots:
        if _touches_support(l, c, level, tol) and c >= level * 0.999 and bullish and wick_ok:
            return True, f"Price bounced off {label} (${level:.2f}). Trend is up — good time to buy."

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
        return True, f"Hit the profit target (${target:.2f}) — time to sell."
    if stop is not None and c <= stop:
        trail_used = cfg.trail_pct_after_vah if peak >= profile.vah_price >= entry else cfg.trail_pct
        return True, f"Price fell {trail_used:.1f}% from its high — sell to protect gains."
    return False, ""


def scan_trades(df: pd.DataFrame, cfg: ModelConfig = DEFAULT_CONFIG, *, ticker: str = "") -> list[TradeMarker]:
    cfg = apply_asset_tuning(cfg, ticker)
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
            long_ok, _ = _long_trend_ok(df, i)
            hit, reason = _entry_trigger(o, h, l, c, vol, profile, ma, median_vol, cfg)
            if hit and long_ok:
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
    ticker: str = "",
    trend_df: pd.DataFrame | None = None,
) -> TradeAdvice:
    cfg = apply_asset_tuning(cfg, ticker)
    trend_source = trend_df if trend_df is not None and len(trend_df) > len(df) else df
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
        if target and stop:
            wait = f"You own this. Sell near ${target:.2f} (profit target), or if price drops to ${stop:.2f}."
        elif target:
            wait = f"You own this. Sell near ${target:.2f} when price gets there."
        elif stop:
            wait = f"You own this. Sell if price drops to ${stop:.2f}."
        else:
            wait = "You own this. Waiting for a sell signal."
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

    long_ok, long_period = _long_trend_ok(trend_source, len(trend_source) - 1)
    long_ma = (
        float(trend_source["close"].rolling(long_period, min_periods=long_period).mean().iloc[-1])
        if long_period > 0
        else float("nan")
    )

    hit, reason = _entry_trigger(o, h, l, c, vol, profile, ma, median_vol, cfg)
    if hit and not long_ok:
        label = f"{long_period}-day" if long_period == LONG_TREND_MA else f"{long_period}-bar"
        return TradeAdvice(
            action="wait",
            reason=(
                f"Bounce looks good, but big-picture trend is down (below {label} average "
                f"${long_ma:.2f}) — skip new buys until price is above that."
            ),
            price=price,
            poc=profile.poc_price,
            val=profile.val_price,
            vah=profile.vah_price,
        )

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

    if not long_ok:
        label = f"{long_period}-day" if long_period == LONG_TREND_MA else f"{long_period}-bar"
        return TradeAdvice(
            action="wait",
            reason=f"Big-picture trend is down (below {label} average) — no new buys for now.",
            price=price,
            poc=profile.poc_price,
            val=profile.val_price,
            vah=profile.vah_price,
        )

    if not _trend_up(c, ma):
        return TradeAdvice(
            action="wait",
            reason="Trend is down — sit on your hands for now.",
            price=price,
            poc=profile.poc_price,
            val=profile.val_price,
            vah=profile.vah_price,
        )

    return TradeAdvice(
        action="wait",
        reason=(
            f"No buy yet. Watch the support floor (${profile.val_price:.2f}) "
            f"or busiest price (${profile.poc_price:.2f}) for a bounce."
        ),
        price=price,
        poc=profile.poc_price,
        val=profile.val_price,
        vah=profile.vah_price,
    )


def long_trend_status(trend_df: pd.DataFrame) -> tuple[str, float, float]:
    """Returns (label, price, long_ma) for display — e.g. ('Up', price, ma)."""
    if trend_df.empty:
        return "Unknown", 0.0, 0.0
    ok, period = _long_trend_ok(trend_df, len(trend_df) - 1)
    price = float(trend_df["close"].iloc[-1])
    if period == 0:
        return "Unknown", price, 0.0
    ma = float(trend_df["close"].rolling(period, min_periods=period).mean().iloc[-1])
    return ("Up" if ok else "Down", price, ma)


def backtest(df: pd.DataFrame, cfg: ModelConfig = DEFAULT_CONFIG, *, ticker: str = "") -> BacktestStats:
    """Round-trip stats from walk-forward scan."""
    markers = scan_trades(df, cfg, ticker=ticker)
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


def config_for_strictness(level: int) -> ModelConfig:
    """1 = loose (more trades), 5 = strict (fewer, pickier)."""
    level = max(1, min(5, level))
    loose = dict(
        volume_mult=0.65,
        wick_ratio=0.20,
        level_tolerance=2.8,
        min_gain_for_trail=0.3,
        trail_pct=2.0,
        cooldown_bars=2,
        min_vah_gain_pct=0.5,
    )
    strict = dict(
        volume_mult=1.15,
        wick_ratio=0.42,
        level_tolerance=1.3,
        min_gain_for_trail=0.8,
        trail_pct=1.2,
        cooldown_bars=5,
        min_vah_gain_pct=1.2,
    )
    t = (level - 1) / 4
    pick = lambda k: loose[k] + t * (strict[k] - loose[k])  # noqa: E731
    return ModelConfig(
        volume_mult=pick("volume_mult"),
        wick_ratio=pick("wick_ratio"),
        level_tolerance=pick("level_tolerance"),
        min_gain_for_trail=pick("min_gain_for_trail"),
        trail_pct=pick("trail_pct"),
        cooldown_bars=round(pick("cooldown_bars")),
        min_vah_gain_pct=pick("min_vah_gain_pct"),
    )


def build_config(ticker: str, period: str, n_bars: int, strictness: int) -> ModelConfig:
    """Pickiness + intraday tuning + crypto trail width."""
    return apply_asset_tuning(adapt_config(period, n_bars, config_for_strictness(strictness)), ticker)


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
        level_tolerance=cfg.level_tolerance,
        cooldown_bars=max(2, cfg.cooldown_bars // 2),
    )