"""Buy/sell signals with trend, rejection wick, and confluence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from liquidity_map.profile import VolumeProfile

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class SignalConfig:
    require_trend_filter: bool = True
    require_rejection_wick: bool = True
    trend_ma_period: int = 50
    volume_spike_pct: float = 1.3
    min_volume_pct: float = 0.8
    min_confluence: int = 3
    cooldown_bars: int = 3
    edge_only: bool = True
    wick_ratio: float = 0.40


DEFAULT_SIGNAL_CONFIG = SignalConfig(
    require_trend_filter=False,
    require_rejection_wick=False,
    min_confluence=3,
)

STRICT_SIGNAL_CONFIG = SignalConfig(
    require_trend_filter=True,
    require_rejection_wick=True,
    min_confluence=3,
)

LEGACY_SIGNAL_CONFIG = SignalConfig(
    require_trend_filter=False,
    require_rejection_wick=False,
    min_confluence=1,
    edge_only=False,
    volume_spike_pct=0.8,
)


@dataclass(frozen=True)
class LiquiditySignal:
    datetime: object
    price: float
    side: Side
    reason: str
    confluence: int
    tags: tuple[str, ...] = ()

    @property
    def strength(self) -> int:
        return self.confluence


@dataclass(frozen=True)
class _Candidate:
    side: Side
    reason: str
    level_label: str
    marker_price: float


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


def _is_high_liquidity(profile: VolumeProfile, bin_idx: int) -> bool:
    vol = profile.volumes[bin_idx]
    total = profile.volumes.sum()
    rank = float(vol / total) if total > 0 else 0.0
    return bool(profile.hvn_mask[bin_idx]) or rank >= 0.012


def _trend_aligned(side: Side, close: float, ma: float) -> bool:
    if np.isnan(ma):
        return True
    return close > ma if side == "buy" else close < ma


def _rejection_wick(side: Side, o: float, h: float, l: float, c: float, ratio: float) -> bool:
    rng = h - l
    if rng <= 0:
        return False
    if side == "buy":
        return (min(o, c) - l) / rng >= ratio and c > l
    return (h - max(o, c)) / rng >= ratio and c < h


def _volume_spike(vol: float, median_vol: float, spike_pct: float) -> bool:
    return median_vol <= 0 or vol >= median_vol * spike_pct


def _find_buy_candidate(
    o: float,
    h: float,
    l: float,
    c: float,
    prev_close: float,
    profile: VolumeProfile,
    tol: float,
    low_bin: int,
    config: SignalConfig,
) -> _Candidate | None:
    if c <= o:
        return None

    for label, level in (("VAL", profile.val_price), ("POC", profile.poc_price)):
        if _near(l, level, tol) and c > level:
            return _Candidate("buy", f"Bounce off {label} ({level:.2f})", label, l - tol * 0.25)

    if _near(l, profile.poc_price, tol * 1.2) and prev_close < profile.poc_price <= c:
        return _Candidate("buy", "POC reclaim", "POC", l - tol * 0.25)

    if config.edge_only:
        return None

    if _is_high_liquidity(profile, low_bin) and c > l + (h - l) * 0.45:
        center = float(profile.bin_centers[low_bin])
        return _Candidate("buy", f"HVN support ({center:.2f})", "HVN", l - tol * 0.25)

    return None


def _find_sell_candidate(
    o: float,
    h: float,
    l: float,
    c: float,
    prev_close: float,
    profile: VolumeProfile,
    tol: float,
    high_bin: int,
    config: SignalConfig,
) -> _Candidate | None:
    if c >= o:
        return None

    for label, level in (("VAH", profile.vah_price), ("POC", profile.poc_price)):
        if _near(h, level, tol) and c < level:
            return _Candidate("sell", f"Reject at {label} ({level:.2f})", label, h + tol * 0.25)

    if _near(h, profile.poc_price, tol * 1.2) and prev_close > profile.poc_price >= c:
        return _Candidate("sell", "POC loss", "POC", h + tol * 0.25)

    if config.edge_only:
        return None

    if _is_high_liquidity(profile, high_bin) and c < h - (h - l) * 0.45:
        center = float(profile.bin_centers[high_bin])
        return _Candidate("sell", f"HVN resistance ({center:.2f})", "HVN", h + tol * 0.25)

    return None


def _score_candidate(
    candidate: _Candidate,
    *,
    side: Side,
    o: float,
    h: float,
    l: float,
    c: float,
    vol: float,
    touch_bin: int,
    profile: VolumeProfile,
    ma: float,
    median_vol: float,
    config: SignalConfig,
) -> tuple[int, tuple[str, ...]]:
    score = 0
    tags: list[str] = []

    if candidate.level_label in ("VAL", "VAH"):
        score += 1
        tags.append("edge")
    elif candidate.level_label == "POC":
        score += 1
        tags.append("poc")

    if _trend_aligned(side, c, ma):
        score += 1
        tags.append("trend")

    if _rejection_wick(side, o, h, l, c, config.wick_ratio):
        score += 1
        tags.append("wick")

    if _volume_spike(vol, median_vol, config.volume_spike_pct):
        score += 1
        tags.append("vol")

    if _is_high_liquidity(profile, touch_bin):
        score += 1
        tags.append("hvn")

    return min(score, 5), tuple(tags)


def detect_liquidity_signals(
    df: pd.DataFrame,
    profile: VolumeProfile,
    config: SignalConfig | None = None,
    *,
    min_volume_pct: float | None = None,
    cooldown_bars: int | None = None,
) -> list[LiquiditySignal]:
    """Emit signals when price rejects liquid zones with enough confluence."""
    cfg = config or DEFAULT_SIGNAL_CONFIG
    if min_volume_pct is not None or cooldown_bars is not None:
        cfg = SignalConfig(
            require_trend_filter=cfg.require_trend_filter,
            require_rejection_wick=cfg.require_rejection_wick,
            trend_ma_period=cfg.trend_ma_period,
            volume_spike_pct=cfg.volume_spike_pct,
            min_volume_pct=min_volume_pct if min_volume_pct is not None else cfg.min_volume_pct,
            min_confluence=cfg.min_confluence,
            cooldown_bars=cooldown_bars if cooldown_bars is not None else cfg.cooldown_bars,
            edge_only=cfg.edge_only,
            wick_ratio=cfg.wick_ratio,
        )

    if len(df) < max(3, cfg.trend_ma_period):
        return []

    tol = _tolerance(profile)
    median_vol = float(df["volume"].median()) if df["volume"].median() > 0 else 0.0
    ma_series = df["close"].rolling(cfg.trend_ma_period, min_periods=cfg.trend_ma_period).mean()

    signals: list[LiquiditySignal] = []
    last_buy_idx = -cfg.cooldown_bars
    last_sell_idx = -cfg.cooldown_bars

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        vol = float(row["volume"])
        if median_vol > 0 and vol < median_vol * cfg.min_volume_pct:
            continue

        ma = float(ma_series.iloc[i])
        low_bin = _bin_index(l, profile.bin_edges)
        high_bin = _bin_index(h, profile.bin_edges)
        prev_close = float(prev["close"])

        buy = _find_buy_candidate(o, h, l, c, prev_close, profile, tol, low_bin, cfg)
        if buy and i - last_buy_idx >= cfg.cooldown_bars:
            if cfg.require_trend_filter and not _trend_aligned("buy", c, ma):
                buy = None
            elif cfg.require_rejection_wick and not _rejection_wick("buy", o, h, l, c, cfg.wick_ratio):
                buy = None

        if buy:
            score, tags = _score_candidate(
                buy,
                side="buy",
                o=o,
                h=h,
                l=l,
                c=c,
                vol=vol,
                touch_bin=low_bin,
                profile=profile,
                ma=ma,
                median_vol=median_vol,
                config=cfg,
            )
            if score >= cfg.min_confluence:
                signals.append(
                    LiquiditySignal(
                        datetime=row["datetime"],
                        price=buy.marker_price,
                        side="buy",
                        reason=f"{buy.reason} · score {score}/5 ({', '.join(tags)})",
                        confluence=score,
                        tags=tags,
                    )
                )
                last_buy_idx = i

        sell = _find_sell_candidate(o, h, l, c, prev_close, profile, tol, high_bin, cfg)
        if sell and i - last_sell_idx >= cfg.cooldown_bars:
            if cfg.require_trend_filter and not _trend_aligned("sell", c, ma):
                sell = None
            elif cfg.require_rejection_wick and not _rejection_wick("sell", o, h, l, c, cfg.wick_ratio):
                sell = None

        if sell:
            score, tags = _score_candidate(
                sell,
                side="sell",
                o=o,
                h=h,
                l=l,
                c=c,
                vol=vol,
                touch_bin=high_bin,
                profile=profile,
                ma=ma,
                median_vol=median_vol,
                config=cfg,
            )
            if score >= cfg.min_confluence:
                signals.append(
                    LiquiditySignal(
                        datetime=row["datetime"],
                        price=sell.marker_price,
                        side="sell",
                        reason=f"{sell.reason} · score {score}/5 ({', '.join(tags)})",
                        confluence=score,
                        tags=tags,
                    )
                )
                last_sell_idx = i

    return signals