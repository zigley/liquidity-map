"""Exit rules to lock in upside before price rides back to POC."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from liquidity_map.profile import VolumeProfile, build_volume_profile


@dataclass(frozen=True)
class ExitConfig:
    """Sell on pullback from peak — don't wait for price to drift back to POC."""

    trail_pct: float = 1.5
    trail_pct_after_vah: float = 1.0
    take_profit_at_vah: bool = True
    min_vah_gain_pct: float = 1.0
    min_gain_pct_before_trail: float = 0.5


DEFAULT_EXIT_CONFIG = ExitConfig()


@dataclass(frozen=True)
class ExitLevels:
    take_profit: float | None
    trail_stop: float | None
    peak: float
    entry: float


def compute_exit_levels(
    entry: float,
    peak: float,
    profile: VolumeProfile,
    config: ExitConfig = DEFAULT_EXIT_CONFIG,
) -> ExitLevels:
    vah = profile.vah_price
    vah_target = entry * (1 + config.min_vah_gain_pct / 100)
    tp = vah if config.take_profit_at_vah and vah >= vah_target else None

    trail_pct = config.trail_pct
    if peak >= vah >= entry:
        trail_pct = config.trail_pct_after_vah

    trail: float | None = None
    min_peak = entry * (1 + config.min_gain_pct_before_trail / 100)
    if peak >= min_peak:
        trail = peak * (1 - trail_pct / 100)
    return ExitLevels(take_profit=tp, trail_stop=trail, peak=peak, entry=entry)


def check_bar_exit(
    *,
    high: float,
    close: float,
    levels: ExitLevels,
    config: ExitConfig = DEFAULT_EXIT_CONFIG,
) -> tuple[bool, str]:
    if levels.take_profit is not None and high >= levels.take_profit:
        return True, f"Take profit at VAH ${levels.take_profit:.2f}"
    if levels.trail_stop is not None and close <= levels.trail_stop:
        return True, f"Trail stop ${levels.trail_stop:.2f} ({config.trail_pct}% below peak ${levels.peak:.2f})"
    return False, ""


def peak_since_index(df: pd.DataFrame, start_idx: int) -> float:
    window = df.iloc[start_idx:]
    return float(window["high"].max()) if len(window) else float(df.iloc[start_idx]["close"])


def find_bar_index(df: pd.DataFrame, dt: object) -> int | None:
    ts = pd.Timestamp(dt)
    matches = df.index[df["datetime"].apply(lambda x: pd.Timestamp(x) == ts)].tolist()
    return int(matches[0]) if matches else None


def simulate_exit_after_buy(
    df: pd.DataFrame,
    buy_idx: int,
    profile: VolumeProfile | None = None,
    config: ExitConfig = DEFAULT_EXIT_CONFIG,
    max_bars: int | None = None,
) -> tuple[int, str, float]:
    """
    Walk forward from buy bar; exit on first VAH touch or trailing stop.
    Returns (exit_bar_index, reason, exit_close).
    """
    if buy_idx >= len(df) - 1:
        last = float(df.iloc[-1]["close"])
        return len(df) - 1, "End of range", last

    entry = float(df.iloc[buy_idx]["close"])
    peak = float(df.iloc[buy_idx]["high"])
    end = len(df) if max_bars is None else min(len(df), buy_idx + max_bars + 1)

    for i in range(buy_idx + 1, end):
        row = df.iloc[i]
        peak = max(peak, float(row["high"]))
        prof = profile if profile is not None else build_volume_profile(df.iloc[: i + 1])
        levels = compute_exit_levels(entry, peak, prof, config)
        hit, reason = check_bar_exit(
            high=float(row["high"]),
            close=float(row["close"]),
            levels=levels,
            config=config,
        )
        if hit:
            return i, reason, float(row["close"])

    last_i = end - 1
    return last_i, "No exit trigger", float(df.iloc[last_i]["close"])


def smart_pair_return_pct(
    df: pd.DataFrame,
    buy_dt: object,
    config: ExitConfig = DEFAULT_EXIT_CONFIG,
) -> tuple[float | None, str]:
    buy_idx = find_bar_index(df, buy_dt)
    if buy_idx is None:
        return None, ""
    entry = float(df.iloc[buy_idx]["close"])
    if entry <= 0:
        return None, ""
    _, reason, exit_px = simulate_exit_after_buy(df, buy_idx, config=config)
    return (exit_px - entry) / entry * 100.0, reason


def current_exit_plan(
    df: pd.DataFrame,
    entry_price: float,
    config: ExitConfig = DEFAULT_EXIT_CONFIG,
) -> tuple[ExitLevels, str]:
    """Live plan for an open long position."""
    entry_idx = max(0, len(df) - 50)
    for i in range(len(df) - 1, -1, -1):
        if float(df.iloc[i]["close"]) <= entry_price * 1.002:
            entry_idx = i
            break
    peak = peak_since_index(df, entry_idx)
    profile = build_volume_profile(df)
    levels = compute_exit_levels(entry_price, peak, profile, config)

    parts: list[str] = []
    if levels.take_profit:
        parts.append(f"**Take profit** at VAH ${levels.take_profit:.2f}")
    if levels.trail_stop:
        parts.append(f"**Trail stop** at ${levels.trail_stop:.2f} ({config.trail_pct}% below peak ${peak:.2f})")
    if not parts:
        parts.append(f"Up {config.min_gain_pct_before_trail:.1f}% to activate trail stop")
    parts.append("POC sell signal is **backup only** (often late)")
    return levels, " · ".join(parts)