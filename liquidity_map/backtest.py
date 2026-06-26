"""Walk-forward backtest for liquidity signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from liquidity_map.profile import build_volume_profile
from liquidity_map.signals import LiquiditySignal, detect_liquidity_signals


@dataclass(frozen=True)
class SignalOutcome:
    datetime: object
    side: str
    reason: str
    strength: int
    entry_price: float
    returns: dict[int, float] = field(default_factory=dict)
    wins: dict[int, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class HorizonStats:
    horizon: int
    n: int
    win_rate: float
    avg_return_pct: float
    median_return_pct: float


@dataclass(frozen=True)
class BacktestResult:
    outcomes: tuple[SignalOutcome, ...]
    horizons: tuple[int, ...]
    rolling_window: int
    use_rolling_profile: bool
    min_strength: int
    overall: tuple[HorizonStats, ...]
    buy: tuple[HorizonStats, ...]
    sell: tuple[HorizonStats, ...]


def _signal_at_bar(window: pd.DataFrame, min_strength: int) -> LiquiditySignal | None:
    profile = build_volume_profile(window)
    signals = detect_liquidity_signals(window, profile)
    if not signals:
        return None
    bar_dt = pd.Timestamp(window.iloc[-1]["datetime"])
    for signal in reversed(signals):
        if pd.Timestamp(signal.datetime) == bar_dt and signal.strength >= min_strength:
            return signal
    return None


def _forward_return(side: str, entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "buy":
        return (exit_price - entry) / entry * 100.0
    return (entry - exit_price) / entry * 100.0


def _summarize(
    outcomes: list[SignalOutcome],
    side_filter: str | None,
    horizons: tuple[int, ...],
) -> tuple[HorizonStats, ...]:
    filtered = [o for o in outcomes if side_filter is None or o.side == side_filter]
    stats: list[HorizonStats] = []
    for h in horizons:
        with_data = [o for o in filtered if h in o.returns]
        if not with_data:
            stats.append(HorizonStats(horizon=h, n=0, win_rate=0.0, avg_return_pct=0.0, median_return_pct=0.0))
            continue
        returns = [o.returns[h] for o in with_data]
        wins = [o.wins[h] for o in with_data]
        stats.append(
            HorizonStats(
                horizon=h,
                n=len(with_data),
                win_rate=sum(wins) / len(wins) * 100.0,
                avg_return_pct=float(sum(returns) / len(returns)),
                median_return_pct=float(pd.Series(returns).median()),
            )
        )
    return tuple(stats)


def run_backtest(
    df: pd.DataFrame,
    *,
    rolling_window: int = 60,
    min_strength: int = 2,
    horizons: tuple[int, ...] = (5, 10, 20),
    use_rolling_profile: bool = True,
    warmup_bars: int = 20,
) -> BacktestResult:
    """
    Walk-forward backtest: profile is built only from past bars (no look-ahead).

    Entry at signal bar close; exit at close N bars later.
    Buy wins when price rises; sell wins when price falls.
    """
    if df.empty or len(df) < warmup_bars + max(horizons) + 1:
        empty = tuple(
            HorizonStats(h, 0, 0.0, 0.0, 0.0) for h in horizons
        )
        return BacktestResult(
            outcomes=(),
            horizons=horizons,
            rolling_window=rolling_window,
            use_rolling_profile=use_rolling_profile,
            min_strength=min_strength,
            overall=empty,
            buy=empty,
            sell=empty,
        )

    max_h = max(horizons)
    outcomes: list[SignalOutcome] = []

    for i in range(warmup_bars, len(df) - max_h):
        if use_rolling_profile:
            start = max(0, i - rolling_window + 1)
            window = df.iloc[start : i + 1].copy()
        else:
            window = df.iloc[: i + 1].copy()

        signal = _signal_at_bar(window, min_strength)
        if signal is None:
            continue

        entry = float(df.iloc[i]["close"])
        returns: dict[int, float] = {}
        wins: dict[int, bool] = {}

        for h in horizons:
            exit_price = float(df.iloc[i + h]["close"])
            ret = _forward_return(signal.side, entry, exit_price)
            returns[h] = ret
            wins[h] = ret > 0

        outcomes.append(
            SignalOutcome(
                datetime=signal.datetime,
                side=signal.side,
                reason=signal.reason,
                strength=signal.strength,
                entry_price=entry,
                returns=returns,
                wins=wins,
            )
        )

    return BacktestResult(
        outcomes=tuple(outcomes),
        horizons=horizons,
        rolling_window=rolling_window,
        use_rolling_profile=use_rolling_profile,
        min_strength=min_strength,
        overall=_summarize(outcomes, None, horizons),
        buy=_summarize(outcomes, "buy", horizons),
        sell=_summarize(outcomes, "sell", horizons),
    )