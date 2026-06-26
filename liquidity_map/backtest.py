"""Walk-forward backtest for liquidity signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from liquidity_map.profile import build_volume_profile
from liquidity_map.signals import (
    DEFAULT_SIGNAL_CONFIG,
    LEGACY_SIGNAL_CONFIG,
    STRICT_SIGNAL_CONFIG,
    LiquiditySignal,
    Side,
    SignalConfig,
    build_trade_pairs,
    detect_liquidity_signals,
    pair_return_pct,
)


@dataclass(frozen=True)
class SignalOutcome:
    datetime: object
    side: str
    reason: str
    confluence: int
    entry_price: float
    trade_label: str = ""
    returns: dict[int, float] = field(default_factory=dict)
    wins: dict[int, bool] = field(default_factory=dict)

    @property
    def strength(self) -> int:
        return self.confluence


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
    signal_config: SignalConfig
    overall: tuple[HorizonStats, ...]
    buy: tuple[HorizonStats, ...]
    sell: tuple[HorizonStats, ...]
    round_trip_win_rate: float = 0.0
    avg_round_trip_pct: float = 0.0
    completed_pairs: int = 0

    @property
    def min_strength(self) -> int:
        return self.signal_config.min_confluence


def _signal_at_bar(window: pd.DataFrame, config: SignalConfig) -> LiquiditySignal | None:
    profile = build_volume_profile(window)
    signals = detect_liquidity_signals(window, profile, config=config)
    if not signals:
        return None
    bar_dt = pd.Timestamp(window.iloc[-1]["datetime"])
    for signal in reversed(signals):
        if pd.Timestamp(signal.datetime) == bar_dt:
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
    signal_config: SignalConfig | None = None,
    horizons: tuple[int, ...] = (5, 10, 20),
    use_rolling_profile: bool = True,
    warmup_bars: int | None = None,
) -> BacktestResult:
    """
    Walk-forward backtest: profile is built only from past bars (no look-ahead).

    Entry at signal bar close; exit at close N bars later.
    Buy wins when price rises; sell wins when price falls.
    """
    cfg = signal_config or DEFAULT_SIGNAL_CONFIG
    warmup = warmup_bars if warmup_bars is not None else max(20, cfg.trend_ma_period)

    if df.empty or len(df) < warmup + max(horizons) + 1:
        empty = tuple(HorizonStats(h, 0, 0.0, 0.0, 0.0) for h in horizons)
        return BacktestResult(
            outcomes=(),
            horizons=horizons,
            rolling_window=rolling_window,
            use_rolling_profile=use_rolling_profile,
            signal_config=cfg,
            overall=empty,
            buy=empty,
            sell=empty,
        )

    max_h = max(horizons)
    outcomes: list[SignalOutcome] = []
    next_side: Side = "buy"
    buy_n = 0
    sell_n = 0

    for i in range(warmup, len(df) - max_h):
        if use_rolling_profile:
            start = max(0, i - rolling_window + 1)
            window = df.iloc[start : i + 1].copy()
        else:
            window = df.iloc[: i + 1].copy()

        signal = _signal_at_bar(window, cfg)
        if signal is None or signal.side != next_side:
            continue

        if signal.side == "buy":
            buy_n += 1
            label = f"B{buy_n}"
        else:
            sell_n += 1
            label = f"S{sell_n}"

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
                confluence=signal.confluence,
                entry_price=entry,
                trade_label=label,
                returns=returns,
                wins=wins,
            )
        )
        next_side = "sell" if next_side == "buy" else "buy"

    alt_signals = [
        LiquiditySignal(
            datetime=o.datetime,
            price=o.entry_price,
            side=o.side,  # type: ignore[arg-type]
            reason=o.reason,
            confluence=o.confluence,
            trade_label=o.trade_label,
        )
        for o in outcomes
    ]
    pairs = build_trade_pairs(alt_signals)
    completed = [p for p in pairs if not p.is_open]
    trip_returns = [r for p in completed if (r := pair_return_pct(p, df)) is not None]
    rt_win = sum(1 for r in trip_returns if r > 0) / len(trip_returns) * 100 if trip_returns else 0.0
    rt_avg = sum(trip_returns) / len(trip_returns) if trip_returns else 0.0

    return BacktestResult(
        outcomes=tuple(outcomes),
        horizons=horizons,
        rolling_window=rolling_window,
        use_rolling_profile=use_rolling_profile,
        signal_config=cfg,
        overall=_summarize(outcomes, None, horizons),
        buy=_summarize(outcomes, "buy", horizons),
        sell=_summarize(outcomes, "sell", horizons),
        round_trip_win_rate=rt_win,
        avg_round_trip_pct=rt_avg,
        completed_pairs=len(completed),
    )


def compare_backtest(
    df: pd.DataFrame,
    *,
    rolling_window: int = 60,
    horizons: tuple[int, ...] = (5, 10, 20),
    use_rolling_profile: bool = True,
) -> tuple[BacktestResult, BacktestResult]:
    """Run strict (filtered) vs legacy signal rules side by side."""
    strict = run_backtest(
        df,
        rolling_window=rolling_window,
        signal_config=STRICT_SIGNAL_CONFIG,
        horizons=horizons,
        use_rolling_profile=use_rolling_profile,
    )
    legacy = run_backtest(
        df,
        rolling_window=rolling_window,
        signal_config=LEGACY_SIGNAL_CONFIG,
        horizons=horizons,
        use_rolling_profile=use_rolling_profile,
    )
    return strict, legacy