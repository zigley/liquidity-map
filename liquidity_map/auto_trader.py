"""Auto-trade liquidity signals via paper trading (yfinance only)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from liquidity_map.paper_broker import (
    PaperPortfolio,
    get_position_qty,
    infer_cost_basis_from_log,
    last_buy_price_from_log,
    last_price,
    paper_buy,
    paper_sell,
    portfolio_value,
)
from liquidity_map.profile import VolumeProfile, build_volume_profile
from liquidity_map.signals import (
    DEFAULT_SIGNAL_CONFIG,
    LiquiditySignal,
    SignalConfig,
    detect_liquidity_signals,
    next_trade_side,
)

ET = ZoneInfo("America/New_York")
STATE_FILE = Path(__file__).resolve().parent.parent / ".trade_state.json"

Action = Literal["buy", "sell", "hold", "skip"]


@dataclass
class TradeConfig:
    dry_run: bool = True
    trade_amount_usd: float = 100.0
    min_confluence: int = 3
    signal_config: SignalConfig = field(default_factory=lambda: DEFAULT_SIGNAL_CONFIG)
    max_daily_trades: int = 5
    require_liquid_spread: bool = False
    sell_full_position: bool = True
    use_confirmed_bar: bool = True
    paper_starting_cash: float = 10_000.0


@dataclass
class TradeResult:
    action: Action
    symbol: str
    signal: LiquiditySignal | None
    message: str
    order_id: str | None = None
    dry_run: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(ET).isoformat())


@dataclass
class TradeState:
    executed_keys: list[str] = field(default_factory=list)
    daily_trade_count: int = 0
    daily_trade_date: str = ""
    trade_log: list[dict] = field(default_factory=list)
    paper_cash: float = 10_000.0
    paper_positions: dict[str, float] = field(default_factory=dict)
    paper_cost_basis: dict[str, float] = field(default_factory=dict)


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def load_trade_config() -> TradeConfig:
    from dotenv import load_dotenv

    load_dotenv()
    return TradeConfig(
        dry_run=_env_bool("AUTO_TRADE_DRY_RUN", False),
        trade_amount_usd=float(os.getenv("AUTO_TRADE_AMOUNT_USD", "100")),
        min_confluence=int(os.getenv("AUTO_TRADE_MIN_CONFLUENCE", os.getenv("AUTO_TRADE_MIN_STRENGTH", "3"))),
        max_daily_trades=int(os.getenv("AUTO_TRADE_MAX_DAILY", "5")),
        require_liquid_spread=_env_bool("AUTO_TRADE_REQUIRE_LIQUID_SPREAD", False),
        paper_starting_cash=float(os.getenv("PAPER_STARTING_CASH", "10000")),
    )


def signal_key(symbol: str, signal: LiquiditySignal) -> str:
    return f"{symbol}|{signal.datetime}|{signal.side}|{signal.reason}"


def load_trade_state(path: Path = STATE_FILE) -> TradeState:
    if not path.exists():
        cfg = load_trade_config()
        return TradeState(paper_cash=cfg.paper_starting_cash)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TradeState(
            executed_keys=list(raw.get("executed_keys", [])),
            daily_trade_count=int(raw.get("daily_trade_count", 0)),
            daily_trade_date=str(raw.get("daily_trade_date", "")),
            trade_log=list(raw.get("trade_log", [])),
            paper_cash=float(raw.get("paper_cash", 10_000)),
            paper_positions=dict(raw.get("paper_positions", {})),
            paper_cost_basis=dict(raw.get("paper_cost_basis", {})),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return TradeState()


def save_trade_state(state: TradeState, path: Path = STATE_FILE) -> None:
    path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")


def get_paper_portfolio(state: TradeState, symbol: str | None = None) -> PaperPortfolio:
    portfolio = PaperPortfolio(
        cash=state.paper_cash,
        positions=dict(state.paper_positions),
        cost_basis=dict(state.paper_cost_basis),
    )
    if not symbol:
        return portfolio

    sym = symbol.upper()
    state_qty = float(portfolio.positions.get(sym, 0.0))
    has_basis = float(portfolio.cost_basis.get(sym, 0.0)) > 0

    if not has_basis:
        log_qty, log_avg = infer_cost_basis_from_log(state.trade_log, sym)
        if state_qty > 0:
            if log_avg > 0:
                portfolio.cost_basis[sym] = log_avg
            else:
                fallback = last_buy_price_from_log(state.trade_log, sym)
                if fallback > 0:
                    portfolio.cost_basis[sym] = fallback
        elif log_qty > 0 and log_avg > 0:
            portfolio.positions[sym] = log_qty
            portfolio.cost_basis[sym] = log_avg

    return portfolio


def _reset_daily_counter(state: TradeState) -> None:
    today = date.today().isoformat()
    if state.daily_trade_date != today:
        state.daily_trade_date = today
        state.daily_trade_count = 0


def get_actionable_signal(
    df: pd.DataFrame,
    profile: VolumeProfile,
    min_confluence: int = 3,
    use_confirmed_bar: bool = True,
    signal_config: SignalConfig | None = None,
    in_position: bool = False,
) -> LiquiditySignal | None:
    if len(df) < 2:
        return None

    base = signal_config or DEFAULT_SIGNAL_CONFIG
    cfg = SignalConfig(
        require_trend_filter=base.require_trend_filter,
        require_rejection_wick=base.require_rejection_wick,
        trend_ma_period=base.trend_ma_period,
        volume_spike_pct=base.volume_spike_pct,
        min_volume_pct=base.min_volume_pct,
        min_confluence=min_confluence,
        cooldown_bars=base.cooldown_bars,
        edge_only=base.edge_only,
        wick_ratio=base.wick_ratio,
    )
    signals = detect_liquidity_signals(df, profile, config=cfg)
    if not signals:
        return None

    needed = next_trade_side(in_position=in_position)
    bar_dt = df["datetime"].iloc[-2 if use_confirmed_bar else -1]
    bar_ts = pd.Timestamp(bar_dt)
    for signal in reversed(signals):
        if pd.Timestamp(signal.datetime) == bar_ts and signal.side == needed:
            return signal
    return None


def _market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time


def evaluate_and_trade(
    symbol: str,
    df: pd.DataFrame,
    config: TradeConfig | None = None,
    state: TradeState | None = None,
    state_path: Path = STATE_FILE,
) -> TradeResult:
    """Evaluate the latest liquidity signal and paper-trade at the last close."""
    cfg = config or load_trade_config()
    st = state if state is not None else load_trade_state(state_path)
    sym = symbol.strip().upper()
    price = last_price(df)

    _reset_daily_counter(st)

    if not _market_open():
        return TradeResult(action="skip", symbol=sym, signal=None, message="Market closed (9:30–16:00 ET)", dry_run=cfg.dry_run)

    profile = build_volume_profile(df)
    portfolio = get_paper_portfolio(st, sym)
    in_position = get_position_qty(portfolio, sym) > 0
    needed = next_trade_side(in_position=in_position)

    signal = get_actionable_signal(
        df,
        profile,
        cfg.min_confluence,
        cfg.use_confirmed_bar,
        cfg.signal_config,
        in_position=in_position,
    )
    if signal is None:
        status = "holding — waiting for SELL" if in_position else "flat — waiting for BUY"
        return TradeResult(
            action="hold",
            symbol=sym,
            signal=None,
            message=f"No {needed.upper()} signal on latest bar ({status})",
            dry_run=cfg.dry_run,
        )

    key = signal_key(sym, signal)
    if key in st.executed_keys:
        return TradeResult(action="skip", symbol=sym, signal=signal, message="Signal already traded", dry_run=cfg.dry_run)

    if st.daily_trade_count >= cfg.max_daily_trades:
        return TradeResult(action="skip", symbol=sym, signal=signal, message="Daily trade limit reached", dry_run=cfg.dry_run)

    if cfg.dry_run:
        side = signal.side.upper()
        amt = f"${cfg.trade_amount_usd:.2f}" if signal.side == "buy" else "full position"
        return TradeResult(
            action="skip",
            symbol=sym,
            signal=signal,
            message=f"Signal only: would {side} {sym} ({amt}) @ ${price:.2f} — {signal.reason}",
            dry_run=True,
        )

    sell_proceeds = 0.0

    try:
        if signal.side == "buy":
            if in_position:
                return TradeResult(action="skip", symbol=sym, signal=signal, message="BUY skipped — already holding", dry_run=cfg.dry_run)
            order_id, msg, portfolio = paper_buy(portfolio, sym, cfg.trade_amount_usd, price)
            action: Action = "buy"
        else:
            sell_qty = get_position_qty(portfolio, sym)
            if sell_qty <= 0:
                return TradeResult(action="skip", symbol=sym, signal=signal, message="SELL skipped — not holding", dry_run=True)
            order_id, msg, portfolio = paper_sell(portfolio, sym, price)
            action = "sell"
            sell_proceeds = sell_qty * price
    except Exception as exc:
        return TradeResult(action="skip", symbol=sym, signal=signal, message=f"Order failed: {exc}", dry_run=True)

    st.paper_cash = portfolio.cash
    st.paper_positions = portfolio.positions
    st.paper_cost_basis = portfolio.cost_basis
    st.executed_keys.append(key)
    st.daily_trade_count += 1
    equity = portfolio_value(portfolio, {sym: price})
    log_entry = {
        "timestamp": datetime.now(ET).isoformat(),
        "symbol": sym,
        "action": action,
        "signal_reason": signal.reason,
        "signal_confluence": signal.confluence,
        "order_id": order_id,
        "price": price,
        "amount_usd": cfg.trade_amount_usd if action == "buy" else round(sell_proceeds, 2) if action == "sell" else 0,
        "paper_equity": round(equity, 2),
        "message": msg,
    }
    st.trade_log.append(log_entry)
    st.trade_log = st.trade_log[-100:]
    save_trade_state(st, state_path)

    return TradeResult(action=action, symbol=sym, signal=signal, message=f"{msg} | equity ${equity:,.2f}", order_id=order_id, dry_run=False)