"""Auto-trade liquidity signals via Robinhood."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from liquidity_map.data import Quote, fetch_quote
from liquidity_map.liquidity_score import quote_rating
from liquidity_map.profile import VolumeProfile, build_volume_profile
from liquidity_map.signals import LiquiditySignal, detect_liquidity_signals

ET = ZoneInfo("America/New_York")
STATE_FILE = Path(__file__).resolve().parent.parent / ".trade_state.json"

Action = Literal["buy", "sell", "hold", "skip"]


@dataclass
class TradeConfig:
    dry_run: bool = True
    trade_amount_usd: float = 100.0
    min_strength: int = 2
    max_daily_trades: int = 5
    require_liquid_spread: bool = True
    sell_full_position: bool = True
    use_confirmed_bar: bool = True


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


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def load_trade_config() -> TradeConfig:
    from dotenv import load_dotenv

    load_dotenv()
    return TradeConfig(
        dry_run=_env_bool("AUTO_TRADE_DRY_RUN", True),
        trade_amount_usd=float(os.getenv("AUTO_TRADE_AMOUNT_USD", "100")),
        min_strength=int(os.getenv("AUTO_TRADE_MIN_STRENGTH", "2")),
        max_daily_trades=int(os.getenv("AUTO_TRADE_MAX_DAILY", "5")),
        require_liquid_spread=_env_bool("AUTO_TRADE_REQUIRE_LIQUID_SPREAD", True),
    )


def signal_key(symbol: str, signal: LiquiditySignal) -> str:
    return f"{symbol}|{signal.datetime}|{signal.side}|{signal.reason}"


def load_trade_state(path: Path = STATE_FILE) -> TradeState:
    if not path.exists():
        return TradeState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TradeState(
            executed_keys=list(raw.get("executed_keys", [])),
            daily_trade_count=int(raw.get("daily_trade_count", 0)),
            daily_trade_date=str(raw.get("daily_trade_date", "")),
            trade_log=list(raw.get("trade_log", [])),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return TradeState()


def save_trade_state(state: TradeState, path: Path = STATE_FILE) -> None:
    path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")


def _reset_daily_counter(state: TradeState) -> None:
    today = date.today().isoformat()
    if state.daily_trade_date != today:
        state.daily_trade_date = today
        state.daily_trade_count = 0


def get_actionable_signal(
    df: pd.DataFrame,
    profile: VolumeProfile,
    min_strength: int = 2,
    use_confirmed_bar: bool = True,
) -> LiquiditySignal | None:
    """Return the signal on the latest confirmed bar, if any."""
    if len(df) < 2:
        return None

    signals = detect_liquidity_signals(df, profile)
    if not signals:
        return None

    bar_dt = df["datetime"].iloc[-2 if use_confirmed_bar else -1]
    bar_ts = pd.Timestamp(bar_dt)
    for signal in reversed(signals):
        if pd.Timestamp(signal.datetime) == bar_ts and signal.strength >= min_strength:
            return signal
    return None


def _market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time


def _get_position_qty(symbol: str) -> float:
    import robin_stocks.robinhood as rh

    holdings = rh.account.build_holdings() or {}
    pos = holdings.get(symbol.upper())
    if not pos:
        return 0.0
    return float(pos.get("quantity", 0) or 0)


def _place_buy(symbol: str, amount_usd: float, dry_run: bool) -> tuple[str | None, str]:
    if dry_run:
        return f"dry-buy-{int(datetime.now().timestamp())}", f"DRY RUN buy ${amount_usd:.2f} of {symbol}"

    import robin_stocks.robinhood as rh

    resp = rh.orders.order_buy_fractional_by_price(symbol, amount_usd)
    if not resp:
        raise RuntimeError("Buy order returned empty response")
    order_id = resp.get("id") or resp.get("order_id") or str(resp)
    return str(order_id), f"Bought ${amount_usd:.2f} of {symbol}"


def _place_sell(symbol: str, qty: float, dry_run: bool) -> tuple[str | None, str]:
    if qty <= 0:
        raise RuntimeError(f"No {symbol} shares to sell")

    if dry_run:
        return f"dry-sell-{int(datetime.now().timestamp())}", f"DRY RUN sell {qty:.4f} shares of {symbol}"

    import robin_stocks.robinhood as rh

    resp = rh.orders.order_sell_market(symbol, qty)
    if not resp:
        raise RuntimeError("Sell order returned empty response")
    order_id = resp.get("id") or resp.get("order_id") or str(resp)
    return str(order_id), f"Sold {qty:.4f} shares of {symbol}"


def evaluate_and_trade(
    symbol: str,
    df: pd.DataFrame,
    config: TradeConfig | None = None,
    state: TradeState | None = None,
    quote: Quote | None = None,
    state_path: Path = STATE_FILE,
) -> TradeResult:
    """Evaluate the latest liquidity signal and optionally place a Robinhood order."""
    cfg = config or load_trade_config()
    st = state if state is not None else load_trade_state(state_path)
    sym = symbol.strip().upper()

    _reset_daily_counter(st)

    if not _market_open():
        return TradeResult(action="skip", symbol=sym, signal=None, message="Market closed (9:30–16:00 ET)", dry_run=cfg.dry_run)

    profile = build_volume_profile(df)
    signal = get_actionable_signal(df, profile, cfg.min_strength, cfg.use_confirmed_bar)
    if signal is None:
        return TradeResult(action="hold", symbol=sym, signal=None, message="No actionable liquidity signal on latest bar", dry_run=cfg.dry_run)

    key = signal_key(sym, signal)
    if key in st.executed_keys:
        return TradeResult(action="skip", symbol=sym, signal=signal, message="Signal already traded", dry_run=cfg.dry_run)

    if st.daily_trade_count >= cfg.max_daily_trades:
        return TradeResult(action="skip", symbol=sym, signal=signal, message="Daily trade limit reached", dry_run=cfg.dry_run)

    if cfg.require_liquid_spread:
        q = quote or fetch_quote(sym)
        rating = quote_rating(q)
        if rating.label == "Illiquid":
            return TradeResult(
                action="skip",
                symbol=sym,
                signal=signal,
                message=f"Spread too wide ({rating.spread_pct or 0:.3f}%)",
                dry_run=cfg.dry_run,
            )

    try:
        if signal.side == "buy":
            order_id, msg = _place_buy(sym, cfg.trade_amount_usd, cfg.dry_run)
            action: Action = "buy"
        else:
            qty = _get_position_qty(sym)
            if qty <= 0:
                return TradeResult(action="skip", symbol=sym, signal=signal, message="Sell signal but no position", dry_run=cfg.dry_run)
            order_id, msg = _place_sell(sym, qty, cfg.dry_run)
            action = "sell"
    except Exception as exc:
        return TradeResult(action="skip", symbol=sym, signal=signal, message=f"Order failed: {exc}", dry_run=cfg.dry_run)

    st.executed_keys.append(key)
    st.daily_trade_count += 1
    log_entry = {
        "timestamp": datetime.now(ET).isoformat(),
        "symbol": sym,
        "action": action,
        "signal_reason": signal.reason,
        "signal_strength": signal.strength,
        "order_id": order_id,
        "dry_run": cfg.dry_run,
        "message": msg,
    }
    st.trade_log.append(log_entry)
    st.trade_log = st.trade_log[-100:]
    save_trade_state(st, state_path)

    return TradeResult(action=action, symbol=sym, signal=signal, message=msg, order_id=order_id, dry_run=cfg.dry_run)