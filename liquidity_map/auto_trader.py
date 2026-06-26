"""Paper trading driven by the unified model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from liquidity_map.data import fetch_bars, is_crypto
from liquidity_map.model import build_config, live_advice
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

ET = ZoneInfo("America/New_York")
STATE_FILE = Path(__file__).resolve().parent.parent / ".trade_state.json"

Action = Literal["buy", "sell", "hold", "skip"]


@dataclass
class TradeConfig:
    dry_run: bool = True
    trade_amount_usd: float = 100.0
    max_daily_trades: int = 5
    paper_starting_cash: float = 10_000.0


@dataclass
class TradeResult:
    action: Action
    symbol: str
    message: str
    dry_run: bool = True
    order_id: str | None = None
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
    peak_prices: dict[str, float] = field(default_factory=dict)


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
        max_daily_trades=int(os.getenv("AUTO_TRADE_MAX_DAILY", "5")),
        paper_starting_cash=float(os.getenv("PAPER_STARTING_CASH", "10000")),
    )


def load_trade_state(path: Path = STATE_FILE) -> TradeState:
    if not path.exists():
        return TradeState(paper_cash=load_trade_config().paper_starting_cash)
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
            peak_prices=dict(raw.get("peak_prices", {})),
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
    if state_qty > 0 and float(portfolio.cost_basis.get(sym, 0.0)) <= 0:
        log_qty, log_avg = infer_cost_basis_from_log(state.trade_log, sym)
        if log_avg > 0:
            portfolio.cost_basis[sym] = log_avg
        else:
            fallback = last_buy_price_from_log(state.trade_log, sym)
            if fallback > 0:
                portfolio.cost_basis[sym] = fallback
    return portfolio


def _reset_daily(state: TradeState) -> None:
    today = date.today().isoformat()
    if state.daily_trade_date != today:
        state.daily_trade_date = today
        state.daily_trade_count = 0


def _market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=30, second=0) <= now <= now.replace(hour=16, minute=0, second=0)


def evaluate_and_trade(
    symbol: str,
    df: pd.DataFrame,
    config: TradeConfig | None = None,
    state: TradeState | None = None,
    state_path: Path = STATE_FILE,
) -> TradeResult:
    cfg = config or load_trade_config()
    st = state if state is not None else load_trade_state(state_path)
    sym = symbol.strip().upper()
    price = last_price(df)
    _reset_daily(st)

    if not is_crypto(sym) and not _market_open():
        return TradeResult("skip", sym, "Market closed (9:30–16:00 ET)", cfg.dry_run)

    portfolio = get_paper_portfolio(st, sym)
    qty = get_position_qty(portfolio, sym)
    in_pos = qty > 0
    entry = float(portfolio.cost_basis.get(sym, price)) if in_pos else None
    peak = float(st.peak_prices.get(sym, entry or price))
    if in_pos:
        peak = max(peak, float(df["high"].iloc[-1]), price)
        st.peak_prices[sym] = peak

    try:
        trend_df = fetch_bars(sym, period="1y", interval="1d")
    except Exception:
        trend_df = df
    model_cfg = build_config(sym, "3mo", len(df), strictness=3)
    advice = live_advice(
        df,
        in_position=in_pos,
        entry_price=entry,
        peak_price=peak,
        cfg=model_cfg,
        ticker=sym,
        trend_df=trend_df,
    )

    if advice.action == "wait":
        return TradeResult("hold", sym, advice.reason, cfg.dry_run)

    key = f"{sym}|{advice.action}|{df['datetime'].iloc[-1]}|{advice.reason[:50]}"
    if key in st.executed_keys:
        return TradeResult("skip", sym, "Already acted on this signal", cfg.dry_run)

    if st.daily_trade_count >= cfg.max_daily_trades:
        return TradeResult("skip", sym, "Daily limit reached", cfg.dry_run)

    if cfg.dry_run:
        amt = f"${cfg.trade_amount_usd:.0f}" if advice.action == "buy" else "full position"
        return TradeResult("skip", sym, f"Would {advice.action.upper()} {sym} ({amt}) @ ${price:.2f} — {advice.reason}", True)

    try:
        if advice.action == "buy":
            if in_pos:
                return TradeResult("skip", sym, "Already holding", True)
            order_id, msg, portfolio = paper_buy(portfolio, sym, cfg.trade_amount_usd, price)
            st.peak_prices[sym] = price
            action: Action = "buy"
            amount = cfg.trade_amount_usd
        else:
            if qty <= 0:
                return TradeResult("skip", sym, "Nothing to sell", True)
            order_id, msg, portfolio = paper_sell(portfolio, sym, price)
            st.peak_prices.pop(sym, None)
            action = "sell"
            amount = qty * price
    except Exception as exc:
        return TradeResult("skip", sym, str(exc), True)

    st.paper_cash = portfolio.cash
    st.paper_positions = portfolio.positions
    st.paper_cost_basis = portfolio.cost_basis
    st.executed_keys.append(key)
    st.daily_trade_count += 1
    equity = portfolio_value(portfolio, {sym: price})
    st.trade_log.append(
        {
            "timestamp": datetime.now(ET).isoformat(),
            "symbol": sym,
            "action": action,
            "reason": advice.reason,
            "order_id": order_id,
            "price": price,
            "amount_usd": round(amount, 2),
            "paper_equity": round(equity, 2),
            "message": msg,
        }
    )
    st.trade_log = st.trade_log[-100:]
    save_trade_state(st, state_path)
    return TradeResult(action, sym, f"{msg} | equity ${equity:,.2f}", False, order_id)