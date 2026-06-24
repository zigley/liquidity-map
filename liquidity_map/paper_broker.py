"""Simulated paper trading — no broker login required."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


@dataclass
class PaperPortfolio:
    cash: float = 10_000.0
    positions: dict[str, float] = field(default_factory=dict)
    cost_basis: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    qty: float
    avg_price: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.market_price

    @property
    def cost_value(self) -> float:
        return self.qty * self.avg_price

    @property
    def pnl(self) -> float:
        return self.market_value - self.cost_value

    @property
    def pnl_pct(self) -> float:
        if self.cost_value <= 0:
            return 0.0
        return (self.pnl / self.cost_value) * 100


def get_position_qty(portfolio: PaperPortfolio, symbol: str) -> float:
    return float(portfolio.positions.get(symbol.upper(), 0.0))


def get_position_info(portfolio: PaperPortfolio, symbol: str, market_price: float) -> PositionInfo | None:
    sym = symbol.upper()
    qty = get_position_qty(portfolio, sym)
    if qty <= 0:
        return None
    avg = float(portfolio.cost_basis.get(sym, market_price))
    return PositionInfo(symbol=sym, qty=qty, avg_price=avg, market_price=market_price)


def last_price(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1])


def paper_buy(
    portfolio: PaperPortfolio,
    symbol: str,
    amount_usd: float,
    price: float,
) -> tuple[str, str, PaperPortfolio]:
    sym = symbol.upper()
    if amount_usd > portfolio.cash:
        raise RuntimeError(f"Insufficient paper cash (${portfolio.cash:.2f})")

    qty = amount_usd / price
    old_qty = portfolio.positions.get(sym, 0.0)
    old_avg = portfolio.cost_basis.get(sym, price)
    new_qty = old_qty + qty

    portfolio.cash -= amount_usd
    portfolio.positions[sym] = new_qty
    portfolio.cost_basis[sym] = ((old_qty * old_avg) + (qty * price)) / new_qty if new_qty > 0 else price

    order_id = f"paper-buy-{int(datetime.now(ET).timestamp())}"
    msg = f"PAPER buy {qty:.4f} {sym} @ ${price:.2f} (${amount_usd:.2f})"
    return order_id, msg, portfolio


def paper_sell(
    portfolio: PaperPortfolio,
    symbol: str,
    price: float,
    sell_all: bool = True,
) -> tuple[str, str, PaperPortfolio]:
    sym = symbol.upper()
    qty = get_position_qty(portfolio, sym)
    if qty <= 0:
        raise RuntimeError(f"No paper position in {sym}")

    proceeds = qty * price
    portfolio.cash += proceeds
    portfolio.positions.pop(sym, None)
    portfolio.cost_basis.pop(sym, None)
    order_id = f"paper-sell-{int(datetime.now(ET).timestamp())}"
    msg = f"PAPER sell {qty:.4f} {sym} @ ${price:.2f} (${proceeds:.2f})"
    return order_id, msg, portfolio


def portfolio_value(portfolio: PaperPortfolio, prices: dict[str, float]) -> float:
    total = portfolio.cash
    for sym, qty in portfolio.positions.items():
        total += qty * prices.get(sym, 0.0)
    return total


def _amount_usd_from_entry(entry: dict) -> float:
    amount = float(entry.get("amount_usd", 0) or 0)
    if amount > 0:
        return amount
    msg = entry.get("message", "")
    match = re.search(r"\(\$([\d,.]+)\)", msg)
    if match:
        return float(match.group(1).replace(",", ""))
    return 0.0


def infer_cost_basis_from_log(trade_log: list[dict], symbol: str) -> tuple[float, float]:
    """Rebuild qty and avg cost from trade log for legacy states."""
    sym = symbol.upper()
    qty = 0.0
    avg = 0.0
    for entry in trade_log:
        if entry.get("symbol", "").upper() != sym:
            continue
        price = float(entry.get("price", 0) or 0)
        if entry.get("action") == "buy":
            amount = _amount_usd_from_entry(entry)
            if amount <= 0 or price <= 0:
                continue
            buy_qty = amount / price
            new_qty = qty + buy_qty
            avg = ((qty * avg) + (buy_qty * price)) / new_qty if new_qty > 0 else price
            qty = new_qty
        elif entry.get("action") == "sell":
            qty = 0.0
            avg = 0.0
    return qty, avg


def last_buy_price_from_log(trade_log: list[dict], symbol: str) -> float:
    sym = symbol.upper()
    for entry in reversed(trade_log):
        if entry.get("symbol", "").upper() == sym and entry.get("action") == "buy":
            price = float(entry.get("price", 0) or 0)
            if price > 0:
                return price
    return 0.0