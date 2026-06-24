"""Simulated paper trading — no broker login required."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


@dataclass
class PaperPortfolio:
    cash: float = 10_000.0
    positions: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = {}


def get_position_qty(portfolio: PaperPortfolio, symbol: str) -> float:
    return float(portfolio.positions.get(symbol.upper(), 0.0))


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
    portfolio.cash -= amount_usd
    portfolio.positions[sym] = portfolio.positions.get(sym, 0.0) + qty
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
    order_id = f"paper-sell-{int(datetime.now(ET).timestamp())}"
    msg = f"PAPER sell {qty:.4f} {sym} @ ${price:.2f} (${proceeds:.2f})"
    return order_id, msg, portfolio


def portfolio_value(portfolio: PaperPortfolio, prices: dict[str, float]) -> float:
    total = portfolio.cash
    for sym, qty in portfolio.positions.items():
        total += qty * prices.get(sym, 0.0)
    return total