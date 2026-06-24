"""Market data loaders via yfinance."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yfinance as yf

Interval = Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"]
Period = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

INTERVAL_BY_PERIOD: dict[str, str] = {
    "1d": "5m",
    "5d": "15m",
    "1mo": "1h",
    "3mo": "1d",
    "6mo": "1d",
    "1y": "1d",
    "2y": "1wk",
    "5y": "1wk",
    "max": "1mo",
}


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    spread_pct: float | None


def auto_interval(period: str) -> str:
    return INTERVAL_BY_PERIOD.get(period, "1d")


def fetch_bars(
    ticker: str,
    period: str = "3mo",
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars via yfinance."""
    symbol = ticker.strip().upper()
    chosen_interval = interval or auto_interval(period)

    if start:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval=chosen_interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    else:
        raw = yf.download(
            symbol,
            period=period,
            interval=chosen_interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    if raw.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [col.lower() for col in raw.columns]

    df = raw.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    ).reset_index()

    datetime_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={datetime_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)

    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def fetch_quote(ticker: str) -> Quote:
    """Best-effort bid/ask from yfinance (may be absent for some symbols)."""
    symbol = ticker.strip().upper()
    info = yf.Ticker(symbol).fast_info
    bid = getattr(info, "last_price", None)
    last = bid
    spread_pct = None
    return Quote(symbol=symbol, bid=bid, ask=None, last=last, spread_pct=spread_pct)


def load_env_credentials() -> tuple[str | None, str | None]:
    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("RH_USERNAME"), os.getenv("RH_PASSWORD")