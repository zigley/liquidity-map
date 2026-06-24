#!/usr/bin/env python3
"""Background daemon: auto-trade liquidity signals (paper trading, no login)."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from liquidity_map.auto_trader import evaluate_and_trade, load_trade_config, load_trade_state
from liquidity_map.data import auto_interval, fetch_bars

ET = ZoneInfo("America/New_York")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-trade liquidity signals (paper)")
    parser.add_argument("--ticker", default="SPY", help="Symbol to trade")
    parser.add_argument("--period", default="1mo", help="yfinance period for signal detection")
    parser.add_argument("--interval", default=None, help="Bar interval (auto if omitted)")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Seconds between checks")
    args = parser.parse_args()

    cfg = load_trade_config()
    interval = args.interval or auto_interval(args.period)
    state = load_trade_state()

    print(f"Paper auto-trader started | {args.ticker} | poll every {args.poll_seconds}s")
    print(f"Paper cash: ${state.paper_cash:,.2f}")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            df = fetch_bars(args.ticker, period=args.period, interval=interval)
            result = evaluate_and_trade(args.ticker, df, config=cfg)
            ts = datetime.now(ET).strftime("%H:%M:%S")
            sig = f" | {result.signal.reason}" if result.signal else ""
            print(f"[{ts}] {result.action.upper():4} {result.symbol} — {result.message}{sig}")
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            ts = datetime.now(ET).strftime("%H:%M:%S")
            print(f"[{ts}] ERROR — {exc}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()