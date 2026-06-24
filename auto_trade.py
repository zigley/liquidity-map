#!/usr/bin/env python3
"""Background daemon: auto-trade liquidity signals via Robinhood."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from liquidity_map.auto_trader import evaluate_and_trade, load_trade_config
from liquidity_map.data import auto_interval, fetch_bars, load_env_credentials, login_robinhood

ET = ZoneInfo("America/New_York")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-trade liquidity signals")
    parser.add_argument("--ticker", default="SPY", help="Symbol to trade")
    parser.add_argument("--period", default="1mo", help="yfinance period for signal detection")
    parser.add_argument("--interval", default=None, help="Bar interval (auto if omitted)")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--live", action="store_true", help="Disable dry-run (REAL orders)")
    args = parser.parse_args()

    cfg = load_trade_config()
    if args.live:
        cfg.dry_run = False

    user, pw = load_env_credentials()
    if not user or not pw:
        raise SystemExit("Set RH_USERNAME and RH_PASSWORD in .env")

    print(f"Logging in to Robinhood as {user}...")
    if not login_robinhood(user, pw):
        raise SystemExit("Robinhood login failed")

    interval = args.interval or auto_interval(args.period)
    mode = "DRY RUN" if cfg.dry_run else "LIVE"
    print(f"Auto-trader started [{mode}] | {args.ticker} | poll every {args.poll_seconds}s")
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