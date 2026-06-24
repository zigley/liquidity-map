# Liquidity Map

Streamlit app that overlays liquidity on stock charts: volume profile, time×price heatmap, POC/value area levels, and buy/sell signals where price aligns with liquid zones.

## Features

- Volume profile (horizontal histogram)
- Liquidity heatmap behind candlesticks
- POC, VAH, VAL levels
- Buy/sell signals at HVN support, VAH/POC resistance, POC reclaim/loss, LVN→HVN moves
- Optional Robinhood bid/ask spread (live liquidity read)
- Auto-trade liquidity signals via Robinhood (dry-run by default)

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or:

```bash
pip install -e .
python -m streamlit run app.py
```

Open http://localhost:8501

## Optional Robinhood spread

Copy `.env.example` to `.env` and set credentials, or enter them in the sidebar.

```
RH_USERNAME=your_email@example.com
RH_PASSWORD=your_password
```

## Auto-trade

Trades the **latest confirmed-bar** liquidity signal (buy at support, sell at resistance).

**Dry-run is ON by default** — no real orders until you disable it.

### In the Streamlit app

1. Log in to Robinhood in the sidebar
2. Enable **Auto-trade**
3. Use **Run trade check now** or enable **Auto-poll**

### Background daemon (recommended for 24/7)

```bash
# Dry-run (safe default)
python auto_trade.py --ticker SPY --poll-seconds 60

# LIVE orders (real money)
python auto_trade.py --ticker SPY --live
```

Configure via `.env`:

```
AUTO_TRADE_DRY_RUN=true
AUTO_TRADE_AMOUNT_USD=100
AUTO_TRADE_MIN_STRENGTH=2
AUTO_TRADE_MAX_DAILY=5
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → main file: `app.py`
4. Add secrets for Robinhood (optional): `RH_USERNAME`, `RH_PASSWORD`

## Project structure

```
app.py                  # Streamlit UI
liquidity_map/
  data.py               # yfinance + Robinhood quotes
  profile.py            # Volume profile engine
  heatmap.py            # Liquidity heatmap
  signals.py            # Buy/sell signal detection
  chart.py              # Plotly chart builder
  liquidity_score.py    # Spread scoring
  auto_trader.py        # Signal execution + risk limits
auto_trade.py           # Background auto-trade daemon
```