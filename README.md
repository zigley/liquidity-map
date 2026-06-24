# Liquidity Map

Streamlit app that overlays liquidity on stock charts: volume profile, time×price heatmap, POC/value area levels, buy/sell signals, and paper auto-trade.

## Features

- Volume profile (horizontal histogram)
- Liquidity heatmap behind candlesticks
- POC, VAH, VAL levels
- Buy/sell signals at HVN support, VAH/POC resistance, POC reclaim/loss, LVN→HVN moves
- Paper auto-trade from liquidity signals (no broker login)

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Paper auto-trade

Trades the **latest confirmed-bar** liquidity signal using a simulated portfolio ($10k starting cash by default).

### In the Streamlit app

1. Enable **Paper auto-trade** in the sidebar
2. Click **Run trade check now** or enable **Auto-poll**

### Background daemon

```bash
python auto_trade.py --ticker SPY --poll-seconds 60
```

Optional `.env` settings:

```
PAPER_STARTING_CASH=10000
AUTO_TRADE_AMOUNT_USD=100
AUTO_TRADE_MIN_STRENGTH=2
AUTO_TRADE_MAX_DAILY=5
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → main file: `app.py`

## Project structure

```
app.py                  # Streamlit UI
liquidity_map/
  data.py               # yfinance data
  profile.py            # Volume profile engine
  heatmap.py            # Liquidity heatmap
  signals.py            # Buy/sell signal detection
  chart.py              # Plotly chart builder
  paper_broker.py       # Simulated portfolio
  auto_trader.py        # Signal execution + risk limits
auto_trade.py           # Background paper-trade daemon
```