# PolyBot — BTC 15-Minute Up/Down Trading Bot

A Python bot that trades Polymarket's BTC 15-minute Up/Down prediction markets. It enters trades only in the final 60–90 seconds of each market when the outcome is nearly decided but market prices may still lag behind reality.

## Strategy

The bot **does not predict** where BTC will go. Instead, it waits until BTC has already moved significantly from the market's starting price, and enters only when the probability of the current direction holding is much higher than what the market price implies.

**Key formula**: `P = Φ(distance / (σ × √time_remaining))` — the probability that BTC stays on the same side of the start price.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in credentials
copy .env.example .env
# Edit .env with your Polymarket wallet details

# 3. Run in dry-run mode (default — no real money)
python bot.py

# 4. Open dashboard (separate terminal)
python dashboard.py
# Visit http://127.0.0.1:5050
```

## Architecture

| Module | Purpose |
|--------|---------|
| `config.py` | All tunable parameters |
| `btc_price.py` | Real-time BTC price (Binance WebSocket) |
| `market_discovery.py` | Find active markets (Polymarket Gamma API) |
| `orderbook_monitor.py` | Exchange orderbook analysis & trap detection |
| `signals.py` | Compute all trading signals every second |
| `strategy.py` | Entry/exit decision engine |
| `executor.py` | Order placement (Polymarket CLOB API) |
| `risk.py` | Position sizing, loss limits, cooldowns |
| `data_store.py` | SQLite trade/signal logging |
| `backtester.py` | Historical strategy simulation |
| `bot.py` | Main orchestration loop |
| `dashboard.py` | Flask web monitoring UI |

## Entry Conditions (ALL required)

- ≤ 90 seconds remaining in the market
- BTC distance from start price ≥ $40
- Probability edge ≥ 3% above fee
- 60-second volatility ≥ $15
- Market spread ≤ 5%
- No liquidity trap detected

## Risk Controls

- **Position sizing**: 2% of bankroll per trade
- **Daily loss limit**: $50 (configurable)
- **Cooldown**: 2 minutes after a loss, 10 minutes after 3 consecutive losses
- **Loss cutting**: Exit if edge drops or position drops 15% from entry
- **Profit taking**: Exit early if position price reaches 0.92+
- **DRY_RUN mode**: Default ON — no real trades until you enable it

## Backtesting

```bash
# Run backtest with synthetic data
python backtester.py

# Use your own CSV data (timestamp, price — one row per second)
# Place files in backtest_data/ directory
```

## Testing

```bash
python -m pytest tests/ -v
```

## Configuration

All parameters are in `config.py`. Key settings:

| Setting | Default | Env Var |
|---------|---------|---------|
| Dry run | `true` | `DRY_RUN` |
| Bet size | 2% | `BET_SIZE_PCT` |
| Daily loss limit | $50 | `MAX_DAILY_LOSS_USD` |
| Max daily trades | 20 | `MAX_DAILY_TRADES` |
| Dashboard port | 5050 | `DASHBOARD_PORT` |

## ⚠️ Disclaimer

This bot trades real money on Polymarket when `DRY_RUN=false`. Use at your own risk. Always start with dry-run mode and small position sizes. Past performance (including backtests) does not guarantee future results.
