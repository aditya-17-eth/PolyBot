"""
PolyBot — Configuration
All tunable parameters in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Polymarket Credentials ─────────────────────────────────────────────
POLYMARKET_HOST = "https://clob.polymarket.com"
POLYMARKET_CHAIN_ID = 137  # Polygon
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
POLYMARKET_SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

GAMMA_API_URL = "https://gamma-api.polymarket.com"

# ── Bot Mode ───────────────────────────────────────────────────────────
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# ── Timing ─────────────────────────────────────────────────────────────
ENTRY_WINDOW_SECS = 90          # Only trade in final N seconds of market
EXIT_WINDOW_SECS = 10           # Stop entering in the last N seconds (too risky)
POLL_INTERVAL_SECS = 1          # Signal computation interval
MARKET_DURATION_SECS = 900      # 15 minutes

# ── Strategy Selection ─────────────────────────────────────────────────
ENABLE_STRATEGY_STANDARD = True
ENABLE_STRATEGY_CROSSING = True
ENABLE_STRATEGY_BREAKOUT = True

# ── Strategy Thresholds ────────────────────────────────────────────────
MIN_DISTANCE_USD = 40.0         # Min BTC distance from start price to consider
MIN_EDGE_PCT = 3.0              # Min probability edge above fee (%)
MAX_FEE_PCT = 1.56              # Maximum expected taker fee (%)
MIN_VOLATILITY_USD = 15.0       # Min 60s BTC price range
MAX_SPREAD = 0.05               # Max bid-ask spread on prediction market
MIN_ORDERBOOK_LIQUIDITY = 50.0  # Min USD liquidity near best price

# ── Position Management ────────────────────────────────────────────────
BET_SIZE_PCT = float(os.getenv("BET_SIZE_PCT", "2.0"))   # % of bankroll per trade
MIN_BET_USD = 1.0               # Minimum trade size
MAX_BET_USD = 100.0             # Maximum trade size

# ── Exit Rules ─────────────────────────────────────────────────────────
PROFIT_TAKE_THRESHOLD = 0.92    # Sell early if position price reaches this
LOSS_CUT_EDGE_MIN = 0.5         # Exit if edge drops below this %
LOSS_CUT_PRICE_DROP = 0.15      # Exit if position drops this much from entry

# ── Risk Controls ──────────────────────────────────────────────────────
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "50.0"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "20"))
COOLDOWN_AFTER_LOSS_SECS = 120  # Pause after a loss
MAX_CONSECUTIVE_LOSSES = 3      # Longer cooldown after N consecutive losses
EXTENDED_COOLDOWN_SECS = 600    # Cooldown after consecutive losses

# ── Data Sources ───────────────────────────────────────────────────────
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_REST_URL = "https://api.binance.com/api/v3"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
BINANCE_ORDERBOOK_URL = "https://api.binance.com/api/v3/depth"

# ── Price Buffer ───────────────────────────────────────────────────────
PRICE_BUFFER_SECS = 120         # Keep last N seconds of price data
WALL_SIZE_THRESHOLD_BTC = 5.0   # Min BTC to count as a "wall"
WALL_DISAPPEAR_WINDOW = 10      # Seconds to track wall disappearance

# ── Logging ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "polybot.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Dashboard ──────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5050"))
DASHBOARD_HOST = "127.0.0.1"

# ── Backtesting ────────────────────────────────────────────────────────
BACKTEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "backtest_data")
