"""
PolyBot — Signal Computation
Computes all trading signals every tick from BTC price, orderbook, and market data.
"""

import logging
from dataclasses import dataclass, asdict

from btc_price import BTCPriceTracker
from orderbook_monitor import OrderbookMonitor
from market_discovery import MarketDiscovery, MarketInfo
import config

logger = logging.getLogger("polybot.signals")


@dataclass
class SignalSnapshot:
    """All signals at a single point in time."""
    # Market context
    market_id: str = ""
    time_remaining: float = 0.0

    # BTC price signals
    btc_price: float = 0.0
    start_price: float = 0.0
    distance: float = 0.0
    distance_abs: float = 0.0
    direction: str = ""             # "UP" or "DOWN"

    # Volatility signals
    volatility_15s: float = 0.0
    volatility_30s: float = 0.0
    volatility_60s: float = 0.0
    vol_acceleration: float = 0.0   # How fast volatility is changing

    # Momentum signals
    momentum_10s: float = 0.0
    momentum_30s: float = 0.0
    rate_of_change: float = 0.0     # $/sec
    consec_ticks: int = 0           # Consecutive ticks in one direction

    # Price range
    high_since_start: float = 0.0
    low_since_start: float = 0.0

    # Exchange orderbook
    ob_imbalance: float = 0.0       # -1 to +1
    buy_wall_size: float = 0.0
    sell_wall_size: float = 0.0
    liquidity_trap_score: float = 0.0

    # Prediction market
    market_prob_up: float = 0.5
    market_prob_down: float = 0.5
    market_spread: float = 0.0
    market_bid_liquidity: float = 0.0
    market_ask_liquidity: float = 0.0

    # Computed
    model_prob: float = 0.0         # Our estimated probability
    edge: float = 0.0               # model_prob - market_prob - fee
    sigma_per_sec: float = 0.0      # Estimated per-second volatility

    # Decision
    signal: str = "SKIP"            # "BUY_UP", "BUY_DOWN", or "SKIP"
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SignalComputer:
    """
    Aggregates data from all sources and computes a unified signal snapshot.
    Called every POLL_INTERVAL_SECS by the main bot loop.
    """

    def __init__(
        self,
        price_tracker: BTCPriceTracker,
        orderbook_monitor: OrderbookMonitor,
        market_discovery: MarketDiscovery,
    ):
        self.price = price_tracker
        self.orderbook = orderbook_monitor
        self.discovery = market_discovery

    def compute(self, market: MarketInfo) -> SignalSnapshot:
        """Compute all signals for the given market."""
        snap = SignalSnapshot()

        if not market:
            snap.skip_reason = "no_market"
            return snap

        snap.market_id = market.market_id
        snap.time_remaining = market.seconds_remaining

        # ── BTC Price Signals ──────────────────────────────────────
        snap.btc_price = self.price.current_price
        snap.start_price = market.start_price

        if snap.start_price > 0 and snap.btc_price > 0:
            snap.distance = snap.btc_price - snap.start_price
            snap.distance_abs = abs(snap.distance)
            snap.direction = "UP" if snap.distance > 0 else "DOWN" if snap.distance < 0 else ""
        else:
            snap.skip_reason = "no_price_data"
            return snap

        # ── Volatility ─────────────────────────────────────────────
        snap.volatility_15s = self.price.get_volatility(15)
        snap.volatility_30s = self.price.get_volatility(30)
        snap.volatility_60s = self.price.get_volatility(60)
        snap.vol_acceleration = self.price.get_volatility_acceleration()

        # ── Momentum ───────────────────────────────────────────────
        snap.momentum_10s = self.price.get_momentum(10)
        snap.momentum_30s = self.price.get_momentum(30)
        snap.rate_of_change = self.price.get_rate_of_change(10)
        snap.consec_ticks = self.price.get_consecutive_ticks()

        # ── Price Range ────────────────────────────────────────────
        if market.start_time:
            start_ts = market.start_time.timestamp()
            high, low = self.price.get_high_low_since(start_ts)
            snap.high_since_start = high
            snap.low_since_start = low

        # ── Exchange Orderbook ─────────────────────────────────────
        ob = self.orderbook.state
        snap.ob_imbalance = ob.imbalance
        snap.buy_wall_size = ob.bid_wall_size
        snap.sell_wall_size = ob.ask_wall_size
        snap.liquidity_trap_score = ob.liquidity_trap_score

        # ── Prediction Market ──────────────────────────────────────
        market_prices = self.discovery.get_market_prices(market)
        snap.market_prob_up = market_prices.get("up", 0.5)
        snap.market_prob_down = market_prices.get("down", 0.5)

        market_ob = self.discovery.get_market_orderbook(market)
        snap.market_spread = market_ob.get("spread", 1.0)
        snap.market_bid_liquidity = market_ob.get("bid_liquidity", 0)
        snap.market_ask_liquidity = market_ob.get("ask_liquidity", 0)

        # ── Model Probability ──────────────────────────────────────
        snap.sigma_per_sec = self.price.get_stdev_per_second(60)
        snap.model_prob = self._estimate_probability(
            snap.distance_abs, snap.time_remaining, snap.sigma_per_sec
        )

        # ── Edge ───────────────────────────────────────────────────
        if snap.direction == "UP":
            market_price = snap.market_prob_up
        elif snap.direction == "DOWN":
            market_price = snap.market_prob_down
        else:
            market_price = 0.5

        fee = config.MAX_FEE_PCT / 100.0
        snap.edge = (snap.model_prob - market_price - fee) * 100  # As percentage

        return snap

    @staticmethod
    def _estimate_probability(distance: float, time_remaining: float, sigma: float) -> float:
        """
        Estimate probability that BTC stays on the same side of the start price.

        Uses the normal CDF: P = Φ(d / (σ × √t))

        Where:
        - d = absolute distance from start price
        - σ = per-second price standard deviation
        - t = seconds remaining
        """
        if time_remaining <= 0:
            return 1.0 if distance > 0 else 0.5

        if sigma <= 0:
            # If we can't estimate volatility, use a conservative default
            # Assume ~$1/sec stdev for BTC (rough average)
            sigma = 1.0

        from scipy.stats import norm
        import math

        denominator = sigma * math.sqrt(time_remaining)
        if denominator <= 0:
            return 0.5

        z = distance / denominator
        prob = norm.cdf(z)

        # Clamp to reasonable range
        return max(0.01, min(0.99, prob))


# ── Standalone Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tracker = BTCPriceTracker()
    monitor = OrderbookMonitor()
    discovery = MarketDiscovery()
    computer = SignalComputer(tracker, monitor, discovery)

    tracker.start()
    monitor.start()

    print("\n⏳ Waiting for price data (5s)...\n")
    _time.sleep(5)

    market = discovery.fetch_active_market()
    if market:
        snap = computer.compute(market)
        print(f"  BTC: ${snap.btc_price:,.2f}")
        print(f"  Distance: ${snap.distance:+,.2f}")
        print(f"  Direction: {snap.direction}")
        print(f"  Volatility(60s): ${snap.volatility_60s:.2f}")
        print(f"  Model prob: {snap.model_prob:.4f}")
        print(f"  Market prob: UP={snap.market_prob_up:.3f} DOWN={snap.market_prob_down:.3f}")
        print(f"  Edge: {snap.edge:+.2f}%")
        print(f"  Time remaining: {snap.time_remaining:.0f}s")
    else:
        print("  No market found. Computing signals with mock data...")
        # Quick test of probability estimation
        from signals import SignalComputer
        for dist in [20, 50, 100, 200]:
            for t in [90, 60, 30]:
                p = SignalComputer._estimate_probability(dist, t, 1.5)
                print(f"    d=${dist}, t={t}s, σ=1.5 → P={p:.4f}")

    tracker.stop()
    monitor.stop()
