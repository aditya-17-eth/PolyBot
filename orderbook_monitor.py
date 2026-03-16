"""
PolyBot — Orderbook Monitor
Monitors exchange orderbooks (Binance) and prediction market orderbooks (Polymarket)
for buy/sell walls, imbalance, and liquidity trap detection.
"""

import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field

import requests

import config

logger = logging.getLogger("polybot.orderbook")


@dataclass
class WallSnapshot:
    """Snapshot of a large order (wall) at a specific time."""
    timestamp: float
    side: str       # "bid" or "ask"
    price: float
    size: float


@dataclass
class OrderbookState:
    """Processed orderbook state."""
    bid_wall_size: float = 0.0          # Largest bid in BTC
    ask_wall_size: float = 0.0          # Largest ask in BTC
    bid_wall_price: float = 0.0
    ask_wall_price: float = 0.0
    imbalance: float = 0.0             # +1 = all buying, -1 = all selling
    total_bid_depth: float = 0.0       # Total BTC on bids (top 20 levels)
    total_ask_depth: float = 0.0       # Total BTC on asks (top 20 levels)
    spread_pct: float = 0.0            # Spread as % of mid price
    liquidity_trap_score: float = 0.0  # 0 = safe, 1 = high trap risk


class OrderbookMonitor:
    """
    Monitors Binance BTC/USDT orderbook for wall detection and imbalance.
    Also tracks wall appearances/disappearances for liquidity trap detection.
    """

    def __init__(self):
        self._session = requests.Session()
        self._state = OrderbookState()
        self._lock = threading.Lock()
        self._wall_history: deque[WallSnapshot] = deque(maxlen=500)
        self._disappeared_walls: deque[WallSnapshot] = deque(maxlen=100)
        self._running = False
        self._poll_thread = None

    @property
    def state(self) -> OrderbookState:
        with self._lock:
            return self._state

    def start(self):
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("Orderbook monitor started")

    def stop(self):
        self._running = False
        logger.info("Orderbook monitor stopped")

    # ── Public API ─────────────────────────────────────────────────────

    def get_imbalance(self) -> float:
        """
        Order book imbalance ratio.
        Positive = more buying pressure, negative = more selling pressure.
        Range: -1.0 to +1.0
        """
        return self._state.imbalance

    def get_wall_info(self) -> dict:
        return {
            "bid_wall_size": self._state.bid_wall_size,
            "ask_wall_size": self._state.ask_wall_size,
            "bid_wall_price": self._state.bid_wall_price,
            "ask_wall_price": self._state.ask_wall_price,
        }

    def get_trap_score(self) -> float:
        """
        Liquidity trap score: 0 = safe, 1 = high risk of trap.
        Based on how frequently large walls appear and disappear.
        """
        return self._state.liquidity_trap_score

    # ── Polling ────────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            try:
                self._fetch_and_analyze()
            except Exception as e:
                logger.warning("Orderbook fetch error: %s", e)
            time.sleep(2)

    def _fetch_and_analyze(self):
        """Fetch Binance orderbook and compute metrics."""
        r = self._session.get(
            config.BINANCE_ORDERBOOK_URL,
            params={"symbol": "BTCUSDT", "limit": 20},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()

        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

        if not bids or not asks:
            return

        now = time.time()

        # Compute depth
        total_bids = sum(q for _, q in bids)
        total_asks = sum(q for _, q in asks)

        # Imbalance: (bids - asks) / (bids + asks)
        total = total_bids + total_asks
        imbalance = (total_bids - total_asks) / total if total > 0 else 0

        # Find largest walls
        bid_wall = max(bids, key=lambda x: x[1]) if bids else (0, 0)
        ask_wall = max(asks, key=lambda x: x[1]) if asks else (0, 0)

        # Spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0

        # Track walls for trap detection
        self._track_walls(now, bid_wall, ask_wall)

        # Compute trap score
        trap_score = self._compute_trap_score(now)

        with self._lock:
            self._state = OrderbookState(
                bid_wall_size=bid_wall[1],
                ask_wall_size=ask_wall[1],
                bid_wall_price=bid_wall[0],
                ask_wall_price=ask_wall[0],
                imbalance=imbalance,
                total_bid_depth=total_bids,
                total_ask_depth=total_asks,
                spread_pct=spread_pct,
                liquidity_trap_score=trap_score,
            )

    def _track_walls(self, now: float, bid_wall: tuple, ask_wall: tuple):
        """Track wall appearances and disappearances for trap detection."""
        threshold = config.WALL_SIZE_THRESHOLD_BTC

        # Record current walls
        if bid_wall[1] >= threshold:
            self._wall_history.append(WallSnapshot(now, "bid", bid_wall[0], bid_wall[1]))
        if ask_wall[1] >= threshold:
            self._wall_history.append(WallSnapshot(now, "ask", ask_wall[0], ask_wall[1]))

        # Check if previous walls disappeared
        window = config.WALL_DISAPPEAR_WINDOW
        recent_walls = [w for w in self._wall_history if now - w.timestamp < window]

        for wall in recent_walls:
            # Check if a wall from a few seconds ago is no longer present
            if now - wall.timestamp > 4:  # Give it at least 4 seconds
                still_exists = any(
                    w for w in recent_walls
                    if w.side == wall.side
                    and abs(w.price - wall.price) < 10
                    and w.timestamp > wall.timestamp
                )
                if not still_exists:
                    self._disappeared_walls.append(wall)

    def _compute_trap_score(self, now: float) -> float:
        """
        Compute liquidity trap score based on wall disappearance frequency.
        More disappearances in recent history = higher trap risk.
        """
        window = 60  # Look at last 60 seconds
        recent = [w for w in self._disappeared_walls if now - w.timestamp < window]
        # Normalize: 0 disappearances = 0 score, 5+ = 1.0
        score = min(len(recent) / 5.0, 1.0)
        return score


# ── Standalone Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    monitor = OrderbookMonitor()
    monitor.start()

    try:
        for _ in range(15):
            time.sleep(2)
            s = monitor.state
            print(
                f"  Imbalance: {s.imbalance:+.3f}  |  "
                f"Bid wall: {s.bid_wall_size:.2f} BTC @ ${s.bid_wall_price:,.0f}  |  "
                f"Ask wall: {s.ask_wall_size:.2f} BTC @ ${s.ask_wall_price:,.0f}  |  "
                f"Trap: {s.liquidity_trap_score:.2f}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
