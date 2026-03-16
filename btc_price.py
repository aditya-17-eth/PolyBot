"""
PolyBot — BTC Price Tracker
Real-time BTC/USD price with rolling buffer for volatility, momentum, and range.
Uses Binance WebSocket (primary) + REST fallback.
"""

import time
import threading
import json
import logging
from collections import deque

import requests
import websocket

import config

logger = logging.getLogger("polybot.btc_price")


class PriceTick:
    """Single price observation."""
    __slots__ = ("timestamp", "price")

    def __init__(self, price: float, timestamp: float = None):
        self.price = price
        self.timestamp = timestamp or time.time()


class BTCPriceTracker:
    """
    Maintains a rolling buffer of BTC/USD price ticks.
    Primary: Binance WebSocket stream.
    Fallback: Binance REST or CoinGecko.
    """

    def __init__(self):
        self._buffer: deque[PriceTick] = deque(maxlen=5000)
        self._lock = threading.Lock()
        self._current_price: float = 0.0
        self._ws: websocket.WebSocketApp = None
        self._ws_thread: threading.Thread = None
        self._running = False
        self._connected = False

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def current_price(self) -> float:
        return self._current_price

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_prices_since(self, seconds: float) -> list[PriceTick]:
        """Return all ticks within the last N seconds."""
        cutoff = time.time() - seconds
        with self._lock:
            return [t for t in self._buffer if t.timestamp >= cutoff]

    def get_volatility(self, seconds: float = 60) -> float:
        """Price range (high - low) in the last N seconds."""
        ticks = self.get_prices_since(seconds)
        if len(ticks) < 2:
            return 0.0
        prices = [t.price for t in ticks]
        return max(prices) - min(prices)

    def get_momentum(self, seconds: float = 30) -> float:
        """
        Price change over the last N seconds.
        Positive = upward, negative = downward.
        """
        ticks = self.get_prices_since(seconds)
        if len(ticks) < 2:
            return 0.0
        return ticks[-1].price - ticks[0].price

    def get_rate_of_change(self, seconds: float = 10) -> float:
        """Average price change per second over the last N seconds."""
        ticks = self.get_prices_since(seconds)
        if len(ticks) < 2:
            return 0.0
        dt = ticks[-1].timestamp - ticks[0].timestamp
        if dt <= 0:
            return 0.0
        return (ticks[-1].price - ticks[0].price) / dt

    def get_high_low_since(self, since_ts: float) -> tuple[float, float]:
        """Return (high, low) since a given timestamp."""
        with self._lock:
            prices = [t.price for t in self._buffer if t.timestamp >= since_ts]
        if not prices:
            return (self._current_price, self._current_price)
        return (max(prices), min(prices))

    def get_consecutive_ticks(self) -> int:
        """
        Count consecutive price ticks in one direction.
        Positive = consecutive up, negative = consecutive down.
        """
        with self._lock:
            ticks = list(self._buffer)
        if len(ticks) < 2:
            return 0

        direction = 0
        count = 0
        for i in range(len(ticks) - 1, 0, -1):
            diff = ticks[i].price - ticks[i - 1].price
            if diff == 0:
                continue
            current_dir = 1 if diff > 0 else -1
            if direction == 0:
                direction = current_dir
                count = 1
            elif current_dir == direction:
                count += 1
            else:
                break

        return count * direction

    def get_volatility_acceleration(self) -> float:
        """Compare recent volatility (last 15s) to slightly older (15-30s ago)."""
        recent = self.get_volatility(15)
        older = self._get_volatility_window(30, 15)
        if older <= 0:
            return 0.0
        return (recent - older) / older

    def get_stdev_per_second(self, seconds: float = 60) -> float:
        """Estimate per-second price standard deviation from recent returns."""
        ticks = self.get_prices_since(seconds)
        if len(ticks) < 10:
            # Fallback: estimate from range
            vol = self.get_volatility(seconds)
            return vol / max(seconds, 1) ** 0.5

        # Compute per-second returns
        returns = []
        for i in range(1, len(ticks)):
            dt = ticks[i].timestamp - ticks[i - 1].timestamp
            if dt > 0:
                ret = (ticks[i].price - ticks[i - 1].price) / dt
                returns.append(ret)

        if not returns:
            return 0.0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Start the price tracker (WebSocket + fallback)."""
        self._running = True
        # Fetch initial price via REST
        self._fetch_rest_price()
        # Start WebSocket in background
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        # Start fallback poller
        threading.Thread(target=self._fallback_poller, daemon=True).start()
        logger.info("BTC price tracker started | initial price: $%.2f", self._current_price)

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("BTC price tracker stopped")

    # ── WebSocket ──────────────────────────────────────────────────────

    def _run_ws(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    config.BINANCE_WS_URL,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                    on_open=self._on_ws_open,
                )
                self._ws.run_forever(ping_interval=30)
            except Exception as e:
                logger.error("WebSocket error: %s", e)
            if self._running:
                self._connected = False
                logger.info("Reconnecting WebSocket in 5s...")
                time.sleep(5)

    def _on_ws_open(self, ws):
        self._connected = True
        logger.info("Binance WebSocket connected")

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            price = float(data["p"])
            ts = data.get("T", time.time() * 1000) / 1000
            self._add_tick(price, ts)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    def _on_ws_error(self, ws, error):
        logger.warning("WebSocket error: %s", error)
        self._connected = False

    def _on_ws_close(self, ws, close_status, close_msg):
        self._connected = False
        logger.info("WebSocket closed: %s %s", close_status, close_msg)

    # ── REST Fallback ──────────────────────────────────────────────────

    def _fetch_rest_price(self):
        try:
            r = requests.get(
                f"{config.BINANCE_REST_URL}/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=5,
            )
            r.raise_for_status()
            price = float(r.json()["price"])
            self._add_tick(price)
        except Exception as e:
            logger.warning("Binance REST failed, trying CoinGecko: %s", e)
            self._fetch_coingecko()

    def _fetch_coingecko(self):
        try:
            r = requests.get(
                config.COINGECKO_URL,
                params={"ids": "bitcoin", "vs_currencies": "usd"},
                timeout=5,
            )
            r.raise_for_status()
            price = float(r.json()["bitcoin"]["usd"])
            self._add_tick(price)
        except Exception as e:
            logger.error("CoinGecko also failed: %s", e)

    def _fallback_poller(self):
        """Poll REST every 5s if WebSocket is down."""
        while self._running:
            time.sleep(5)
            if not self._connected:
                self._fetch_rest_price()

    # ── Internals ──────────────────────────────────────────────────────

    def _add_tick(self, price: float, ts: float = None):
        tick = PriceTick(price, ts)
        self._current_price = price
        with self._lock:
            self._buffer.append(tick)

    def _get_volatility_window(self, start_secs_ago: float, end_secs_ago: float) -> float:
        """Volatility in a specific past window."""
        now = time.time()
        with self._lock:
            prices = [
                t.price for t in self._buffer
                if (now - start_secs_ago) <= t.timestamp <= (now - end_secs_ago)
            ]
        if len(prices) < 2:
            return 0.0
        return max(prices) - min(prices)


# ── Standalone Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tracker = BTCPriceTracker()
    tracker.start()

    try:
        for _ in range(30):
            time.sleep(1)
            p = tracker.current_price
            v = tracker.get_volatility(30)
            m = tracker.get_momentum(10)
            print(f"  BTC: ${p:,.2f}  |  Vol(30s): ${v:.2f}  |  Mom(10s): ${m:+.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
