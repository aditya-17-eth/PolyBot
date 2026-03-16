"""
PolyBot — Market Discovery
Find active BTC 15-minute Up/Down markets on Polymarket via the Gamma API.
"""

import time
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import requests

import config

logger = logging.getLogger("polybot.market_discovery")


@dataclass
class MarketInfo:
    """Represents a single BTC 15-min Up/Down market."""
    market_id: str = ""
    condition_id: str = ""
    slug: str = ""
    question: str = ""
    start_time: datetime = None
    end_time: datetime = None
    start_price: float = 0.0        # BTC reference price at market open
    token_id_up: str = ""            # Token ID for the UP outcome
    token_id_down: str = ""          # Token ID for the DOWN outcome
    outcome_prices: dict = field(default_factory=dict)  # {"up": 0.60, "down": 0.40}
    active: bool = False
    closed: bool = False

    @property
    def seconds_remaining(self) -> float:
        if not self.end_time:
            return float("inf")
        delta = self.end_time - datetime.now(timezone.utc)
        return max(delta.total_seconds(), 0)

    @property
    def seconds_elapsed(self) -> float:
        if not self.start_time:
            return 0
        delta = datetime.now(timezone.utc) - self.start_time
        return max(delta.total_seconds(), 0)

    @property
    def is_in_entry_window(self) -> bool:
        remaining = self.seconds_remaining
        return config.EXIT_WINDOW_SECS < remaining <= config.ENTRY_WINDOW_SECS

    @property
    def is_expired(self) -> bool:
        return self.seconds_remaining <= 0


class MarketDiscovery:
    """
    Discovers and tracks BTC 15-minute Up/Down markets from Polymarket.
    Uses the Gamma API to find active events matching the BTC Up/Down series.
    """

    SERIES_SLUG = "btc-up-or-down-15m"
    EVENT_SLUG_PREFIX = "btc-updown-15m-"
    WINDOW_SECS = 900  # 15 minutes

    def __init__(self):
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._current_market: MarketInfo = None
        self._last_fetch_time = 0
        self._fetch_interval = 30  # seconds between API calls

    @property
    def current_market(self) -> MarketInfo:
        return self._current_market

    def fetch_active_market(self) -> MarketInfo:
        """
        Find the currently active BTC 15m Up/Down market.
        Returns None if no suitable market is found.
        """
        now = time.time()
        # Don't spam the API
        if now - self._last_fetch_time < self._fetch_interval and self._current_market:
            if not self._current_market.is_expired:
                return self._current_market

        self._last_fetch_time = now

        # Strategy 1: Compute current/next window timestamps and query by slug
        market = self._fetch_by_slug()
        if market:
            self._current_market = market
            return market

        # Strategy 2: Text search for "Bitcoin Up or Down" on /events
        market = self._fetch_by_search()
        if market:
            self._current_market = market
            return market

        logger.warning("No active BTC 15m Up/Down market found")
        return None

    def get_market_orderbook(self, market: MarketInfo = None) -> dict:
        """
        Fetch order book data for the market's UP token from the CLOB API.
        Returns {"bids": [...], "asks": [...], "spread": float, "best_bid": float, "best_ask": float}
        """
        m = market or self._current_market
        if not m or not m.token_id_up:
            return {}

        try:
            # Fetch orderbook for UP token
            r = self._session.get(
                f"{config.POLYMARKET_HOST}/book",
                params={"token_id": m.token_id_up},
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0
            best_ask = float(asks[0]["price"]) if asks else 1
            spread = best_ask - best_bid

            # Calculate liquidity near the top
            bid_liquidity = sum(float(b.get("size", 0)) for b in bids[:5])
            ask_liquidity = sum(float(a.get("size", 0)) for a in asks[:5])

            return {
                "bids": bids,
                "asks": asks,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "midpoint": (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.5,
            }
        except Exception as e:
            logger.warning("Failed to fetch market orderbook: %s", e)
            return {}

    def get_market_prices(self, market: MarketInfo = None) -> dict:
        """Get current UP and DOWN prices from the CLOB midpoint."""
        m = market or self._current_market
        if not m:
            return {"up": 0.5, "down": 0.5}

        try:
            prices = {}
            for label, token_id in [("up", m.token_id_up), ("down", m.token_id_down)]:
                if not token_id:
                    continue
                r = self._session.get(
                    f"{config.POLYMARKET_HOST}/midpoint",
                    params={"token_id": token_id},
                    timeout=5,
                )
                r.raise_for_status()
                prices[label] = float(r.json().get("mid", 0.5))
            return prices
        except Exception as e:
            logger.warning("Failed to fetch market prices: %s", e)
            return {"up": 0.5, "down": 0.5}

    # ── Private Methods ────────────────────────────────────────────────

    def _compute_window_timestamps(self, count: int = 4) -> list:
        """
        Compute Unix timestamps for the current and upcoming 15-min windows.
        Each window starts on a 15-minute boundary (0, 15, 30, 45 past the hour).
        Returns a list of timestamps to try.
        """
        now = int(time.time())
        # Find the most recent 15-minute boundary
        current_window = (now // self.WINDOW_SECS) * self.WINDOW_SECS
        # Return current window, the one before it (might still be resolving),
        # and a few upcoming windows
        timestamps = []
        for i in range(-1, count):
            timestamps.append(current_window + i * self.WINDOW_SECS)
        return timestamps

    def _fetch_by_slug(self) -> MarketInfo:
        """
        Try to find the market by computing the event slug from the current time.
        Event slugs follow the pattern: btc-updown-15m-{unix_timestamp}
        where the timestamp is the start of the 15-minute window.
        """
        timestamps = self._compute_window_timestamps()
        best = None

        for ts in timestamps:
            slug = f"{self.EVENT_SLUG_PREFIX}{ts}"
            try:
                r = self._session.get(
                    f"{config.GAMMA_API_URL}/events",
                    params={
                        "slug": slug,
                        "limit": 1,
                    },
                    timeout=10,
                )
                r.raise_for_status()
                events = r.json()
                if not events:
                    continue

                if isinstance(events, dict):
                    events = [events]

                candidate = self._parse_best_event(events)
                if candidate and candidate.seconds_remaining > 0:
                    if best is None or candidate.seconds_remaining < best.seconds_remaining:
                        best = candidate
                        logger.info("Found market via slug: %s (%.0fs remaining)",
                                    slug, candidate.seconds_remaining)

            except Exception as e:
                logger.debug("Slug fetch for %s failed: %s", slug, e)
                continue

        return best

    def _fetch_by_search(self) -> MarketInfo:
        """
        Search for BTC Up/Down events using text query on the /events endpoint.
        Fallback when slug-based lookup fails.
        """
        search_queries = [
            "Bitcoin Up or Down 15",
            "BTC Up or Down",
        ]

        for query in search_queries:
            try:
                r = self._session.get(
                    f"{config.GAMMA_API_URL}/events",
                    params={
                        "_q": query,
                        "active": "true",
                        "closed": "false",
                        "limit": 20,
                    },
                    timeout=10,
                )
                r.raise_for_status()
                events = r.json()

                if isinstance(events, dict):
                    events = [events]

                # Filter for actual BTC 15m Up/Down events by slug pattern
                btc_events = []
                for ev in events:
                    ev_slug = ev.get("slug", "").lower()
                    ev_title = ev.get("title", "").lower()
                    # Match by slug pattern or title
                    if "btc-updown-15m" in ev_slug or "btc-up-or-down" in ev_slug:
                        btc_events.append(ev)
                    elif "bitcoin up or down" in ev_title and "15" in ev_title:
                        btc_events.append(ev)

                if btc_events:
                    result = self._parse_best_event(btc_events)
                    if result:
                        logger.info("Found market via search query '%s'", query)
                        return result

            except Exception as e:
                logger.warning("Search fetch failed for query '%s': %s", query, e)
                continue

        return None

    def _parse_best_event(self, events: list) -> MarketInfo:
        """From a list of events, find the best active market."""
        now = datetime.now(timezone.utc)
        best = None

        for event in events:
            markets = event.get("markets", [])
            if not markets and event.get("condition_id"):
                markets = [event]

            for m in markets:
                info = self._parse_market(m)
                if info and info.seconds_remaining > 0:
                    if best is None or info.seconds_remaining < best.seconds_remaining:
                        best = info
        return best

    def _parse_market(self, data: dict) -> MarketInfo:
        """
        Parse market JSON into a MarketInfo object.
        Handles the actual Gamma API response format where:
        - endDate is an ISO datetime string
        - eventStartTime is an ISO datetime string (the 15-min window start)
        - outcomes is a JSON-encoded string like '["Up", "Down"]'
        - clobTokenIds is a JSON-encoded string with token IDs
        - outcomePrices is a JSON-encoded string like '["0.5", "0.5"]'
        """
        try:
            info = MarketInfo()
            info.market_id = data.get("id", data.get("condition_id", ""))
            info.condition_id = data.get("conditionId") or data.get("condition_id") or ""
            info.slug = data.get("slug", data.get("market_slug", ""))
            info.question = data.get("question", "")
            info.active = data.get("active", False)
            info.closed = data.get("closed", False)

            # Parse end time — try multiple field names
            end_str = (
                data.get("endDate")
                or data.get("end_date_iso")
                or data.get("end_date")
                or ""
            )
            if end_str:
                info.end_time = datetime.fromisoformat(
                    end_str.replace("Z", "+00:00")
                )

            # Parse start time — eventStartTime is the actual window start
            start_str = (
                data.get("eventStartTime")
                or data.get("startDate")
                or data.get("start_date")
                or ""
            )
            if start_str:
                info.start_time = datetime.fromisoformat(
                    start_str.replace("Z", "+00:00")
                )
            elif info.end_time:
                # Fallback: start = end - 15 minutes
                info.start_time = info.end_time - timedelta(
                    seconds=config.MARKET_DURATION_SECS
                )

            # ── Extract token IDs ──────────────────────────────────
            # Try structured tokens array first
            tokens = data.get("tokens", [])
            if tokens and len(tokens) >= 2:
                for t in tokens:
                    outcome = t.get("outcome", "").lower()
                    token_id = t.get("token_id", "")
                    if "up" in outcome or "yes" in outcome:
                        info.token_id_up = token_id
                    elif "down" in outcome or "no" in outcome:
                        info.token_id_down = token_id
            else:
                # Parse clobTokenIds (JSON-encoded string in API response)
                clob_ids = data.get("clobTokenIds", data.get("clob_token_ids", ""))
                if isinstance(clob_ids, str) and clob_ids:
                    try:
                        clob_ids = json.loads(clob_ids)
                    except (json.JSONDecodeError, TypeError):
                        clob_ids = []

                # Parse outcomes to map token IDs correctly
                outcomes_raw = data.get("outcomes", "")
                if isinstance(outcomes_raw, str) and outcomes_raw:
                    try:
                        outcomes_list = json.loads(outcomes_raw)
                    except (json.JSONDecodeError, TypeError):
                        outcomes_list = []
                elif isinstance(outcomes_raw, list):
                    outcomes_list = outcomes_raw
                else:
                    outcomes_list = []

                if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                    if outcomes_list and len(outcomes_list) >= 2:
                        # Map by outcome label
                        for i, label in enumerate(outcomes_list):
                            label_lower = label.lower() if isinstance(label, str) else ""
                            if label_lower in ("up", "yes"):
                                info.token_id_up = clob_ids[i]
                            elif label_lower in ("down", "no"):
                                info.token_id_down = clob_ids[i]
                    else:
                        # Default: first = Up, second = Down
                        info.token_id_up = clob_ids[0]
                        info.token_id_down = clob_ids[1]

            # ── Outcome prices ─────────────────────────────────────
            prices_str = data.get("outcomePrices", data.get("outcome_prices", ""))
            if prices_str:
                if isinstance(prices_str, str):
                    try:
                        prices_list = json.loads(prices_str)
                    except (json.JSONDecodeError, TypeError):
                        prices_list = []
                else:
                    prices_list = prices_str

                if isinstance(prices_list, list) and len(prices_list) >= 2:
                    info.outcome_prices = {
                        "up": float(prices_list[0]),
                        "down": float(prices_list[1]),
                    }

            # Try to extract start price from description
            desc = data.get("description", "")
            info.start_price = self._extract_start_price(desc)

            return info

        except Exception as e:
            logger.warning("Failed to parse market: %s", e)
            return None

    @staticmethod
    def _extract_start_price(description: str) -> float:
        """
        Try to find the BTC reference/start price from the market description.
        Common patterns: "reference price of $64,500" or "starting at 64500"
        """
        import re
        patterns = [
            r'\$?([\d,]+\.?\d*)\s*(?:reference|start|opening)',
            r'(?:reference|start|opening)\s*(?:price)?\s*(?:of|at|:)?\s*\$?([\d,]+\.?\d*)',
            r'(?:BTC|Bitcoin)\s*(?:price)?\s*(?:at|of|:)?\s*\$?([\d,]+\.?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(",", "")
                try:
                    price = float(price_str)
                    if 10000 < price < 1000000:  # Sanity check for BTC price
                        return price
                except ValueError:
                    continue
        return 0.0


# ── Standalone Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    discovery = MarketDiscovery()

    print("\n🔍 Searching for active BTC 15m Up/Down markets...\n")
    market = discovery.fetch_active_market()

    if market:
        print(f"  ✅ Found market:")
        print(f"     Question: {market.question}")
        print(f"     Market ID: {market.market_id}")
        print(f"     End time: {market.end_time}")
        print(f"     Seconds remaining: {market.seconds_remaining:.0f}")
        print(f"     Token UP:   {market.token_id_up}")
        print(f"     Token DOWN: {market.token_id_down}")
        print(f"     Start price: ${market.start_price:,.2f}")
        print(f"     Outcome prices: {market.outcome_prices}")
        print(f"     In entry window: {market.is_in_entry_window}")

        print("\n📊 Fetching order book...")
        ob = discovery.get_market_orderbook(market)
        if ob:
            print(f"     Best bid: {ob.get('best_bid')}")
            print(f"     Best ask: {ob.get('best_ask')}")
            print(f"     Spread: {ob.get('spread', 0):.4f}")
            print(f"     Bid liquidity: {ob.get('bid_liquidity', 0):.2f}")
    else:
        print("  ❌ No active BTC 15m market found")
        print("     This may mean no market is currently running.")
        print("     Markets run every 15 minutes, try again shortly.")
