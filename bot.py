"""
PolyBot — Main Bot Loop
Orchestrates market discovery, signal computation, strategy, and execution.
"""

import time
import logging
import sys
from datetime import datetime, timezone

import config
from btc_price import BTCPriceTracker
from orderbook_monitor import OrderbookMonitor
from market_discovery import MarketDiscovery, MarketInfo
from signals import SignalComputer, SignalSnapshot
from strategy import Strategy, Action, Position
from executor import Executor
from risk import RiskManager
from data_store import DataStore
from dashboard import run_dashboard

# ── Logging Setup ──────────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s │ %(levelname)-5s │ %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=fmt, datefmt=datefmt)
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)

logger = logging.getLogger("polybot.bot")


class PolyBot:
    """
    Main bot that orchestrates the full trading loop:
    1. Discover active BTC 15m market
    2. Wait until entry window
    3. Compute signals every second
    4. Evaluate entry/exit decisions
    5. Execute trades
    6. Log everything
    7. Repeat
    """

    def __init__(self):
        # Components
        self.store = DataStore()
        self.price_tracker = BTCPriceTracker()
        self.orderbook_monitor = OrderbookMonitor()
        self.market_discovery = MarketDiscovery()
        self.signal_computer = SignalComputer(
            self.price_tracker, self.orderbook_monitor, self.market_discovery
        )
        self.strategy = Strategy()
        self.executor = Executor()
        self.risk = RiskManager(self.store, self.executor.get_balance())

        # State
        self._running = False
        self._current_market: MarketInfo = None
        self._last_signal: SignalSnapshot = None
        self._market_trade_count = 0
        self._status = "INITIALIZING"

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_signal(self) -> SignalSnapshot:
        return self._last_signal

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Start the bot."""
        self._running = True

        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║               🤖 PolyBot — BTC 15m Trader                   ║
║                                                              ║
║  Mode:     {'DRY RUN (paper trading)' if config.DRY_RUN else '🔴 LIVE TRADING'}{'                ' if config.DRY_RUN else '                       '}║
║  Bankroll: ${self.risk.bankroll:>10,.2f}                                  ║
║  Bet size: {config.BET_SIZE_PCT}% of bankroll                                ║
║  Max loss: ${config.MAX_DAILY_LOSS_USD:>10,.2f} / day                         ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(banner)

        # Start data feeds
        logger.info("Starting data feeds...")
        self.price_tracker.start()
        self.orderbook_monitor.start()

        # Wait for initial price data
        logger.info("Waiting for price data...")
        for _ in range(10):
            if self.price_tracker.current_price > 0:
                break
            time.sleep(1)

        if self.price_tracker.current_price <= 0:
            logger.error("Could not get BTC price — check network connection")
            self.stop()
            return

        logger.info(f"BTC price: ${self.price_tracker.current_price:,.2f} — ready!")
        self._status = "RUNNING"

        # Start web dashboard
        run_dashboard(bot=self, store=self.store)

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the bot gracefully."""
        self._running = False
        self._status = "STOPPED"
        self.price_tracker.stop()
        self.orderbook_monitor.stop()
        logger.info("Bot stopped")

    # ── Main Loop ──────────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error("Tick error: %s", e, exc_info=True)
            time.sleep(config.POLL_INTERVAL_SECS)

    def _tick(self):
        """Single iteration of the main loop."""

        # ── Step 1: Find or update the active market ───────────────
        market = self.market_discovery.fetch_active_market()

        # Compare by market_id, not object identity — fetch returns new
        # MarketInfo objects each call even for the same underlying market.
        current_id = self._current_market.market_id if self._current_market else None
        new_id = market.market_id if market else None

        if new_id != current_id:
            if market:
                self._on_new_market(market)
            else:
                self._status = "WAITING_FOR_MARKET"
                return
        elif market:
            # Same market, just update the reference (refreshed data)
            self._current_market = market

        if not market:
            return

        # Check if market has expired
        if market.is_expired:
            self._on_market_expired(market)
            return

        # ── Step 2: Compute signals (always, so dashboard has data) ─
        snap = self.signal_computer.compute(market)
        self._last_signal = snap

        # Log signal periodically (every 5 ticks to avoid spam)
        if int(time.time()) % 5 == 0:
            self.store.log_signal(snap.to_dict())

        # ── Step 3: Check if we're in the entry window ─────────────
        if not market.is_in_entry_window and not self.strategy.has_position():
            remaining = market.seconds_remaining
            if remaining > config.ENTRY_WINDOW_SECS:
                self._status = f"WAITING ({remaining:.0f}s until entry window)"
            else:
                self._status = f"MARKET_ENDING ({remaining:.0f}s left)"
            return

        # ── Step 4: Evaluate exit if we have a position ────────────
        if self.strategy.has_position():
            self._evaluate_exit(snap, market)
            return

        # ── Step 5: Check risk limits ──────────────────────────────
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            self._status = f"RISK_BLOCKED: {reason}"
            return

        # ── Step 6: Evaluate entry ─────────────────────────────────
        self._evaluate_entry(snap, market)

    # ── Market Lifecycle ───────────────────────────────────────────────

    def _on_new_market(self, market: MarketInfo):
        """Handle a new market appearing."""
        self._current_market = market
        self._market_trade_count = 0
        self.strategy.clear_position()

        logger.info("═" * 60)
        logger.info("📊 NEW MARKET: %s", market.question or market.slug)
        logger.info("   End: %s | Remaining: %.0fs", market.end_time, market.seconds_remaining)
        logger.info(f"   Start price: ${market.start_price:,.2f}")
        logger.info("   Token UP:   %s", market.token_id_up[:20] if market.token_id_up else "?")
        logger.info("   Token DOWN: %s", market.token_id_down[:20] if market.token_id_down else "?")
        logger.info("═" * 60)

        self.store.log_market_session({
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "start_time": market.start_time.isoformat() if market.start_time else "",
            "end_time": market.end_time.isoformat() if market.end_time else "",
            "start_price": market.start_price,
            "token_id_up": market.token_id_up,
            "token_id_down": market.token_id_down,
            "outcome": "",
            "trades_taken": 0,
        })

    def _on_market_expired(self, market: MarketInfo):
        """Handle market expiration."""
        if self.strategy.has_position():
            # Position will settle at expiry — log it
            pos = self.strategy.current_position
            logger.info("⏰ Market expired with open position: %s", pos.side)
            # Estimate P&L: if price settled in our direction, we win
            settle_price = 1.0 if (
                (pos.side == "UP" and self.price_tracker.current_price > market.start_price) or
                (pos.side == "DOWN" and self.price_tracker.current_price < market.start_price)
            ) else 0.0

            pnl = (settle_price - pos.entry_price) * pos.size_usd / pos.entry_price
            self._record_trade(pos, settle_price, pnl, "SETTLED_EXPIRY")
            self.strategy.clear_position()

        self._current_market = None
        self._status = "WAITING_FOR_MARKET"
        logger.info("Market expired — waiting for next market")

    # ── Entry / Exit ───────────────────────────────────────────────────

    def _evaluate_entry(self, snap: SignalSnapshot, market: MarketInfo):
        """Decide whether to enter a trade."""
        decision = self.strategy.evaluate_entry(snap)

        if decision.action == Action.SKIP:
            self._status = f"SCANNING (skip: {decision.reason})"
            if snap.time_remaining <= config.ENTRY_WINDOW_SECS:
                # Only log skips during entry window
                logger.debug(
                    "SKIP │ t=%.0fs d=$%.0f edge=%.1f%% │ %s",
                    snap.time_remaining, snap.distance, snap.edge, decision.reason
                )
            return

        # ── Execute entry ──────────────────────────────────────────
        token_id = market.token_id_up if decision.side == "UP" else market.token_id_down
        bet_size = self.risk.calculate_position_size()

        logger.info("━" * 50)
        logger.info(
            "🎯 ENTRY SIGNAL: %s │ edge=%.1f%% │ P=%.3f │ mkt=%.3f │ $%.2f",
            decision.side, decision.edge, decision.model_prob,
            decision.market_price, bet_size
        )

        result = self.executor.buy(token_id, bet_size, decision.side)

        if result["success"]:
            position = Position(
                side=decision.side,
                token_id=token_id,
                entry_price=result["fill_price"],
                size_usd=bet_size,
                entry_time=time.time(),
                market_id=market.market_id,
                entry_edge=decision.edge,
                entry_model_prob=decision.model_prob,
            )
            self.strategy.set_position(position)
            self._market_trade_count += 1
            self._status = f"IN_POSITION: {decision.side} @ {result['fill_price']:.3f}"

            logger.info(
                "✅ FILLED: %s @ %.3f │ %.2f shares │ %s",
                decision.side, result["fill_price"], result["shares"],
                "DRY" if result["dry_run"] else "LIVE"
            )
        else:
            logger.warning("❌ Entry failed: %s", result.get("error", "unknown"))
            self._status = "ENTRY_FAILED"

        logger.info("━" * 50)

    def _evaluate_exit(self, snap: SignalSnapshot, market: MarketInfo):
        """Decide whether to exit an open position."""
        decision = self.strategy.evaluate_exit(snap)
        pos = self.strategy.current_position

        if decision.action == Action.SKIP:
            self._status = f"HOLDING: {pos.side} @ {pos.entry_price:.3f} │ {decision.reason}"
            return

        # ── Execute exit ───────────────────────────────────────────
        logger.info("━" * 50)

        if decision.action == Action.EXIT_PROFIT:
            logger.info("💰 PROFIT TAKE: %s", decision.reason)
        else:
            logger.info("🛑 LOSS CUT: %s", decision.reason)

        result = self.executor.sell(pos.token_id, None, pos.side)

        if result["success"]:
            pnl = (decision.market_price - pos.entry_price) * pos.size_usd / pos.entry_price
            exit_reason = "PROFIT_TAKE" if decision.action == Action.EXIT_PROFIT else "LOSS_CUT"
            self._record_trade(pos, decision.market_price, pnl, exit_reason)
            self.strategy.clear_position()
            self._status = f"EXITED: {exit_reason} │ P&L: ${pnl:+.2f}"
        else:
            logger.warning("❌ Exit failed: %s — holding to expiry", result.get("error"))

        logger.info("━" * 50)

    def _record_trade(self, pos: Position, exit_price: float, pnl: float, reason: str):
        """Record a completed trade."""
        self.risk.record_trade_result(pnl)

        snap = self._last_signal or SignalSnapshot()
        self.store.log_trade({
            "market_id": pos.market_id,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size_usd": pos.size_usd,
            "pnl": round(pnl, 4),
            "distance": snap.distance,
            "time_remaining": snap.time_remaining,
            "volatility": snap.volatility_60s,
            "edge": pos.entry_edge,
            "model_prob": pos.entry_model_prob,
            "market_prob": pos.entry_price,
            "exit_reason": reason,
        })

        emoji = "📈" if pnl >= 0 else "📉"
        logger.info(
            "%s TRADE RESULT: %s │ entry=%.3f exit=%.3f │ P&L=$%.2f │ %s",
            emoji, pos.side, pos.entry_price, exit_price, pnl, reason
        )

    # ── Info ───────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Get full bot state for the dashboard."""
        snap = self._last_signal or SignalSnapshot()
        market = self._current_market

        return {
            "status": self._status,
            "dry_run": config.DRY_RUN,
            "btc_price": self.price_tracker.current_price,
            "ws_connected": self.price_tracker.is_connected,
            "market": {
                "question": market.question if market else "",
                "start_price": market.start_price if market else 0,
                "seconds_remaining": market.seconds_remaining if market else 0,
                "in_entry_window": market.is_in_entry_window if market else False,
            } if market else None,
            "signal": snap.to_dict() if snap else {},
            "position": {
                "side": self.strategy.current_position.side,
                "entry_price": self.strategy.current_position.entry_price,
                "entry_edge": self.strategy.current_position.entry_edge,
            } if self.strategy.has_position() else None,
            "risk": self.risk.get_status(),
            "stats": self.store.get_stats(),
        }


# ── Entry Point ────────────────────────────────────────────────────────

def main():
    setup_logging()
    bot = PolyBot()
    bot.start()


if __name__ == "__main__":
    main()
