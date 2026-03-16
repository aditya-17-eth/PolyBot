"""
PolyBot — Risk Management
Position sizing, daily limits, cooldown timers, and loss protection.
"""

import time
import logging
from datetime import datetime, timezone

from data_store import DataStore
import config

logger = logging.getLogger("polybot.risk")


class RiskManager:
    """
    Manages risk controls:
    - Position sizing as % of bankroll
    - Daily loss limit
    - Daily trade count limit
    - Cooldown after losses
    - Consecutive loss tracking
    """

    def __init__(self, data_store: DataStore, initial_bankroll: float = 1000.0):
        self.store = data_store
        self.bankroll = initial_bankroll
        self._last_loss_time: float = 0
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0
        self._daily_trades: int = 0
        self._last_reset_date: str = ""

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed right now.
        Returns (allowed, reason).
        """
        self._check_daily_reset()

        # Daily loss limit
        daily_pnl = self.store.get_daily_pnl()
        if daily_pnl <= -config.MAX_DAILY_LOSS_USD:
            return False, f"daily_loss_limit (${daily_pnl:.2f} <= -${config.MAX_DAILY_LOSS_USD})"

        # Daily trade count
        daily_trades = self.store.get_daily_trade_count()
        if daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"max_daily_trades ({daily_trades} >= {config.MAX_DAILY_TRADES})"

        # Cooldown after loss
        if self._last_loss_time > 0:
            cooldown = config.COOLDOWN_AFTER_LOSS_SECS
            if self._consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
                cooldown = config.EXTENDED_COOLDOWN_SECS

            elapsed = time.time() - self._last_loss_time
            if elapsed < cooldown:
                remaining = cooldown - elapsed
                return False, f"cooldown ({remaining:.0f}s remaining, {self._consecutive_losses} consecutive losses)"

        return True, "ok"

    def calculate_position_size(self) -> float:
        """
        Calculate the USD amount to risk on this trade.
        Uses a fixed percentage of current bankroll.
        """
        size = self.bankroll * (config.BET_SIZE_PCT / 100.0)
        size = max(size, config.MIN_BET_USD)
        size = min(size, config.MAX_BET_USD)

        # Further reduce after consecutive losses (Kelly-like scaling)
        if self._consecutive_losses > 0:
            scale = max(0.25, 1.0 - (self._consecutive_losses * 0.2))
            size *= scale
            logger.info(
                "Position scaled down to %.1f%% due to %d consecutive losses",
                scale * 100, self._consecutive_losses
            )

        return round(size, 2)

    def record_trade_result(self, pnl: float):
        """Record the result of a completed trade."""
        self.bankroll += pnl

        if pnl < 0:
            self._last_loss_time = time.time()
            self._consecutive_losses += 1
            logger.info(
                "📉 Loss recorded: $%.2f | Consecutive losses: %d | Bankroll: $%.2f",
                pnl, self._consecutive_losses, self.bankroll
            )
        else:
            self._consecutive_losses = 0
            logger.info(
                "📈 Win recorded: $%.2f | Bankroll: $%.2f",
                pnl, self.bankroll
            )

    def get_status(self) -> dict:
        """Get current risk status."""
        can, reason = self.can_trade()
        return {
            "can_trade": can,
            "reason": reason,
            "bankroll": round(self.bankroll, 2),
            "daily_pnl": round(self.store.get_daily_pnl(), 2),
            "daily_trades": self.store.get_daily_trade_count(),
            "consecutive_losses": self._consecutive_losses,
            "next_position_size": self.calculate_position_size(),
            "cooldown_active": not can and "cooldown" in reason,
        }

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self._last_reset_date = today
            logger.info("📅 New trading day: %s", today)
