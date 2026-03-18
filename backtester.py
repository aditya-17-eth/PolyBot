"""
PolyBot — Backtesting Engine
Replays historical BTC price data and simulates 15-minute markets to test strategy.
"""

import os
import csv
import math
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

from scipy.stats import norm

import config

logger = logging.getLogger("polybot.backtester")


@dataclass
class BacktestTrade:
    """Record of a single backtested trade."""
    market_start: float = 0.0
    entry_time: float = 0.0
    exit_time: float = 0.0
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    distance: float = 0.0
    time_remaining: float = 0.0
    model_prob: float = 0.0
    edge: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    total_markets: int = 0
    markets_traded: int = 0
    markets_skipped: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_edge: float = 0.0
    trades: list = field(default_factory=list)


class Backtester:
    """
    Simulates the bot strategy on historical BTC price data.
    
    Expected CSV format:
        timestamp,price
        1700000000,64500.50
        1700000001,64501.20
        ...
    
    One row per second (or close to it).
    """

    def __init__(self):
        self.bet_size = 10.0  # Fixed bet size for backtesting
        self.fee_pct = config.MAX_FEE_PCT / 100.0

    def run(self, price_file: str, market_duration: int = 900) -> BacktestResult:
        """
        Run a backtest on a price data file.
        
        Args:
            price_file: Path to CSV with (timestamp, price) rows
            market_duration: Simulated market duration in seconds (default 900 = 15min)
        """
        prices = self._load_prices(price_file)
        if not prices:
            logger.error("No price data loaded from %s", price_file)
            return BacktestResult()

        logger.info("Loaded %d price ticks covering %.1f hours",
                     len(prices), (prices[-1][0] - prices[0][0]) / 3600)

        result = BacktestResult()
        trades = []
        equity_curve = [0.0]
        peak_equity = 0.0
        max_drawdown = 0.0

        # Simulate markets
        market_start_idx = 0
        while market_start_idx < len(prices) - market_duration:
            market_prices = prices[market_start_idx:market_start_idx + market_duration]
            start_price = market_prices[0][1]
            result.total_markets += 1

            trade = self._simulate_market(market_prices, start_price)

            if trade:
                result.markets_traded += 1
                trades.append(trade)

                # Update equity
                equity = equity_curve[-1] + trade.pnl
                equity_curve.append(equity)
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)
            else:
                result.markets_skipped += 1

            # Move to next market
            market_start_idx += market_duration

        # Compute stats
        result.total_trades = len(trades)
        result.trades = trades

        if trades:
            result.wins = sum(1 for t in trades if t.pnl > 0)
            result.losses = sum(1 for t in trades if t.pnl <= 0)
            result.win_rate = result.wins / result.total_trades * 100
            result.total_pnl = sum(t.pnl for t in trades)
            result.avg_pnl = result.total_pnl / result.total_trades
            result.max_drawdown = max_drawdown
            result.best_trade = max(t.pnl for t in trades)
            result.worst_trade = min(t.pnl for t in trades)
            result.avg_edge = sum(t.edge for t in trades) / result.total_trades

        return result

    def _simulate_market(self, market_prices: list, start_price: float) -> BacktestTrade:
        """
        Simulate a single 15-minute market.
        Returns a trade if one was taken, None if skipped.
        """
        duration = len(market_prices)
        entry_start = duration - config.ENTRY_WINDOW_SECS
        entry_end = duration - config.EXIT_WINDOW_SECS

        if entry_start < 60:  # Need at least 60s of price history
            return None

        # Compute volatility from the price data in this market
        recent_prices = [p[1] for p in market_prices[max(0, entry_start - 60):entry_start]]
        if len(recent_prices) < 10:
            return None

        # Per-second volatility
        returns = []
        for i in range(1, len(recent_prices)):
            returns.append(recent_prices[i] - recent_prices[i - 1])
        
        if not returns:
            return None
            
        sigma = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
        vol_60s = max(recent_prices) - min(recent_prices)

        # Scan for entry opportunity
        for i in range(entry_start, min(entry_end, duration)):
            current_price = market_prices[i][1]
            time_remaining = duration - i
            distance = current_price - start_price
            distance_abs = abs(distance)

            # Apply entry conditions
            if distance_abs < config.MIN_DISTANCE_USD:
                continue
            if vol_60s < config.MIN_VOLATILITY_USD:
                continue
            if sigma <= 0:
                continue

            # Estimate probability
            denom = sigma * math.sqrt(time_remaining)
            if denom <= 0:
                continue
            z = distance_abs / denom
            model_prob = norm.cdf(z)

            # Simulate market price (assume market is somewhat efficient)
            # Market price roughly tracks our model but with a lag
            market_price = model_prob * 0.85 + 0.5 * 0.15  # Market underestimates by ~15%

            # Edge
            edge = (model_prob - market_price - self.fee_pct) * 100

            if edge < config.MIN_EDGE_PCT:
                continue

            # ── Take the trade ─────────────────────────────────────
            side = "UP" if distance > 0 else "DOWN"
            entry_price = market_price

            # Determine if trade wins (check final price)
            final_price = market_prices[-1][1]
            final_distance = final_price - start_price

            if (side == "UP" and final_distance > 0) or (side == "DOWN" and final_distance < 0):
                # Win: position settles at 1.0
                exit_price = 1.0
            else:
                # Loss: position settles at 0.0
                exit_price = 0.0

            # Check for early exit opportunities
            for j in range(i + 1, duration):
                check_price = market_prices[j][1]
                check_dist = abs(check_price - start_price)
                check_time = duration - j
                
                if check_time > 0 and sigma > 0:
                    check_denom = sigma * math.sqrt(check_time)
                    check_z = check_dist / check_denom
                    check_prob = norm.cdf(check_z)
                    sim_market = check_prob * 0.9 + 0.5 * 0.1

                    # Profit take
                    if sim_market >= config.PROFIT_TAKE_THRESHOLD:
                        exit_price = sim_market
                        break

                    # Loss cut — direction reversed
                    reversed_dir = (side == "UP" and check_price < start_price) or \
                                   (side == "DOWN" and check_price > start_price)
                    if reversed_dir and sim_market < entry_price:
                        exit_price = 1.0 - sim_market  # We're on the wrong side now
                        break

            pnl = (exit_price - entry_price) * self.bet_size / entry_price
            pnl -= self.bet_size * self.fee_pct  # Subtract fee

            return BacktestTrade(
                market_start=market_prices[0][0],
                entry_time=market_prices[i][0],
                exit_time=market_prices[-1][0],
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=round(pnl, 4),
                distance=distance,
                time_remaining=time_remaining,
                model_prob=model_prob,
                edge=edge,
                exit_reason="SETTLE" if exit_price in (0, 1) else "EARLY_EXIT",
            )

        return None

    @staticmethod
    def _load_prices(filepath: str) -> list:
        """Load price data from CSV."""
        prices = []
        try:
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        ts = float(row[0])
                        price = float(row[1])
                        prices.append((ts, price))
        except Exception as e:
            logger.error("Failed to load %s: %s", filepath, e)
        return prices

    @staticmethod
    def print_results(result: BacktestResult):
        """Pretty-print backtest results."""
        div = "=" * 60
        subdiv = "-" * 37
        print(f"\n{div}")
        print("  BACKTEST RESULTS")
        print(div)
        print(f"  Markets simulated:  {result.total_markets}")
        print(f"  Markets traded:     {result.markets_traded}")
        print(f"  Markets skipped:    {result.markets_skipped}")
        print(f"  Total trades:       {result.total_trades}")
        print(f"  Wins:               {result.wins}")
        print(f"  Losses:             {result.losses}")
        print(f"  Win rate:           {result.win_rate:.1f}%")
        print(f"  {subdiv}")
        print(f"  Total P&L:          ${result.total_pnl:+.2f}")
        print(f"  Average P&L:        ${result.avg_pnl:+.4f}")
        print(f"  Best trade:         ${result.best_trade:+.4f}")
        print(f"  Worst trade:        ${result.worst_trade:+.4f}")
        print(f"  Max drawdown:       ${result.max_drawdown:.2f}")
        print(f"  Average edge:       {result.avg_edge:.2f}%")
        print(f"{div}\n")


# ── Generate Sample Data ──────────────────────────────────────────────

def generate_sample_data(filepath: str, hours: int = 24):
    """Generate synthetic BTC price data for backtesting."""
    import random

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    price = 65000.0
    timestamp = 1700000000.0
    total_seconds = hours * 3600

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price"])

        for _ in range(total_seconds):
            # Random walk with slight drift and occasional jumps
            change = random.gauss(0, 1.5)  # ~$1.5 per second stdev
            if random.random() < 0.001:  # Occasional jump
                change += random.choice([-50, 50])
            price += change
            price = max(price, 50000)  # Floor
            writer.writerow([timestamp, round(price, 2)])
            timestamp += 1

    print(f"  Generated {total_seconds} ticks ({hours}h) to {filepath}")


# ── Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sample_file = os.path.join(config.BACKTEST_DATA_DIR, "sample_btc.csv")

    if not os.path.exists(sample_file):
        print("\n📁 No backtest data found. Generating sample data...\n")
        generate_sample_data(sample_file, hours=48)

    bt = Backtester()
    result = bt.run(sample_file)
    bt.print_results(result)
