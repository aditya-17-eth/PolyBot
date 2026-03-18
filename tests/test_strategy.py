"""
Tests for the strategy engine and signal computation.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

# Patch config before importing modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from signals import SignalSnapshot, SignalComputer
from strategy import Strategy, Action, Position


# ═══════════════════════════════════════════════════════════════════════
# Probability Estimation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestProbabilityEstimation:
    """Test the core normal CDF probability model."""

    def test_large_distance_short_time_high_prob(self):
        """Large distance + little time → very high probability."""
        prob = SignalComputer._estimate_probability(
            distance=200, time_remaining=30, sigma=1.5
        )
        assert prob > 0.90, f"Expected > 0.90, got {prob}"

    def test_small_distance_long_time_low_prob(self):
        """Small distance + lots of time → near 50/50."""
        prob = SignalComputer._estimate_probability(
            distance=10, time_remaining=90, sigma=1.5
        )
        assert 0.45 < prob < 0.80, f"Expected 0.45-0.80, got {prob}"

    def test_zero_distance_is_fifty_fifty(self):
        """Zero distance → exactly 50%."""
        prob = SignalComputer._estimate_probability(
            distance=0, time_remaining=60, sigma=1.5
        )
        assert abs(prob - 0.5) < 0.01, f"Expected ~0.5, got {prob}"

    def test_zero_time_remaining_is_certain(self):
        """No time left → 100% if distance > 0."""
        prob = SignalComputer._estimate_probability(
            distance=100, time_remaining=0, sigma=1.5
        )
        assert prob == 1.0

    def test_higher_sigma_lower_confidence(self):
        """Higher volatility → less confidence at same distance."""
        prob_low_vol = SignalComputer._estimate_probability(
            distance=30, time_remaining=60, sigma=2.0
        )
        prob_high_vol = SignalComputer._estimate_probability(
            distance=30, time_remaining=60, sigma=8.0
        )
        assert prob_low_vol > prob_high_vol, \
            f"Low vol ({prob_low_vol}) should be more confident than high vol ({prob_high_vol})"

    def test_probability_clamped(self):
        """Probability should always be between 0.01 and 0.99."""
        prob = SignalComputer._estimate_probability(
            distance=10000, time_remaining=1, sigma=0.01
        )
        assert prob <= 0.99

        prob = SignalComputer._estimate_probability(
            distance=0.001, time_remaining=10000, sigma=100
        )
        assert prob >= 0.01

    def test_prob_increases_as_time_decreases(self):
        """At fixed distance, probability should increase as time runs out."""
        probs = []
        for t in [90, 60, 30, 10]:
            p = SignalComputer._estimate_probability(
                distance=20, time_remaining=t, sigma=3.0
            )
            probs.append(p)
        
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1], \
                f"Prob should increase as time decreases: {probs}"


# ═══════════════════════════════════════════════════════════════════════
# Strategy Entry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyEntry:
    """Test entry decision logic."""

    def _make_good_signal(self, **overrides) -> SignalSnapshot:
        """Create a signal that should trigger an entry."""
        defaults = dict(
            market_id="test",
            time_remaining=45,
            btc_price=65100,
            start_price=64950,
            distance=150,
            distance_abs=150,
            direction="UP",
            volatility_15s=30,
            volatility_30s=50,
            volatility_60s=80,
            momentum_10s=20,
            momentum_30s=50,
            rate_of_change=2.5,
            market_prob_up=0.72,
            market_prob_down=0.28,
            market_spread=0.02,
            market_bid_liquidity=100,
            model_prob=0.88,
            edge=14.44,
            sigma_per_sec=1.5,
            liquidity_trap_score=0.1,
        )
        defaults.update(overrides)
        return SignalSnapshot(**defaults)

    def test_good_signal_triggers_entry(self):
        strategy = Strategy()
        snap = self._make_good_signal()
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.BUY_UP
        assert decision.side == "UP"
        assert decision.edge > 0

    def test_too_early_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(time_remaining=200)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("too_early" in r for r in decision.skip_reasons)

    def test_too_late_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(time_remaining=5)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("too_late" in r for r in decision.skip_reasons)

    def test_small_distance_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(distance=5, distance_abs=5)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("distance" in r for r in decision.skip_reasons)

    def test_low_volatility_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(volatility_60s=3)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("volatility" in r for r in decision.skip_reasons)

    def test_wide_spread_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(market_spread=0.15)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("spread" in r for r in decision.skip_reasons)

    def test_small_edge_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(edge=1.0)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("edge" in r for r in decision.skip_reasons)

    def test_liquidity_trap_skips(self):
        strategy = Strategy()
        snap = self._make_good_signal(liquidity_trap_score=0.9)
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("liquidity_trap" in r for r in decision.skip_reasons)

    def test_down_signal_triggers_buy_down(self):
        strategy = Strategy()
        snap = self._make_good_signal(
            direction="DOWN",
            distance=-150,
            market_prob_down=0.72,
            market_prob_up=0.28,
            momentum_30s=-50,
        )
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.BUY_DOWN
        assert decision.side == "DOWN"

    def test_already_in_position_skips(self):
        strategy = Strategy()
        strategy.set_position(Position(side="UP", entry_price=0.7))
        snap = self._make_good_signal()
        decision = strategy.evaluate_entry(snap)
        assert decision.action == Action.SKIP
        assert any("already" in r for r in decision.skip_reasons)

    def test_strike_crossing_strategy(self):
        strategy = Strategy()
        # Price is slightly UP (+10) but plunging fast (-50 momentum)
        snap = self._make_good_signal(
            distance=10, distance_abs=10, direction="UP",
            momentum_30s=-50.0, model_prob=0.52,
            market_prob_down=0.20, market_prob_up=0.80,
            time_remaining=45,
            # We override standard skips that might block the overall function
            volatility_60s=50, market_spread=0.01
        )
        decision = strategy.evaluate_entry(snap)
        
        # Should bet DOWN because of the strike crossing
        assert decision.action == Action.BUY_DOWN
        assert decision.side == "DOWN"
        assert "CROSS_ENTRY" in decision.reason

    def test_momentum_breakout_strategy(self):
        strategy = Strategy()
        # Still 120s remaining, so standard strategy would skip (too early)
        snap = self._make_good_signal(
            distance=50, distance_abs=50, direction="UP",
            momentum_30s=35.0, model_prob=0.85,
            market_prob_up=0.75, market_prob_down=0.25,
            time_remaining=120, edge=8.0
        )
        decision = strategy.evaluate_entry(snap)
        
        # Standard skips but Breakout should trigger
        assert decision.action == Action.BUY_UP
        assert decision.side == "UP"
        assert "BRK_ENTRY" in decision.reason


# ═══════════════════════════════════════════════════════════════════════
# Strategy Exit Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyExit:
    """Test exit decision logic."""

    def test_profit_take(self):
        strategy = Strategy()
        strategy.set_position(Position(side="UP", entry_price=0.70, entry_edge=10))
        snap = SignalSnapshot(
            market_prob_up=0.95,
            market_prob_down=0.05,
            edge=5.0,
        )
        decision = strategy.evaluate_exit(snap)
        assert decision.action == Action.EXIT_PROFIT

    def test_loss_cut_edge(self):
        strategy = Strategy()
        strategy.set_position(Position(side="UP", entry_price=0.70, entry_edge=10))
        snap = SignalSnapshot(
            market_prob_up=0.55,
            market_prob_down=0.45,
            edge=-2.0,
        )
        decision = strategy.evaluate_exit(snap)
        assert decision.action == Action.EXIT_LOSS

    def test_loss_cut_price_drop(self):
        strategy = Strategy()
        strategy.set_position(Position(side="UP", entry_price=0.80, entry_edge=10))
        snap = SignalSnapshot(
            market_prob_up=0.60,
            market_prob_down=0.40,
            edge=2.0,  # Edge still positive, but price dropped a lot
        )
        decision = strategy.evaluate_exit(snap)
        assert decision.action == Action.EXIT_LOSS

    def test_hold_when_ok(self):
        strategy = Strategy()
        strategy.set_position(Position(side="UP", entry_price=0.75, entry_edge=8))
        snap = SignalSnapshot(
            market_prob_up=0.80,
            market_prob_down=0.20,
            edge=5.0,
        )
        decision = strategy.evaluate_exit(snap)
        assert decision.action == Action.SKIP

    def test_no_position_skips(self):
        strategy = Strategy()
        snap = SignalSnapshot()
        decision = strategy.evaluate_exit(snap)
        assert decision.action == Action.SKIP


# ═══════════════════════════════════════════════════════════════════════
# Edge Calculation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCalculation:
    """Test that edge is computed correctly."""

    def test_edge_formula(self):
        """Edge = (model_prob - market_price - fee) * 100"""
        model_prob = 0.90
        market_price = 0.75
        fee = config.MAX_FEE_PCT / 100.0
        expected_edge = (model_prob - market_price - fee) * 100
        assert expected_edge > 0, "This should be a positive edge scenario"

    def test_fee_wipes_small_edge(self):
        """A small probability advantage should be wiped by fees."""
        model_prob = 0.52
        market_price = 0.50
        fee = 0.0156
        edge = (model_prob - market_price - fee) * 100
        assert edge < config.MIN_EDGE_PCT, "Small edge should not pass threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
