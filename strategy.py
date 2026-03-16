"""
PolyBot — Strategy Engine
Decides when to enter and exit trades based on signals.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from signals import SignalSnapshot
import config

logger = logging.getLogger("polybot.strategy")


class Action(Enum):
    SKIP = "SKIP"
    BUY_UP = "BUY_UP"
    BUY_DOWN = "BUY_DOWN"
    EXIT_PROFIT = "EXIT_PROFIT"
    EXIT_LOSS = "EXIT_LOSS"


@dataclass
class TradeDecision:
    """Output of the strategy engine."""
    action: Action = Action.SKIP
    side: str = ""                # "UP" or "DOWN"
    confidence: float = 0.0       # 0 to 1
    edge: float = 0.0             # Probability edge %
    model_prob: float = 0.0
    market_price: float = 0.0
    reason: str = ""
    skip_reasons: list = None     # All reasons for skipping

    def __post_init__(self):
        if self.skip_reasons is None:
            self.skip_reasons = []


@dataclass
class Position:
    """Tracks an open position."""
    side: str = ""                # "UP" or "DOWN"
    token_id: str = ""
    entry_price: float = 0.0
    size_usd: float = 0.0
    entry_time: float = 0.0
    market_id: str = ""
    entry_edge: float = 0.0
    entry_model_prob: float = 0.0


class Strategy:
    """
    Entry/exit decision engine.

    Entry conditions (ALL must be met):
    1. Time remaining ≤ ENTRY_WINDOW_SECS and > EXIT_WINDOW_SECS
    2. Distance from start price ≥ MIN_DISTANCE_USD
    3. Edge (model_prob - market_price - fee) ≥ MIN_EDGE_PCT
    4. 60s volatility ≥ MIN_VOLATILITY_USD
    5. Market spread ≤ MAX_SPREAD
    6. Liquidity trap score < 0.7
    7. Momentum direction aligns with distance direction

    Exit conditions (ANY triggers exit):
    - Profit take: market price ≥ PROFIT_TAKE_THRESHOLD
    - Loss cut: edge drops below LOSS_CUT_EDGE_MIN
    - Loss cut: position price dropped LOSS_CUT_PRICE_DROP from entry
    """

    def __init__(self):
        self.current_position: Position = None

    def evaluate_entry(self, snap: SignalSnapshot) -> TradeDecision:
        """Evaluate whether to enter a trade."""
        decision = TradeDecision()
        skip_reasons = []

        # ── Time Check ─────────────────────────────────────────────
        if snap.time_remaining > config.ENTRY_WINDOW_SECS:
            skip_reasons.append(f"too_early ({snap.time_remaining:.0f}s remaining)")

        if snap.time_remaining <= config.EXIT_WINDOW_SECS:
            skip_reasons.append(f"too_late ({snap.time_remaining:.0f}s remaining)")

        # ── Direction Check ────────────────────────────────────────
        if not snap.direction:
            skip_reasons.append("no_direction (price at start)")

        # ── Distance Check ─────────────────────────────────────────
        if snap.distance_abs < config.MIN_DISTANCE_USD:
            skip_reasons.append(
                f"distance_too_small (${snap.distance_abs:.2f} < ${config.MIN_DISTANCE_USD})"
            )

        # ── Volatility Check ───────────────────────────────────────
        if snap.volatility_60s < config.MIN_VOLATILITY_USD:
            skip_reasons.append(
                f"low_volatility (${snap.volatility_60s:.2f} < ${config.MIN_VOLATILITY_USD})"
            )

        # ── Spread Check ───────────────────────────────────────────
        if snap.market_spread > config.MAX_SPREAD:
            skip_reasons.append(
                f"spread_too_wide ({snap.market_spread:.4f} > {config.MAX_SPREAD})"
            )

        # ── Liquidity Trap Check ───────────────────────────────────
        if snap.liquidity_trap_score > 0.7:
            skip_reasons.append(
                f"liquidity_trap (score: {snap.liquidity_trap_score:.2f})"
            )

        # ── Edge Check ─────────────────────────────────────────────
        if snap.edge < config.MIN_EDGE_PCT:
            skip_reasons.append(
                f"edge_too_small ({snap.edge:.2f}% < {config.MIN_EDGE_PCT}%)"
            )

        # ── Momentum Alignment ─────────────────────────────────────
        # If distance is UP, momentum should not be strongly negative
        if snap.direction == "UP" and snap.momentum_30s < -snap.distance_abs * 0.3:
            skip_reasons.append("momentum_against_direction")
        elif snap.direction == "DOWN" and snap.momentum_30s > snap.distance_abs * 0.3:
            skip_reasons.append("momentum_against_direction")

        # ── Already in position ────────────────────────────────────
        if self.current_position:
            skip_reasons.append("already_in_position")

        # ── Decision ───────────────────────────────────────────────
        if skip_reasons:
            decision.action = Action.SKIP
            decision.skip_reasons = skip_reasons
            decision.reason = skip_reasons[0]  # Primary reason
            return decision

        # All conditions met — generate entry signal
        if snap.direction == "UP":
            decision.action = Action.BUY_UP
            decision.side = "UP"
            decision.market_price = snap.market_prob_up
        else:
            decision.action = Action.BUY_DOWN
            decision.side = "DOWN"
            decision.market_price = snap.market_prob_down

        decision.edge = snap.edge
        decision.model_prob = snap.model_prob
        decision.confidence = min(snap.model_prob, 0.99)
        decision.reason = (
            f"ENTRY: d=${snap.distance:+.0f}, "
            f"t={snap.time_remaining:.0f}s, "
            f"edge={snap.edge:.1f}%, "
            f"P={snap.model_prob:.3f}"
        )

        return decision

    def evaluate_exit(self, snap: SignalSnapshot) -> TradeDecision:
        """Evaluate whether to exit an open position."""
        decision = TradeDecision()

        if not self.current_position:
            decision.action = Action.SKIP
            decision.reason = "no_position"
            return decision

        pos = self.current_position

        # Get current market price for our side
        if pos.side == "UP":
            current_price = snap.market_prob_up
        else:
            current_price = snap.market_prob_down

        # ── Profit Take ────────────────────────────────────────────
        if current_price >= config.PROFIT_TAKE_THRESHOLD:
            decision.action = Action.EXIT_PROFIT
            decision.side = pos.side
            decision.market_price = current_price
            decision.reason = (
                f"PROFIT_TAKE: price {current_price:.3f} >= "
                f"{config.PROFIT_TAKE_THRESHOLD} (entry: {pos.entry_price:.3f})"
            )
            return decision

        # ── Loss Cut — Edge Disappeared ────────────────────────────
        if snap.edge < config.LOSS_CUT_EDGE_MIN:
            decision.action = Action.EXIT_LOSS
            decision.side = pos.side
            decision.market_price = current_price
            decision.reason = (
                f"LOSS_CUT_EDGE: edge {snap.edge:.2f}% < "
                f"{config.LOSS_CUT_EDGE_MIN}% (entry edge: {pos.entry_edge:.2f}%)"
            )
            return decision

        # ── Loss Cut — Price Drop ──────────────────────────────────
        price_drop = pos.entry_price - current_price
        if price_drop > config.LOSS_CUT_PRICE_DROP:
            decision.action = Action.EXIT_LOSS
            decision.side = pos.side
            decision.market_price = current_price
            decision.reason = (
                f"LOSS_CUT_PRICE: dropped {price_drop:.3f} from entry "
                f"{pos.entry_price:.3f} → {current_price:.3f}"
            )
            return decision

        # ── Momentum Reversal ──────────────────────────────────────
        # If momentum has strongly reversed against our position
        if pos.side == "UP" and snap.momentum_30s < -snap.distance_abs * 0.5:
            if current_price < pos.entry_price:
                decision.action = Action.EXIT_LOSS
                decision.side = pos.side
                decision.market_price = current_price
                decision.reason = "LOSS_CUT_MOMENTUM: strong reversal against UP"
                return decision

        if pos.side == "DOWN" and snap.momentum_30s > snap.distance_abs * 0.5:
            if current_price < pos.entry_price:
                decision.action = Action.EXIT_LOSS
                decision.side = pos.side
                decision.market_price = current_price
                decision.reason = "LOSS_CUT_MOMENTUM: strong reversal against DOWN"
                return decision

        # Hold position
        decision.action = Action.SKIP
        decision.reason = f"HOLD: price={current_price:.3f}, edge={snap.edge:.2f}%"
        return decision

    def set_position(self, position: Position):
        self.current_position = position

    def clear_position(self):
        self.current_position = None

    def has_position(self) -> bool:
        return self.current_position is not None


# ── Standalone Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    strategy = Strategy()

    # Simulate a good signal
    snap = SignalSnapshot(
        market_id="test",
        time_remaining=45,
        btc_price=65100,
        start_price=64950,
        distance=150,
        distance_abs=150,
        direction="UP",
        volatility_60s=80,
        momentum_30s=50,
        rate_of_change=2.5,
        market_prob_up=0.72,
        market_prob_down=0.28,
        market_spread=0.02,
        model_prob=0.88,
        edge=14.44,
        sigma_per_sec=1.5,
    )

    decision = strategy.evaluate_entry(snap)
    print(f"\n  Decision: {decision.action.value}")
    print(f"  Side: {decision.side}")
    print(f"  Edge: {decision.edge:.2f}%")
    print(f"  Reason: {decision.reason}")
    if decision.skip_reasons:
        for r in decision.skip_reasons:
            print(f"    - {r}")

    # Test with poor signal
    snap2 = SignalSnapshot(
        market_id="test",
        time_remaining=45,
        btc_price=64960,
        start_price=64950,
        distance=10,
        distance_abs=10,
        direction="UP",
        volatility_60s=5,
        market_prob_up=0.52,
        market_spread=0.08,
        model_prob=0.53,
        edge=0.5,
    )

    decision2 = strategy.evaluate_entry(snap2)
    print(f"\n  Decision: {decision2.action.value}")
    print(f"  Skip reasons:")
    for r in decision2.skip_reasons:
        print(f"    - {r}")
