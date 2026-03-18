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
        """Evaluate whether to enter a trade using multiple sub-strategies."""
        if self.current_position:
            return TradeDecision(action=Action.SKIP, skip_reasons=["already_in_position"], reason="already_in_position")

        if not snap.direction:
            return TradeDecision(action=Action.SKIP, skip_reasons=["no_direction"], reason="no_direction")

        # Global disqualifiers
        global_skips = []
        if snap.market_spread > config.MAX_SPREAD:
            global_skips.append(f"spread_too_wide ({snap.market_spread:.4f} > {config.MAX_SPREAD})")
        if snap.liquidity_trap_score > 0.7:
            global_skips.append(f"liquidity_trap (score: {snap.liquidity_trap_score:.2f})")
        if snap.time_remaining <= config.EXIT_WINDOW_SECS:
            global_skips.append(f"too_late ({snap.time_remaining:.0f}s remaining)")
        if snap.volatility_60s < config.MIN_VOLATILITY_USD:
            global_skips.append(f"low_volatility (${snap.volatility_60s:.2f})")

        if global_skips:
            return TradeDecision(action=Action.SKIP, skip_reasons=global_skips, reason=global_skips[0])

        candidates = []
        skip_reasons = []

        # ── Strategy 1: Standard Late Window ───────────────────────
        s1_decision = self._eval_standard(snap)
        if s1_decision.action != Action.SKIP:
            candidates.append(s1_decision)
        else:
            skip_reasons.extend(s1_decision.skip_reasons)

        # ── Strategy 2: Strike Crossing (Mean Reversion) ───────────
        s2_decision = self._eval_strike_crossing(snap)
        if s2_decision.action != Action.SKIP:
            candidates.append(s2_decision)
        else:
            skip_reasons.extend(s2_decision.skip_reasons)

        # ── Strategy 3: Early Momentum Breakout ────────────────────
        s3_decision = self._eval_momentum_breakout(snap)
        if s3_decision.action != Action.SKIP:
            candidates.append(s3_decision)
        else:
            skip_reasons.extend(s3_decision.skip_reasons)

        if candidates:
            # Pick the candidate with the highest edge
            best_decision = max(candidates, key=lambda d: d.edge)
            return best_decision

        # Remove duplicates from skip reasons
        unique_skips = list(dict.fromkeys(skip_reasons))
        return TradeDecision(action=Action.SKIP, skip_reasons=unique_skips, reason=unique_skips[0] if unique_skips else "no_signal")

    def _eval_standard(self, snap: SignalSnapshot) -> TradeDecision:
        skips = []
        if snap.time_remaining > config.ENTRY_WINDOW_SECS:
            skips.append(f"std_too_early ({snap.time_remaining:.0f}s > {config.ENTRY_WINDOW_SECS})")
        if snap.distance_abs < config.MIN_DISTANCE_USD:
            skips.append(f"std_distance_too_small (${snap.distance_abs:.2f})")
        if snap.edge < config.MIN_EDGE_PCT:
            skips.append(f"std_edge_too_small ({snap.edge:.2f}%)")
            
        if snap.direction == "UP" and snap.momentum_30s < -snap.distance_abs * 0.3:
            skips.append("std_momentum_against_UP")
        elif snap.direction == "DOWN" and snap.momentum_30s > snap.distance_abs * 0.3:
            skips.append("std_momentum_against_DOWN")

        if skips:
            return TradeDecision(action=Action.SKIP, skip_reasons=skips)

        side = snap.direction
        market_price = snap.market_prob_up if side == "UP" else snap.market_prob_down
        return TradeDecision(
            action=Action.BUY_UP if side == "UP" else Action.BUY_DOWN,
            side=side,
            confidence=min(snap.model_prob, 0.99),
            edge=snap.edge,
            model_prob=snap.model_prob,
            market_price=market_price,
            reason=f"STD_ENTRY: d=${snap.distance:+.0f}, t={snap.time_remaining:.0f}s, edge={snap.edge:.1f}%"
        )

    def _eval_strike_crossing(self, snap: SignalSnapshot) -> TradeDecision:
        skips = []
        if snap.time_remaining > config.ENTRY_WINDOW_SECS:
            skips.append(f"cross_too_early")
        
        thresh = config.MIN_DISTANCE_USD * 1.5
        if snap.distance_abs > thresh:
            skips.append(f"cross_too_far (${snap.distance_abs:.2f} > {thresh})")

        opposite_side = "DOWN" if snap.direction == "UP" else "UP"
        momentum_req = 40.0
        
        if snap.direction == "UP" and snap.momentum_30s > -momentum_req:
            skips.append("cross_no_downward_momentum")
        elif snap.direction == "DOWN" and snap.momentum_30s < momentum_req:
            skips.append("cross_no_upward_momentum")

        cross_prob = 1.0 - snap.model_prob
        market_cross_prob = snap.market_prob_down if opposite_side == "DOWN" else snap.market_prob_up
        fee = config.MAX_FEE_PCT / 100.0
        cross_edge = (cross_prob - market_cross_prob - fee) * 100

        if cross_edge < config.MIN_EDGE_PCT:
            skips.append(f"cross_edge_too_small ({cross_edge:.2f}%)")

        if skips:
            return TradeDecision(action=Action.SKIP, skip_reasons=skips)

        return TradeDecision(
            action=Action.BUY_UP if opposite_side == "UP" else Action.BUY_DOWN,
            side=opposite_side,
            confidence=min(cross_prob, 0.99),
            edge=cross_edge,
            model_prob=cross_prob,
            market_price=market_cross_prob,
            reason=f"CROSS_ENTRY: betting {opposite_side}, current d=${snap.distance:+.0f}, mom=${snap.momentum_30s:.1f}"
        )

    def _eval_momentum_breakout(self, snap: SignalSnapshot) -> TradeDecision:
        skips = []
        if snap.time_remaining <= config.ENTRY_WINDOW_SECS:
            skips.append("brk_too_late")
        if snap.time_remaining > 240:
            skips.append("brk_too_early")
            
        if snap.distance_abs < config.MIN_DISTANCE_USD:
            skips.append("brk_dist_too_small")

        if snap.direction == "UP":
            if snap.momentum_30s < snap.distance_abs * 0.5:
                skips.append("brk_mom_too_weak_UP")
        else:
            if snap.momentum_30s > -snap.distance_abs * 0.5:
                skips.append("brk_mom_too_weak_DOWN")

        if snap.edge <= 0.5:
            skips.append(f"brk_edge_too_small ({snap.edge:.2f}%)")

        if skips:
            return TradeDecision(action=Action.SKIP, skip_reasons=skips)

        side = snap.direction
        market_price = snap.market_prob_up if side == "UP" else snap.market_prob_down
        return TradeDecision(
            action=Action.BUY_UP if side == "UP" else Action.BUY_DOWN,
            side=side,
            confidence=min(snap.model_prob, 0.99),
            edge=snap.edge,
            model_prob=snap.model_prob,
            market_price=market_price,
            reason=f"BRK_ENTRY: d=${snap.distance:+.0f}, mom=${snap.momentum_30s:.1f}, t={snap.time_remaining:.0f}s"
        )

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
