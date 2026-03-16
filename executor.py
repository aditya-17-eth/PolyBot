"""
PolyBot — Order Executor
Handles trade execution via Polymarket CLOB API using py-clob-client.
Supports DRY_RUN mode for paper trading.
"""

import time
import logging

import config

logger = logging.getLogger("polybot.executor")


class Executor:
    """
    Places and manages orders on Polymarket.
    In DRY_RUN mode, simulates execution without spending real money.
    """

    def __init__(self):
        self._client = None
        self._authenticated = False
        self._init_client()

    def _init_client(self):
        """Initialize the Polymarket CLOB client."""
        if config.DRY_RUN:
            logger.info("🔸 Executor in DRY_RUN mode — no real trades")
            return

        if not config.POLYMARKET_PRIVATE_KEY:
            logger.warning("No private key configured — falling back to DRY_RUN")
            return

        try:
            from py_clob_client.client import ClobClient

            self._client = ClobClient(
                config.POLYMARKET_HOST,
                key=config.POLYMARKET_PRIVATE_KEY,
                chain_id=config.POLYMARKET_CHAIN_ID,
                signature_type=config.POLYMARKET_SIGNATURE_TYPE,
                funder=config.POLYMARKET_FUNDER_ADDRESS or None,
            )
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
            self._authenticated = True
            logger.info("✅ Polymarket client authenticated")
        except Exception as e:
            logger.error("Failed to initialize Polymarket client: %s", e)
            logger.warning("Falling back to DRY_RUN mode")
            self._client = None

    @property
    def is_live(self) -> bool:
        return self._authenticated and not config.DRY_RUN

    def buy(self, token_id: str, amount_usd: float, side_label: str = "") -> dict:
        """
        Buy a token (UP or DOWN outcome).

        Args:
            token_id: The CLOB token ID for the outcome
            amount_usd: Amount in USDC to spend
            side_label: "UP" or "DOWN" for logging

        Returns:
            dict with trade details
        """
        result = {
            "success": False,
            "token_id": token_id,
            "side": side_label,
            "amount_usd": amount_usd,
            "fill_price": 0.0,
            "shares": 0.0,
            "order_id": "",
            "dry_run": config.DRY_RUN,
            "error": "",
        }

        if config.DRY_RUN or not self._client:
            return self._simulate_buy(token_id, amount_usd, side_label, result)

        try:
            return self._execute_buy(token_id, amount_usd, side_label, result)
        except Exception as e:
            result["error"] = str(e)
            logger.error("Buy failed: %s", e)
            return result

    def sell(self, token_id: str, shares: float = None, side_label: str = "") -> dict:
        """
        Sell a position (for early exit / loss cut / profit take).

        Args:
            token_id: The CLOB token ID
            shares: Number of shares to sell (None = sell all)
            side_label: "UP" or "DOWN" for logging

        Returns:
            dict with trade details
        """
        result = {
            "success": False,
            "token_id": token_id,
            "side": side_label,
            "shares": shares,
            "fill_price": 0.0,
            "amount_usd": 0.0,
            "order_id": "",
            "dry_run": config.DRY_RUN,
            "error": "",
        }

        if config.DRY_RUN or not self._client:
            return self._simulate_sell(token_id, shares, side_label, result)

        try:
            return self._execute_sell(token_id, shares, side_label, result)
        except Exception as e:
            result["error"] = str(e)
            logger.error("Sell failed: %s", e)
            return result

    def get_balance(self) -> float:
        """Get current USDC balance on Polymarket."""
        if config.DRY_RUN or not self._client:
            return 1000.0  # Simulate $1000 bankroll

        try:
            # py-clob-client doesn't have a direct balance call;
            # we can check via the positions endpoint or on-chain
            # For now, return a placeholder — the real balance should
            # be tracked via the data store
            return 1000.0
        except Exception as e:
            logger.warning("Failed to get balance: %s", e)
            return 0.0

    def get_midpoint(self, token_id: str) -> float:
        """Get the current midpoint price for a token."""
        if not self._client:
            return 0.5

        try:
            mid = self._client.get_midpoint(token_id)
            return float(mid.get("mid", 0.5))
        except Exception as e:
            logger.warning("Failed to get midpoint: %s", e)
            return 0.5

    # ── Private: Live Execution ────────────────────────────────────────

    def _execute_buy(self, token_id: str, amount_usd: float, side_label: str, result: dict) -> dict:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        logger.info(
            "📤 LIVE BUY: %s | $%.2f | token=%s",
            side_label, amount_usd, token_id[:16]
        )

        args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usd,
            side=BUY,
            fee_rate_bps=156,  # 1.56%
            nonce=None,
        )

        signed_order = self._client.create_market_order(args)
        resp = self._client.post_order(signed_order, OrderType.FOK)

        if resp and resp.get("success"):
            result["success"] = True
            result["order_id"] = resp.get("orderID", "")
            # Estimate fill price from the response
            fills = resp.get("matchedOrders", [])
            if fills:
                total_cost = sum(float(f.get("makerAmountFilled", 0)) for f in fills)
                total_shares = sum(float(f.get("takerAmountFilled", 0)) for f in fills)
                if total_shares > 0:
                    result["fill_price"] = total_cost / total_shares
                    result["shares"] = total_shares
            logger.info("✅ BUY filled: order=%s", result["order_id"][:16])
        else:
            result["error"] = resp.get("errorMsg", "Unknown error") if resp else "Empty response"
            logger.warning("❌ BUY failed: %s", result["error"])

        return result

    def _execute_sell(self, token_id: str, shares: float, side_label: str, result: dict) -> dict:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        if shares is None or shares <= 0:
            result["error"] = "No shares to sell"
            return result

        logger.info(
            "📤 LIVE SELL: %s | %.2f shares | token=%s",
            side_label, shares, token_id[:16]
        )

        args = MarketOrderArgs(
            token_id=token_id,
            amount=shares,
            side=SELL,
            fee_rate_bps=156,
            nonce=None,
        )

        signed_order = self._client.create_market_order(args)
        resp = self._client.post_order(signed_order, OrderType.FOK)

        if resp and resp.get("success"):
            result["success"] = True
            result["order_id"] = resp.get("orderID", "")
            fills = resp.get("matchedOrders", [])
            if fills:
                total_received = sum(float(f.get("takerAmountFilled", 0)) for f in fills)
                result["amount_usd"] = total_received
                result["fill_price"] = total_received / shares if shares > 0 else 0
            logger.info("✅ SELL filled: order=%s", result["order_id"][:16])
        else:
            result["error"] = resp.get("errorMsg", "Unknown error") if resp else "Empty response"
            logger.warning("❌ SELL failed: %s", result["error"])

        return result

    # ── Private: Simulation ────────────────────────────────────────────

    def _simulate_buy(self, token_id: str, amount_usd: float, side_label: str, result: dict) -> dict:
        """Simulate a buy order."""
        # Estimate fill price from midpoint
        mid = 0.5
        try:
            import requests
            r = requests.get(
                f"{config.POLYMARKET_HOST}/midpoint",
                params={"token_id": token_id},
                timeout=5,
            )
            r.raise_for_status()
            mid = float(r.json().get("mid", 0.5))
        except Exception:
            pass

        fee = amount_usd * (config.MAX_FEE_PCT / 100)
        effective_amount = amount_usd - fee
        shares = effective_amount / mid if mid > 0 else 0

        result["success"] = True
        result["fill_price"] = mid
        result["shares"] = shares
        result["order_id"] = f"DRY-{int(time.time())}"

        logger.info(
            "🔸 DRY BUY: %s | $%.2f → %.2f shares @ %.3f | fee=$%.2f",
            side_label, amount_usd, shares, mid, fee
        )
        return result

    def _simulate_sell(self, token_id: str, shares: float, side_label: str, result: dict) -> dict:
        """Simulate a sell order."""
        mid = 0.5
        try:
            import requests
            r = requests.get(
                f"{config.POLYMARKET_HOST}/midpoint",
                params={"token_id": token_id},
                timeout=5,
            )
            r.raise_for_status()
            mid = float(r.json().get("mid", 0.5))
        except Exception:
            pass

        proceeds = (shares or 0) * mid
        fee = proceeds * (config.MAX_FEE_PCT / 100)
        net = proceeds - fee

        result["success"] = True
        result["fill_price"] = mid
        result["amount_usd"] = net
        result["shares"] = shares or 0
        result["order_id"] = f"DRY-SELL-{int(time.time())}"

        logger.info(
            "🔸 DRY SELL: %s | %.2f shares @ %.3f → $%.2f (fee=$%.2f)",
            side_label, shares or 0, mid, net, fee
        )
        return result
