"""
PolyBot — Web Dashboard
Flask-based monitoring dashboard showing bot status, trades, and signals.
"""

import json
import logging
import threading
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify, request

import config
from data_store import DataStore

logger = logging.getLogger("polybot.dashboard")

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "polybot-dashboard"

# Shared state — set by the bot
_bot_ref = None
_store = None


def init_dashboard(bot=None, store=None):
    """Initialize dashboard with bot reference and data store."""
    global _bot_ref, _store
    _bot_ref = bot
    _store = store or DataStore()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    """Full bot state for the dashboard."""
    if _bot_ref:
        return jsonify(_bot_ref.get_state())

    # Fallback: just return data store info
    stats = _store.get_stats() if _store else {}
    daily_pnl = _store.get_daily_pnl() if _store else 0
    daily_trades = _store.get_daily_trade_count() if _store else 0

    return jsonify({
        "status": "DASHBOARD_ONLY",
        "dry_run": config.DRY_RUN,
        "btc_price": 0,
        "ws_connected": False,
        "market": None,
        "signal": {},
        "position": None,
        "risk": {
            "can_trade": False,
            "reason": "bot_not_running",
            "bankroll": 0,
            "daily_pnl": daily_pnl,
            "daily_trades": daily_trades,
            "consecutive_losses": 0,
            "next_position_size": 0,
            "cooldown_active": False,
        },
        "strategies": {
            "standard": getattr(config, "ENABLE_STRATEGY_STANDARD", True),
            "crossing": getattr(config, "ENABLE_STRATEGY_CROSSING", True),
            "breakout": getattr(config, "ENABLE_STRATEGY_BREAKOUT", True),
        },
        "stats": stats,
    })


@app.route("/api/trades")
def api_trades():
    """Recent trades."""
    if _store:
        return jsonify(_store.get_recent_trades(50))
    return jsonify([])


@app.route("/api/signals")
def api_signals():
    """Recent signals."""
    if _store:
        return jsonify(_store.get_recent_signals(50))
    return jsonify([])


@app.route("/api/strategies", methods=["POST"])
def api_update_strategies():
    """Update active trading strategies."""
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
        
    if "standard" in data:
        config.ENABLE_STRATEGY_STANDARD = bool(data["standard"])
    if "crossing" in data:
        config.ENABLE_STRATEGY_CROSSING = bool(data["crossing"])
    if "breakout" in data:
        config.ENABLE_STRATEGY_BREAKOUT = bool(data["breakout"])
        
    return jsonify({"success": True})


@app.route("/api/stats")
def api_stats():
    """Overall stats."""
    if _store:
        return jsonify(_store.get_stats())
    return jsonify({})


def run_dashboard(bot=None, store=None):
    """Run dashboard in a separate thread."""
    init_dashboard(bot, store)
    thread = threading.Thread(
        target=lambda: app.run(
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    thread.start()
    logger.info("Dashboard running at http://%s:%d", config.DASHBOARD_HOST, config.DASHBOARD_PORT)
    return thread


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = DataStore()
    init_dashboard(store=store)
    print(f"\n🌐 Dashboard: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}\n")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=True)
