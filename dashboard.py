"""
PolyBot — Web Dashboard
Flask-based monitoring dashboard showing bot status, trades, and signals.
"""

import json
import logging
import threading
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify

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
    return jsonify({
        "status": "DASHBOARD_ONLY",
        "dry_run": config.DRY_RUN,
        "btc_price": 0,
        "ws_connected": False,
        "market": None,
        "signal": {},
        "position": None,
        "risk": {
            "can_trade": True,
            "reason": "ok",
            "bankroll": 0,
            "daily_pnl": _store.get_daily_pnl() if _store else 0,
            "daily_trades": _store.get_daily_trade_count() if _store else 0,
            "consecutive_losses": 0,
            "next_position_size": 0,
            "cooldown_active": False,
        },
        "stats": _store.get_stats() if _store else {},
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
