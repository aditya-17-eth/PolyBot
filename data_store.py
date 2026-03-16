"""
PolyBot — Data Store
SQLite-backed storage for trades, signals, and price history.
"""

import sqlite3
import json
import time
import threading
from datetime import datetime, timezone
import config


class DataStore:
    """Thread-safe SQLite data store for bot telemetry."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                market_id   TEXT,
                side        TEXT,
                entry_price REAL,
                exit_price  REAL,
                size_usd    REAL,
                pnl         REAL,
                distance    REAL,
                time_remaining REAL,
                volatility  REAL,
                edge        REAL,
                model_prob  REAL,
                market_prob REAL,
                exit_reason TEXT,
                dry_run     INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                market_id       TEXT,
                btc_price       REAL,
                start_price     REAL,
                distance        REAL,
                time_remaining  REAL,
                volatility_15s  REAL,
                volatility_30s  REAL,
                volatility_60s  REAL,
                momentum        REAL,
                rate_of_change  REAL,
                high_since_start REAL,
                low_since_start  REAL,
                ob_imbalance    REAL,
                buy_wall_size   REAL,
                sell_wall_size  REAL,
                spread          REAL,
                market_prob_up  REAL,
                market_prob_down REAL,
                model_prob      REAL,
                edge            REAL,
                consec_ticks    INTEGER,
                vol_acceleration REAL,
                signal          TEXT
            );

            CREATE TABLE IF NOT EXISTS price_ticks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                price     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS market_sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id       TEXT UNIQUE,
                condition_id    TEXT,
                start_time      TEXT,
                end_time        TEXT,
                start_price     REAL,
                token_id_up     TEXT,
                token_id_down   TEXT,
                outcome         TEXT,
                trades_taken    INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
            CREATE INDEX IF NOT EXISTS idx_price_ts ON price_ticks(timestamp);
        """)
        conn.commit()

    # ── Trade Logging ──────────────────────────────────────────────────

    def log_trade(self, trade: dict):
        conn = self._get_conn()
        trade["timestamp"] = datetime.now(timezone.utc).isoformat()
        trade["dry_run"] = 1 if config.DRY_RUN else 0
        conn.execute("""
            INSERT INTO trades (timestamp, market_id, side, entry_price, exit_price,
                size_usd, pnl, distance, time_remaining, volatility, edge, 
                model_prob, market_prob, exit_reason, dry_run)
            VALUES (:timestamp, :market_id, :side, :entry_price, :exit_price,
                :size_usd, :pnl, :distance, :time_remaining, :volatility, :edge,
                :model_prob, :market_prob, :exit_reason, :dry_run)
        """, trade)
        conn.commit()

    def get_trades_today(self) -> list:
        conn = self._get_conn()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
            (today,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self) -> float:
        trades = self.get_trades_today()
        return sum(t.get("pnl", 0) or 0 for t in trades)

    def get_daily_trade_count(self) -> int:
        return len(self.get_trades_today())

    # ── Signal Logging ─────────────────────────────────────────────────

    def log_signal(self, sig: dict):
        conn = self._get_conn()
        sig["timestamp"] = datetime.now(timezone.utc).isoformat()
        cols = [
            "timestamp", "market_id", "btc_price", "start_price", "distance",
            "time_remaining", "volatility_15s", "volatility_30s", "volatility_60s",
            "momentum", "rate_of_change", "high_since_start", "low_since_start",
            "ob_imbalance", "buy_wall_size", "sell_wall_size", "spread",
            "market_prob_up", "market_prob_down", "model_prob", "edge",
            "consec_ticks", "vol_acceleration", "signal"
        ]
        placeholders = ", ".join(f":{c}" for c in cols)
        col_names = ", ".join(cols)
        # Fill missing fields with None
        for c in cols:
            sig.setdefault(c, None)
        conn.execute(f"INSERT INTO signals ({col_names}) VALUES ({placeholders})", sig)
        conn.commit()

    def get_recent_signals(self, limit: int = 100) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Price Ticks ────────────────────────────────────────────────────

    def log_price_tick(self, price: float, ts: float = None):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO price_ticks (timestamp, price) VALUES (?, ?)",
            (ts or time.time(), price)
        )
        conn.commit()

    # ── Market Sessions ────────────────────────────────────────────────

    def log_market_session(self, session: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO market_sessions 
                (market_id, condition_id, start_time, end_time, start_price, 
                 token_id_up, token_id_down, outcome, trades_taken)
            VALUES (:market_id, :condition_id, :start_time, :end_time, :start_price,
                    :token_id_up, :token_id_down, :outcome, :trades_taken)
        """, session)
        conn.commit()

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        wins = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0]
        total_pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()[0]
        avg_pnl = conn.execute("SELECT COALESCE(AVG(pnl), 0) FROM trades").fetchone()[0]
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 4),
            "daily_pnl": round(self.get_daily_pnl(), 2),
            "daily_trades": self.get_daily_trade_count(),
        }


if __name__ == "__main__":
    store = DataStore()
    print("Database initialized at:", config.DB_PATH)
    print("Stats:", store.get_stats())
