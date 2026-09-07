"""
Production Database Module — SQLite persistent storage.
No simulated data. Stores real analysis results, feed health, and trade signals.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///data/trader.db").replace("sqlite:///", "")


class ProductionDB:
    """Production-grade persistent storage for trading analysis."""

    def __init__(self):
        self.db_path = DB_PATH if not DB_PATH.startswith("sqlite:///") else DB_PATH.replace("sqlite:///", "")
        # Ensure absolute path relative to project root
        if not self.db_path.startswith("/"):
            self.db_path = os.path.join(os.getcwd(), self.db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                dominant_strategy TEXT,
                dominant_score REAL,
                sentiment TEXT,
                vol_regime TEXT,
                recommendation TEXT,
                rate REAL,
                warning_text TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feed_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                feeds_active INTEGER,
                rate_source TEXT,
                vol_source TEXT,
                error_message TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT,
                strategy_name TEXT,
                score REAL,
                confidence TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_analysis(self, result: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_results
            (pair, timestamp, dominant_strategy, dominant_score, sentiment, vol_regime, recommendation, rate, warning_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("pair"),
            datetime.now().isoformat(),
            result.get("dominant_strategy"),
            result.get("dominant_score"),
            result.get("news_sentiment"),
            result.get("volatility", {}).get("regime"),
            result.get("master_recommendation", "").split("\n")[0] if result.get("master_recommendation") else None,
            result.get("current_rate"),
            result.get("master_warning"),
        ))
        conn.commit()
        conn.close()

    def get_latest_analysis(self, pair: Optional[str] = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if pair:
            cursor.execute("SELECT * FROM analysis_results WHERE pair = ? ORDER BY timestamp DESC LIMIT 10", (pair,))
        else:
            cursor.execute("SELECT * FROM analysis_results ORDER BY timestamp DESC LIMIT 20")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def save_feed_health(self, health: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feed_health (timestamp, feeds_active, rate_source, vol_source, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            1 if health.get("feeds_active") else 0,
            health.get("rate_feed_source"),
            health.get("vol_feed_source"),
            health.get("warning"),
        ))
        conn.commit()
        conn.close()
