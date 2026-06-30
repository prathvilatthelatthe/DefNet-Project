"""
DeforestNet - Database Module
SQLite database setup and management for alerts, officers, and system data.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger("database")


class Database:
    """SQLite database manager for DeforestNet."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            from configs.config import DATABASE_CONFIG
            db_path = DATABASE_CONFIG["path"]

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()
        logger.info(f"Database initialized at: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Create all required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    cause TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    area_hectares REAL NOT NULL,
                    severity TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'pending',
                    image_path TEXT,
                    heatmap_path TEXT,
                    assigned_officer_id INTEGER,
                    acknowledged_at DATETIME,
                    resolved_at DATETIME,
                    notes TEXT,
                    FOREIGN KEY (assigned_officer_id) REFERENCES officers(id)
                )
            """)

            # Officers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS officers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    telegram_chat_id TEXT,
                    firebase_token TEXT,
                    region TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME
                )
            """)

            # Notifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL,
                    officer_id INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    sent_at DATETIME,
                    delivered_at DATETIME,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (alert_id) REFERENCES alerts(id),
                    FOREIGN KEY (officer_id) REFERENCES officers(id)
                )
            """)

            # Predictions table (for storing model predictions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    prediction_path TEXT,
                    processing_time_ms REAL,
                    class_distribution TEXT,
                    deforestation_detected BOOLEAN DEFAULT 0,
                    alert_generated BOOLEAN DEFAULT 0
                )
            """)

            # System logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    module TEXT,
                    message TEXT NOT NULL
                )
            """)

            # Model versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    checkpoint_path TEXT NOT NULL,
                    accuracy REAL,
                    f1_score REAL,
                    is_active BOOLEAN DEFAULT 0,
                    notes TEXT
                )
            """)

            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_location ON alerts(latitude, longitude)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_officers_region ON officers(region)")

            logger.info("Database tables created successfully")

    # ==================== ALERT OPERATIONS ====================

    def create_alert(
        self,
        latitude: float,
        longitude: float,
        cause: str,
        confidence: float,
        area_hectares: float,
        severity: str = "medium",
        image_path: str = None,
        heatmap_path: str = None
    ) -> int:
        """Create a new alert and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (latitude, longitude, cause, confidence,
                                   area_hectares, severity, image_path, heatmap_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (latitude, longitude, cause, confidence, area_hectares,
                  severity, image_path, heatmap_path))
            alert_id = cursor.lastrowid
            logger.info(f"Created alert #{alert_id}: {cause} at ({latitude}, {longitude})")
            return alert_id

    def get_alert(self, alert_id: int) -> Optional[Dict]:
        """Get alert by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_alerts(
        self,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get alerts with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM alerts WHERE status = ?
                    ORDER BY timestamp DESC LIMIT ? OFFSET ?
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM alerts
                    ORDER BY timestamp DESC LIMIT ? OFFSET ?
                """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def update_alert_status(self, alert_id: int, status: str, notes: str = None) -> bool:
        """Update alert status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status == "acknowledged":
                cursor.execute("""
                    UPDATE alerts SET status = ?, acknowledged_at = ?, notes = ?
                    WHERE id = ?
                """, (status, datetime.now(), notes, alert_id))
            elif status == "resolved":
                cursor.execute("""
                    UPDATE alerts SET status = ?, resolved_at = ?, notes = ?
                    WHERE id = ?
                """, (status, datetime.now(), notes, alert_id))
            else:
                cursor.execute("""
                    UPDATE alerts SET status = ?, notes = ? WHERE id = ?
                """, (status, notes, alert_id))
            return cursor.rowcount > 0

    # ==================== OFFICER OPERATIONS ====================

    def create_officer(
        self,
        name: str,
        phone: str = None,
        email: str = None,
        telegram_chat_id: str = None,
        region: str = None
    ) -> int:
        """Create a new officer and return their ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO officers (name, phone, email, telegram_chat_id, region)
                VALUES (?, ?, ?, ?, ?)
            """, (name, phone, email, telegram_chat_id, region))
            officer_id = cursor.lastrowid
            logger.info(f"Created officer #{officer_id}: {name}")
            return officer_id

    def get_officer(self, officer_id: int) -> Optional[Dict]:
        """Get officer by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM officers WHERE id = ?", (officer_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_officers(self, region: str = None, active_only: bool = True) -> List[Dict]:
        """Get officers with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM officers WHERE 1=1"
            params = []

            if active_only:
                query += " AND is_active = 1"
            if region:
                query += " AND region = ?"
                params.append(region)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def update_officer(self, officer_id: int, **kwargs) -> bool:
        """Update officer details."""
        if not kwargs:
            return False

        with self.get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            query = f"UPDATE officers SET {set_clause} WHERE id = ?"
            cursor.execute(query, list(kwargs.values()) + [officer_id])
            return cursor.rowcount > 0

    # ==================== NOTIFICATION OPERATIONS ====================

    def log_notification(
        self,
        alert_id: int,
        officer_id: int,
        channel: str,
        status: str = "pending"
    ) -> int:
        """Log a notification attempt."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notifications (alert_id, officer_id, channel, status, sent_at)
                VALUES (?, ?, ?, ?, ?)
            """, (alert_id, officer_id, channel, status, datetime.now()))
            return cursor.lastrowid

    def update_notification(
        self,
        notification_id: int,
        status: str,
        error_message: str = None
    ) -> bool:
        """Update notification status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status == "delivered":
                cursor.execute("""
                    UPDATE notifications SET status = ?, delivered_at = ?
                    WHERE id = ?
                """, (status, datetime.now(), notification_id))
            elif status == "failed":
                cursor.execute("""
                    UPDATE notifications SET status = ?, error_message = ?,
                    retry_count = retry_count + 1 WHERE id = ?
                """, (status, error_message, notification_id))
            else:
                cursor.execute("""
                    UPDATE notifications SET status = ? WHERE id = ?
                """, (status, notification_id))
            return cursor.rowcount > 0

    # ==================== PREDICTION OPERATIONS ====================

    def log_prediction(
        self,
        image_path: str,
        prediction_path: str,
        processing_time_ms: float,
        class_distribution: str,
        deforestation_detected: bool = False
    ) -> int:
        """Log a model prediction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (image_path, prediction_path, processing_time_ms,
                                        class_distribution, deforestation_detected)
                VALUES (?, ?, ?, ?, ?)
            """, (image_path, prediction_path, processing_time_ms,
                  class_distribution, deforestation_detected))
            return cursor.lastrowid

    # ==================== MODEL VERSION OPERATIONS ====================

    def register_model(
        self,
        version: str,
        checkpoint_path: str,
        accuracy: float = None,
        f1_score: float = None,
        notes: str = None
    ) -> int:
        """Register a new model version."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_versions (version, checkpoint_path, accuracy, f1_score, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (version, checkpoint_path, accuracy, f1_score, notes))
            return cursor.lastrowid

    def set_active_model(self, model_id: int) -> bool:
        """Set a model version as active (deactivate others)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_versions SET is_active = 0")
            cursor.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (model_id,))
            return cursor.rowcount > 0

    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions WHERE is_active = 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    # ==================== STATISTICS ====================

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Alert counts
            cursor.execute("SELECT COUNT(*) FROM alerts")
            stats["total_alerts"] = cursor.fetchone()[0]

            cursor.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")
            stats["alerts_by_status"] = dict(cursor.fetchall())

            cursor.execute("SELECT cause, COUNT(*) FROM alerts GROUP BY cause")
            stats["alerts_by_cause"] = dict(cursor.fetchall())

            # Officer count
            cursor.execute("SELECT COUNT(*) FROM officers WHERE is_active = 1")
            stats["active_officers"] = cursor.fetchone()[0]

            # Prediction count
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats["total_predictions"] = cursor.fetchone()[0]

            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM alerts
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            stats["alerts_last_24h"] = cursor.fetchone()[0]

            return stats


# Global database instance
_db_instance = None


def get_database() -> Database:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


if __name__ == "__main__":
    # Test database
    db = get_database()

    # Test creating an officer
    officer_id = db.create_officer(
        name="Test Officer",
        phone="+911234567890",
        email="test@example.com",
        telegram_chat_id="123456789",
        region="Western Ghats"
    )
    print(f"✓ Created test officer: #{officer_id}")

    # Test creating an alert
    alert_id = db.create_alert(
        latitude=10.234,
        longitude=76.543,
        cause="Logging",
        confidence=0.94,
        area_hectares=2.5,
        severity="high"
    )
    print(f"✓ Created test alert: #{alert_id}")

    # Get statistics
    stats = db.get_statistics()
    print(f"✓ Database statistics: {stats}")

    print("\n✓ Database setup complete!")
