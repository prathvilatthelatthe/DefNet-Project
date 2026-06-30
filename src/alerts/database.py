"""
DeforestNet - SQLite Database Layer
Alert and officer storage with SQLite.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.alerts.models import Alert, Officer
from src.utils.logger import get_logger
from configs.config import DATABASE_CONFIG

logger = get_logger("database")


class AlertDatabase:
    """
    SQLite database for storing alerts and officers.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = DATABASE_CONFIG["path"]

        self.db_path = db_path
        self._is_memory = (db_path == ':memory:')

        # For in-memory DB, keep a persistent connection
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(':memory:')
            self._persistent_conn.row_factory = sqlite3.Row

        self._create_tables()
        logger.info(f"Database initialized: {db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection as context manager."""
        if self._is_memory:
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    affected_area_hectares REAL DEFAULT 0,
                    severity TEXT DEFAULT 'low',
                    latitude REAL DEFAULT 0,
                    longitude REAL DEFAULT 0,
                    region TEXT DEFAULT '',
                    class_distribution TEXT DEFAULT '{}',
                    satellite_image_path TEXT DEFAULT '',
                    prediction_map_path TEXT DEFAULT '',
                    heatmap_path TEXT DEFAULT '',
                    gradcam_path TEXT DEFAULT '',
                    assigned_officer_id TEXT DEFAULT '',
                    assigned_officer_name TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    notification_tier TEXT DEFAULT '',
                    notification_sent_at TEXT DEFAULT '',
                    acknowledged_at TEXT DEFAULT '',
                    resolved_at TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Officers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS officers (
                    officer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT DEFAULT '',
                    email TEXT DEFAULT '',
                    telegram_chat_id TEXT DEFAULT '',
                    region TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Alert history (for tracking status changes)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT,
                    changed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    changed_by TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)

    # ==================== ALERT OPERATIONS ====================

    def insert_alert(self, alert: Alert) -> str:
        """Insert a new alert. Returns alert_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (
                    alert_id, timestamp, cause, confidence,
                    affected_area_hectares, severity,
                    latitude, longitude, region,
                    class_distribution,
                    satellite_image_path, prediction_map_path,
                    heatmap_path, gradcam_path,
                    assigned_officer_id, assigned_officer_name,
                    status, notification_tier,
                    notification_sent_at, acknowledged_at, resolved_at,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.timestamp, alert.cause,
                alert.confidence, alert.affected_area_hectares,
                alert.severity,
                alert.latitude, alert.longitude, alert.region,
                json.dumps(alert.class_distribution),
                alert.satellite_image_path, alert.prediction_map_path,
                alert.heatmap_path, alert.gradcam_path,
                alert.assigned_officer_id, alert.assigned_officer_name,
                alert.status, alert.notification_tier,
                alert.notification_sent_at, alert.acknowledged_at,
                alert.resolved_at, alert.notes
            ))

        logger.info(f"Alert inserted: {alert.alert_id} ({alert.cause})")
        return alert.alert_id

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_alert(row)

    def get_all_alerts(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Alert]:
        """Get all alerts, optionally filtered by status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute(
                    "SELECT * FROM alerts WHERE status = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (status, limit, offset)
                )
            else:
                cursor.execute(
                    "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )

            rows = cursor.fetchall()

        return [self._row_to_alert(row) for row in rows]

    def update_alert_status(
        self,
        alert_id: str,
        new_status: str,
        changed_by: str = "",
        notes: str = ""
    ):
        """Update alert status and log change."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current status
            cursor.execute("SELECT status FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            old_status = row["status"] if row else ""

            # Update status
            now = datetime.utcnow().isoformat()
            if new_status == "acknowledged":
                cursor.execute(
                    "UPDATE alerts SET status = ?, acknowledged_at = ? WHERE alert_id = ?",
                    (new_status, now, alert_id)
                )
            elif new_status == "resolved":
                cursor.execute(
                    "UPDATE alerts SET status = ?, resolved_at = ? WHERE alert_id = ?",
                    (new_status, now, alert_id)
                )
            else:
                cursor.execute(
                    "UPDATE alerts SET status = ? WHERE alert_id = ?",
                    (new_status, alert_id)
                )

            # Log history
            cursor.execute("""
                INSERT INTO alert_history (alert_id, old_status, new_status, changed_by, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (alert_id, old_status, new_status, changed_by, notes))

        logger.info(f"Alert {alert_id}: {old_status} -> {new_status}")

    def update_alert_notification(
        self,
        alert_id: str,
        tier: str,
        sent_at: Optional[str] = None
    ):
        """Update notification information."""
        if sent_at is None:
            sent_at = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE alerts SET notification_tier = ?, notification_sent_at = ?, status = 'sent' WHERE alert_id = ?",
                (tier, sent_at, alert_id)
            )

    def get_alert_count(self, status: Optional[str] = None) -> int:
        """Get count of alerts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = ?", (status,))
            else:
                cursor.execute("SELECT COUNT(*) as cnt FROM alerts")
            return cursor.fetchone()["cnt"]

    def get_alerts_by_region(self, region: str) -> List[Alert]:
        """Get alerts for a specific region."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM alerts WHERE region = ? ORDER BY timestamp DESC",
                (region,)
            )
            rows = cursor.fetchall()
        return [self._row_to_alert(row) for row in rows]

    def get_alert_statistics(self) -> Dict:
        """Get alert statistics summary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) as cnt FROM alerts")
            total = cursor.fetchone()["cnt"]

            # By status
            cursor.execute("SELECT status, COUNT(*) as cnt FROM alerts GROUP BY status")
            by_status = {row["status"]: row["cnt"] for row in cursor.fetchall()}

            # By cause
            cursor.execute("SELECT cause, COUNT(*) as cnt FROM alerts GROUP BY cause")
            by_cause = {row["cause"]: row["cnt"] for row in cursor.fetchall()}

            # By severity
            cursor.execute("SELECT severity, COUNT(*) as cnt FROM alerts GROUP BY severity")
            by_severity = {row["severity"]: row["cnt"] for row in cursor.fetchall()}

            # Average confidence
            cursor.execute("SELECT AVG(confidence) as avg_conf FROM alerts")
            avg_confidence = cursor.fetchone()["avg_conf"] or 0.0

        return {
            "total_alerts": total,
            "by_status": by_status,
            "by_cause": by_cause,
            "by_severity": by_severity,
            "average_confidence": avg_confidence
        }

    def _row_to_alert(self, row) -> Alert:
        """Convert database row to Alert object."""
        data = dict(row)
        if 'class_distribution' in data and isinstance(data['class_distribution'], str):
            data['class_distribution'] = json.loads(data['class_distribution'])
        # Remove extra fields not in Alert
        data.pop('created_at', None)
        return Alert.from_dict(data)

    # ==================== OFFICER OPERATIONS ====================

    def insert_officer(self, officer: Officer) -> str:
        """Insert a new officer. Returns officer_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO officers
                (officer_id, name, phone, email, telegram_chat_id, region, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                officer.officer_id, officer.name, officer.phone,
                officer.email, officer.telegram_chat_id,
                officer.region, 1 if officer.is_active else 0
            ))

        logger.info(f"Officer inserted: {officer.officer_id} ({officer.name})")
        return officer.officer_id

    def get_officer(self, officer_id: str) -> Optional[Officer]:
        """Get officer by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM officers WHERE officer_id = ?", (officer_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        data = dict(row)
        data['is_active'] = bool(data.get('is_active', True))
        data.pop('created_at', None)
        return Officer.from_dict(data)

    def get_officers_by_region(self, region: str) -> List[Officer]:
        """Get active officers in a region."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM officers WHERE region = ? AND is_active = 1",
                (region,)
            )
            rows = cursor.fetchall()

        officers = []
        for row in rows:
            data = dict(row)
            data['is_active'] = bool(data.get('is_active', True))
            data.pop('created_at', None)
            officers.append(Officer.from_dict(data))
        return officers

    def get_all_officers(self, active_only: bool = True) -> List[Officer]:
        """Get all officers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute("SELECT * FROM officers WHERE is_active = 1")
            else:
                cursor.execute("SELECT * FROM officers")
            rows = cursor.fetchall()

        officers = []
        for row in rows:
            data = dict(row)
            data['is_active'] = bool(data.get('is_active', True))
            data.pop('created_at', None)
            officers.append(Officer.from_dict(data))
        return officers
