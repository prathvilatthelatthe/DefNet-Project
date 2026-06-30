"""
DeforestNet - Notification Manager
Orchestrates 3-tier notification system with failover.

Tier 1: Firebase FCM (push notifications) - fastest
Tier 2: Telegram Bot (instant message) - most reliable
Tier 3: Email via Gmail SMTP (detailed report) - most complete

All tiers are 100% FREE.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.notifications.telegram_notifier import TelegramNotifier
from src.notifications.email_notifier import EmailNotifier
from src.notifications.fcm_notifier import FCMNotifier
from src.alerts.models import Alert, Officer
from src.alerts.database import AlertDatabase
from src.utils.logger import get_logger
from configs.config import NOTIFICATION_CONFIG

logger = get_logger("notification_manager")


class NotificationTier(Enum):
    """Notification delivery tiers."""
    TIER_1_FCM = "tier_1_fcm"
    TIER_2_TELEGRAM = "tier_2_telegram"
    TIER_3_EMAIL = "tier_3_email"


class NotificationResult:
    """Result of a notification attempt."""

    def __init__(self):
        self.tier_results: Dict[str, Dict] = {}
        self.success = False
        self.successful_tiers: List[str] = []
        self.failed_tiers: List[str] = []
        self.timestamp = datetime.utcnow().isoformat()

    def add_result(self, tier: str, result: Dict):
        """Add a tier result."""
        self.tier_results[tier] = result
        if result.get("ok"):
            self.successful_tiers.append(tier)
            self.success = True
        else:
            self.failed_tiers.append(tier)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "successful_tiers": self.successful_tiers,
            "failed_tiers": self.failed_tiers,
            "tier_results": self.tier_results,
            "timestamp": self.timestamp
        }

    def __repr__(self):
        return (f"NotificationResult(success={self.success}, "
                f"tiers={self.successful_tiers})")


class NotificationManager:
    """
    Central notification orchestrator.

    Manages all 3 notification tiers with automatic failover:
    1. Try FCM push notification (fastest)
    2. Try Telegram message (most reliable)
    3. Try Email (most detailed)

    If a higher tier fails, automatically falls through to the next.
    All notifications are logged for audit trail.
    """

    def __init__(self, db: Optional[AlertDatabase] = None):
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()
        self.fcm = FCMNotifier()
        self.db = db

        # Track notification history
        self.history: List[Dict] = []

        # Log configuration status
        status = []
        if self.fcm.is_configured:
            status.append("FCM: LIVE")
        else:
            status.append("FCM: DEMO")
        if self.telegram.is_configured:
            status.append("Telegram: LIVE")
        else:
            status.append("Telegram: DEMO")
        if self.email.is_configured:
            status.append("Email: LIVE")
        else:
            status.append("Email: DEMO")

        logger.info(f"NotificationManager initialized [{', '.join(status)}]")

    @property
    def status(self) -> Dict:
        """Get status of all notification tiers."""
        return {
            "fcm": {
                "configured": self.fcm.is_configured,
                "mode": "live" if self.fcm.is_configured else "demo"
            },
            "telegram": {
                "configured": self.telegram.is_configured,
                "mode": "live" if self.telegram.is_configured else "demo"
            },
            "email": {
                "configured": self.email.is_configured,
                "mode": "live" if self.email.is_configured else "demo"
            }
        }

    def send_alert_notification(
        self,
        alert: Alert,
        officer: Optional[Officer] = None,
        tiers: Optional[List[str]] = None
    ) -> NotificationResult:
        """
        Send alert through all configured notification tiers.

        Tries each tier in order (FCM -> Telegram -> Email).
        Continues to all tiers regardless of individual success/failure.

        Args:
            alert: Alert to send
            officer: Target officer (optional, uses alert assignment)
            tiers: Specific tiers to use (None = all available)

        Returns:
            NotificationResult with per-tier results
        """
        result = NotificationResult()
        use_tiers = tiers or ["fcm", "telegram", "email"]

        # Get officer details
        officer_email = ""
        officer_chat_id = ""
        fcm_token = ""

        if officer:
            officer_email = officer.email
            officer_chat_id = officer.telegram_chat_id
        elif hasattr(alert, 'assigned_officer_id') and alert.assigned_officer_id:
            # Try to get from database
            if self.db:
                db_officer = self.db.get_officer(alert.assigned_officer_id)
                if db_officer:
                    officer_email = db_officer.email
                    officer_chat_id = db_officer.telegram_chat_id

        # Tier 1: Firebase Cloud Messaging
        if "fcm" in use_tiers:
            try:
                fcm_result = self.fcm.send_alert(alert, device_token=fcm_token)
                result.add_result("fcm", fcm_result)
                logger.info(f"FCM: {'OK' if fcm_result.get('ok') else 'FAILED'}")
            except Exception as e:
                result.add_result("fcm", {"ok": False, "error": str(e)})
                logger.error(f"FCM error: {e}")

        # Tier 2: Telegram
        if "telegram" in use_tiers:
            try:
                tg_result = self.telegram.send_alert(alert, chat_id=officer_chat_id)
                result.add_result("telegram", tg_result)
                logger.info(f"Telegram: {'OK' if tg_result.get('ok') else 'FAILED'}")
            except Exception as e:
                result.add_result("telegram", {"ok": False, "error": str(e)})
                logger.error(f"Telegram error: {e}")

        # Tier 3: Email
        if "email" in use_tiers:
            try:
                email_result = self.email.send_alert(alert, to_email=officer_email)
                result.add_result("email", email_result)
                logger.info(f"Email: {'OK' if email_result.get('ok') else 'FAILED'}")
            except Exception as e:
                result.add_result("email", {"ok": False, "error": str(e)})
                logger.error(f"Email error: {e}")

        # Update alert in database
        if self.db and result.success:
            tier_name = result.successful_tiers[0] if result.successful_tiers else "none"
            self.db.update_alert_notification(
                alert.alert_id,
                tier=tier_name,
                sent_at=result.timestamp
            )

        # Log to history
        history_entry = {
            "alert_id": alert.alert_id,
            "result": result.to_dict(),
            "timestamp": result.timestamp
        }
        self.history.append(history_entry)

        logger.info(
            f"Alert {alert.alert_id} notification: "
            f"{len(result.successful_tiers)} succeeded, "
            f"{len(result.failed_tiers)} failed"
        )

        return result

    def send_alert_with_failover(
        self,
        alert: Alert,
        officer: Optional[Officer] = None
    ) -> NotificationResult:
        """
        Send alert with failover - try each tier until one succeeds.

        Unlike send_alert_notification which tries all tiers,
        this stops after the first successful delivery.

        Args:
            alert: Alert to send
            officer: Target officer

        Returns:
            NotificationResult
        """
        result = NotificationResult()
        tier_order = [
            ("fcm", self._try_fcm),
            ("telegram", self._try_telegram),
            ("email", self._try_email)
        ]

        officer_email = ""
        officer_chat_id = ""

        if officer:
            officer_email = officer.email
            officer_chat_id = officer.telegram_chat_id

        for tier_name, send_func in tier_order:
            try:
                tier_result = send_func(alert, officer_email=officer_email,
                                        chat_id=officer_chat_id)
                result.add_result(tier_name, tier_result)

                if tier_result.get("ok"):
                    logger.info(f"Alert {alert.alert_id} sent via {tier_name}")
                    break
                else:
                    logger.warning(f"{tier_name} failed, trying next tier...")
            except Exception as e:
                result.add_result(tier_name, {"ok": False, "error": str(e)})
                logger.warning(f"{tier_name} error: {e}, trying next tier...")

        # Update DB
        if self.db and result.success:
            self.db.update_alert_notification(
                alert.alert_id,
                tier=result.successful_tiers[0],
                sent_at=result.timestamp
            )

        self.history.append({
            "alert_id": alert.alert_id,
            "mode": "failover",
            "result": result.to_dict()
        })

        return result

    def _try_fcm(self, alert: Alert, **kwargs) -> Dict:
        """Try sending via FCM."""
        return self.fcm.send_alert(alert)

    def _try_telegram(self, alert: Alert, chat_id: str = "", **kwargs) -> Dict:
        """Try sending via Telegram."""
        return self.telegram.send_alert(alert, chat_id=chat_id)

    def _try_email(self, alert: Alert, officer_email: str = "", **kwargs) -> Dict:
        """Try sending via Email."""
        return self.email.send_alert(alert, to_email=officer_email)

    def send_batch_notifications(
        self,
        alerts: List[Alert],
        officers: Optional[List[Officer]] = None
    ) -> List[NotificationResult]:
        """
        Send notifications for multiple alerts.

        Args:
            alerts: List of alerts to send
            officers: Corresponding officers (optional)

        Returns:
            List of NotificationResult objects
        """
        results = []

        for i, alert in enumerate(alerts):
            officer = officers[i] if officers and i < len(officers) else None
            result = self.send_alert_notification(alert, officer)
            results.append(result)

        total = len(results)
        success = sum(1 for r in results if r.success)
        logger.info(f"Batch notification: {success}/{total} alerts sent successfully")

        return results

    def send_daily_summary(
        self,
        alerts: List[Alert],
        recipient_email: str = "",
        telegram_chat_id: str = ""
    ) -> Dict:
        """
        Send daily summary via email and/or Telegram.

        Args:
            alerts: List of today's alerts
            recipient_email: Email recipient
            telegram_chat_id: Telegram chat ID

        Returns:
            Dict with results per channel
        """
        results = {}

        # Email summary
        if recipient_email or self.email.demo_mode:
            email_to = recipient_email or "demo@example.com"
            results["email"] = self.email.send_daily_summary(alerts, email_to)

        # Telegram summary
        if telegram_chat_id or self.telegram.demo_mode:
            chat = telegram_chat_id or "DEMO_CHAT"
            results["telegram"] = self.telegram.send_batch_summary(alerts, chat)

        return results

    def get_notification_history(self, limit: int = 50) -> List[Dict]:
        """Get recent notification history."""
        return self.history[-limit:]

    def get_statistics(self) -> Dict:
        """Get notification statistics."""
        total = len(self.history)
        if total == 0:
            return {
                "total_notifications": 0,
                "success_rate": 0.0,
                "by_tier": {},
                "demo_mode_tiers": [
                    t for t in ["fcm", "telegram", "email"]
                    if not getattr(getattr(self, t.replace("fcm", "fcm")), "is_configured", False)
                ]
            }

        success = sum(1 for h in self.history if h.get("result", {}).get("success"))

        tier_counts = {"fcm": 0, "telegram": 0, "email": 0}
        for h in self.history:
            for tier in h.get("result", {}).get("successful_tiers", []):
                if tier in tier_counts:
                    tier_counts[tier] += 1

        return {
            "total_notifications": total,
            "successful": success,
            "failed": total - success,
            "success_rate": success / total if total > 0 else 0.0,
            "by_tier": tier_counts,
            "status": self.status
        }

    def test_all_tiers(self) -> Dict:
        """
        Test all notification tiers.

        Returns:
            Dict with test results per tier
        """
        results = {}

        # Test FCM
        results["fcm"] = {
            "configured": self.fcm.is_configured,
            "test": self.fcm.send_notification(
                "test_token", "Test", "Testing FCM"
            )
        }

        # Test Telegram
        tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "test_chat")
        results["telegram"] = {
            "configured": self.telegram.is_configured,
            "test": self.telegram.send_text(tg_chat_id, "DeforestNet: Telegram notification test successful!")
        }

        # Test Email
        results["email"] = {
            "configured": self.email.is_configured,
            "test": self.email.test_connection()
        }

        return results


def create_notification_manager(db: Optional[AlertDatabase] = None) -> NotificationManager:
    """
    Create a NotificationManager with default configuration.

    Loads environment variables from .env if available.

    Args:
        db: Optional AlertDatabase for status tracking

    Returns:
        Configured NotificationManager
    """
    # Try to load .env
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded .env configuration")
    except ImportError:
        pass

    return NotificationManager(db=db)


if __name__ == "__main__":
    print("Testing Notification Manager...")
    print("=" * 50)

    manager = create_notification_manager()

    # Check status
    print("\nTier Status:")
    for tier, info in manager.status.items():
        mode = "LIVE" if info["configured"] else "DEMO"
        print(f"  {tier}: {mode}")

    # Test all tiers
    print("\nTesting all tiers...")
    test_results = manager.test_all_tiers()
    for tier, result in test_results.items():
        ok = result["test"].get("ok", False)
        demo = result["test"].get("demo", False)
        print(f"  {tier}: ok={ok}, demo={demo}")

    # Test with a demo alert
    from src.alerts.models import Alert
    alert = Alert(
        cause="Mining",
        confidence=0.91,
        affected_area_hectares=3.5,
        severity="high",
        latitude=10.234,
        longitude=76.543,
        region="Western Ghats"
    )

    print(f"\nSending test alert: {alert.alert_id}")
    result = manager.send_alert_notification(alert)
    print(f"  Success: {result.success}")
    print(f"  Successful tiers: {result.successful_tiers}")
    print(f"  Failed tiers: {result.failed_tiers}")

    # Test failover
    print("\nTesting failover mode...")
    result2 = manager.send_alert_with_failover(alert)
    print(f"  Success: {result2.success}")
    print(f"  First successful tier: {result2.successful_tiers[0] if result2.successful_tiers else 'none'}")

    # Statistics
    stats = manager.get_statistics()
    print(f"\nStatistics: {stats['total_notifications']} notifications, "
          f"{stats['success_rate']:.0%} success rate")

    print("\n[OK] Notification Manager test complete!")
