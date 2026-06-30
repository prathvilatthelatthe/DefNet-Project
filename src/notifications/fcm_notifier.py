"""
DeforestNet - Firebase Cloud Messaging Notifier (Tier 1)
Send push notifications via Firebase FCM.
FREE tier: 500,000 messages/month.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from configs.config import NOTIFICATION_CONFIG

logger = get_logger("fcm_notifier")

# Try to import firebase_admin
FIREBASE_AVAILABLE = False
try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    pass


class FCMNotifier:
    """
    Send push notifications via Firebase Cloud Messaging.

    Free tier: 500,000 messages/month (more than enough).

    Setup:
        1. Create project at https://console.firebase.google.com
        2. Go to Project Settings > Service Accounts
        3. Click "Generate new private key"
        4. Save as configs/firebase_credentials.json
        5. Set FIREBASE_ENABLED=true in .env
    """

    def __init__(self, credentials_file: Optional[str] = None):
        firebase_config = NOTIFICATION_CONFIG["firebase"]

        self.credentials_file = credentials_file or firebase_config.get("credentials_file", "")
        self.timeout = firebase_config.get("timeout_seconds", 10)
        self._initialized = False

        # Check if Firebase should be enabled
        firebase_enabled = os.environ.get("FIREBASE_ENABLED", "false").lower() == "true"
        creds_exist = Path(self.credentials_file).exists() if self.credentials_file else False

        if FIREBASE_AVAILABLE and firebase_enabled and creds_exist:
            self._init_firebase()
        else:
            self.demo_mode = True
            reasons = []
            if not FIREBASE_AVAILABLE:
                reasons.append("firebase-admin not installed")
            if not firebase_enabled:
                reasons.append("FIREBASE_ENABLED=false")
            if not creds_exist:
                reasons.append("credentials file not found")
            logger.info(f"FCM notifier in DEMO mode ({', '.join(reasons)})")

    def _init_firebase(self):
        """Initialize Firebase Admin SDK."""
        try:
            # Check if already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.credentials_file)
                firebase_admin.initialize_app(cred)

            self._initialized = True
            self.demo_mode = False
            logger.info("Firebase FCM initialized successfully")
        except Exception as e:
            self._initialized = False
            self.demo_mode = True
            logger.warning(f"Firebase init failed, using demo mode: {e}")

    @property
    def is_configured(self) -> bool:
        """Check if FCM is properly configured."""
        return self._initialized and not self.demo_mode

    def send_notification(
        self,
        token: str,
        title: str,
        body: str,
        data: Dict = None,
        image_url: str = ""
    ) -> Dict:
        """
        Send a push notification to a device.

        Args:
            token: FCM device token
            title: Notification title
            body: Notification body text
            data: Optional data payload
            image_url: Optional image URL

        Returns:
            Dict with send status
        """
        if self.demo_mode:
            logger.info(f"[DEMO] FCM notification: {title} - {body[:50]}...")
            return {
                "ok": True,
                "demo": True,
                "token": token[:20] + "..." if len(token) > 20 else token,
                "title": title,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            notification = messaging.Notification(
                title=title,
                body=body,
                image=image_url if image_url else None
            )

            message = messaging.Message(
                notification=notification,
                data=data or {},
                token=token
            )

            response = messaging.send(message)
            logger.info(f"FCM notification sent: {response}")
            return {
                "ok": True,
                "message_id": response,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"FCM send error: {e}")
            return {"ok": False, "error": str(e)}

    def send_topic_notification(
        self,
        topic: str,
        title: str,
        body: str,
        data: Dict = None
    ) -> Dict:
        """
        Send notification to a topic (all subscribers).

        Args:
            topic: Topic name (e.g., "deforestation_alerts")
            title: Notification title
            body: Notification body
            data: Optional data payload

        Returns:
            Dict with send status
        """
        if self.demo_mode:
            logger.info(f"[DEMO] FCM topic '{topic}': {title}")
            return {"ok": True, "demo": True, "topic": topic, "title": title}

        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body
                ),
                data=data or {},
                topic=topic
            )

            response = messaging.send(message)
            logger.info(f"FCM topic notification sent: {response}")
            return {"ok": True, "message_id": response}

        except Exception as e:
            logger.error(f"FCM topic send error: {e}")
            return {"ok": False, "error": str(e)}

    def send_multicast(
        self,
        tokens: List[str],
        title: str,
        body: str,
        data: Dict = None
    ) -> Dict:
        """
        Send notification to multiple devices.

        Args:
            tokens: List of FCM device tokens
            title: Notification title
            body: Notification body
            data: Optional data payload

        Returns:
            Dict with send status and per-token results
        """
        if self.demo_mode:
            logger.info(f"[DEMO] FCM multicast to {len(tokens)} devices: {title}")
            return {
                "ok": True,
                "demo": True,
                "total": len(tokens),
                "success_count": len(tokens),
                "failure_count": 0
            }

        try:
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=title,
                    body=body
                ),
                data=data or {},
                tokens=tokens
            )

            response = messaging.send_each_for_multicast(message)
            logger.info(
                f"FCM multicast: {response.success_count} sent, "
                f"{response.failure_count} failed"
            )
            return {
                "ok": True,
                "success_count": response.success_count,
                "failure_count": response.failure_count,
                "total": len(tokens)
            }

        except Exception as e:
            logger.error(f"FCM multicast error: {e}")
            return {"ok": False, "error": str(e)}

    def send_alert(self, alert, device_token: str = "") -> Dict:
        """
        Send a deforestation alert as push notification.

        Args:
            alert: Alert object from src.alerts.models
            device_token: FCM device token (uses demo token if empty)

        Returns:
            Dict with send status
        """
        token = device_token or "demo_device_token"

        severity_prefix = {
            "low": "Monitor", "medium": "Investigate",
            "high": "URGENT", "critical": "CRITICAL"
        }
        prefix = severity_prefix.get(alert.severity, "Alert")

        title = f"{prefix}: {alert.cause} Detected"
        body = (
            f"{alert.affected_area_hectares:.1f} hectares at "
            f"{alert.latitude:.3f}N, {alert.longitude:.3f}E. "
            f"Confidence: {alert.confidence:.0%}"
        )

        data = {
            "alert_id": alert.alert_id,
            "cause": alert.cause,
            "severity": alert.severity,
            "latitude": str(alert.latitude),
            "longitude": str(alert.longitude)
        }

        return self.send_notification(token, title, body, data)


if __name__ == "__main__":
    print("Testing FCM Notifier...")
    notifier = FCMNotifier()

    print(f"  Demo mode: {notifier.demo_mode}")
    print(f"  Configured: {notifier.is_configured}")
    print(f"  Firebase available: {FIREBASE_AVAILABLE}")

    # Test in demo mode
    result = notifier.send_notification(
        "test_token",
        "Test Alert",
        "Deforestation detected in Western Ghats"
    )
    print(f"  Send result: ok={result.get('ok')}, demo={result.get('demo', False)}")

    # Test topic
    result = notifier.send_topic_notification(
        "deforestation_alerts",
        "Daily Report",
        "5 new alerts today"
    )
    print(f"  Topic result: ok={result.get('ok')}")

    print("[OK] FCM notifier test complete!")
