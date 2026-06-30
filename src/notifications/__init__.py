"""
DeforestNet - Notifications Package
3-Tier notification system (all FREE).

Tier 1: Firebase FCM (push notifications)
Tier 2: Telegram Bot API (instant messages)
Tier 3: Email via Gmail SMTP (detailed reports)
"""

from .telegram_notifier import TelegramNotifier
from .email_notifier import EmailNotifier
from .fcm_notifier import FCMNotifier
from .notification_manager import (
    NotificationManager,
    NotificationResult,
    NotificationTier,
    create_notification_manager
)

__all__ = [
    # Tier notifiers
    "TelegramNotifier",
    "EmailNotifier",
    "FCMNotifier",
    # Manager
    "NotificationManager",
    "NotificationResult",
    "NotificationTier",
    "create_notification_manager"
]
