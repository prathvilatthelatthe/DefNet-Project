"""
DeforestNet - Telegram Bot Notifier (Tier 2)
Send deforestation alerts via Telegram Bot API.
100% FREE - Unlimited messages.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from configs.config import NOTIFICATION_CONFIG

logger = get_logger("telegram")

# Try to import telegram library
try:
    import requests as http_requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class TelegramNotifier:
    """
    Send alerts via Telegram Bot API.

    Uses direct HTTP requests to the Telegram Bot API.
    No paid services required - completely free.

    Setup:
        1. Create bot via @BotFather on Telegram
        2. Get bot token
        3. Set TELEGRAM_BOT_TOKEN environment variable
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token or os.environ.get(
            "TELEGRAM_BOT_TOKEN",
            NOTIFICATION_CONFIG["telegram"].get("bot_token", "")
        )
        self.timeout = NOTIFICATION_CONFIG["telegram"].get("timeout_seconds", 30)
        self.demo_mode = not bool(self.bot_token) or self.bot_token == "your_telegram_bot_token_here"

        if self.demo_mode:
            logger.info("Telegram notifier in DEMO mode (no bot token)")
        else:
            logger.info("Telegram notifier initialized with real bot token")

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return not self.demo_mode

    def _api_call(self, method: str, data: Dict = None, files: Dict = None) -> Dict:
        """Make a Telegram Bot API call."""
        if not REQUESTS_AVAILABLE:
            return {"ok": False, "description": "requests library not installed"}

        url = self.TELEGRAM_API.format(token=self.bot_token, method=method)
        try:
            if files:
                response = http_requests.post(url, data=data, files=files, timeout=self.timeout)
            else:
                response = http_requests.post(url, json=data, timeout=self.timeout)
            return response.json()
        except Exception as e:
            logger.error(f"Telegram API error: {e}")
            return {"ok": False, "description": str(e)}

    def get_bot_info(self) -> Dict:
        """Get bot information to verify token."""
        if self.demo_mode:
            return {"ok": True, "result": {"username": "demo_bot", "first_name": "Demo Bot"}, "demo": True}
        return self._api_call("getMe")

    def get_updates(self) -> Dict:
        """Get recent messages to find chat IDs."""
        if self.demo_mode:
            return {"ok": True, "result": [], "demo": True}
        return self._api_call("getUpdates")

    def get_chat_id_from_updates(self) -> Optional[str]:
        """Extract chat ID from recent messages."""
        updates = self.get_updates()
        if updates.get("ok") and updates.get("result"):
            for update in updates["result"]:
                if "message" in update:
                    return str(update["message"]["chat"]["id"])
        return None

    def send_text(self, chat_id: str, text: str, parse_mode: str = "HTML") -> Dict:
        """
        Send a text message.

        Args:
            chat_id: Telegram chat ID
            text: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"

        Returns:
            API response dict
        """
        if self.demo_mode:
            # Sanitize for Windows console (emoji can fail on cp1252)
            safe_preview = text[:80].encode("ascii", "replace").decode("ascii")
            logger.info(f"[DEMO] Telegram message to {chat_id}: {safe_preview}...")
            return {
                "ok": True,
                "demo": True,
                "result": {
                    "message_id": 1,
                    "chat": {"id": chat_id},
                    "text": text,
                    "date": int(datetime.utcnow().timestamp())
                }
            }

        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        result = self._api_call("sendMessage", data)

        if result.get("ok"):
            logger.info(f"Telegram message sent to {chat_id}")
        else:
            logger.error(f"Telegram send failed: {result.get('description', 'Unknown error')}")

        return result

    def send_photo(
        self,
        chat_id: str,
        photo_path: str,
        caption: str = ""
    ) -> Dict:
        """
        Send a photo with caption.

        Args:
            chat_id: Telegram chat ID
            photo_path: Path to image file
            caption: Photo caption

        Returns:
            API response dict
        """
        if self.demo_mode:
            logger.info(f"[DEMO] Telegram photo to {chat_id}: {photo_path}")
            return {"ok": True, "demo": True, "result": {"message_id": 2}}

        if not Path(photo_path).exists():
            logger.warning(f"Photo not found: {photo_path}")
            return self.send_text(chat_id, caption or "Alert photo not available")

        data = {"chat_id": chat_id, "caption": caption[:1024], "parse_mode": "HTML"}
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            result = self._api_call("sendPhoto", data=data, files=files)

        return result

    def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: str = ""
    ) -> Dict:
        """Send a document/file."""
        if self.demo_mode:
            logger.info(f"[DEMO] Telegram document to {chat_id}: {file_path}")
            return {"ok": True, "demo": True, "result": {"message_id": 3}}

        if not Path(file_path).exists():
            return {"ok": False, "description": f"File not found: {file_path}"}

        data = {"chat_id": chat_id, "caption": caption[:1024]}
        with open(file_path, "rb") as doc:
            files = {"document": doc}
            result = self._api_call("sendDocument", data=data, files=files)

        return result

    def send_alert(self, alert, chat_id: str = "") -> Dict:
        """
        Send a full deforestation alert.

        Args:
            alert: Alert object from src.alerts.models
            chat_id: Target chat ID (uses officer's chat_id if empty)

        Returns:
            API response dict with send status
        """
        # Determine chat_id
        target_chat = chat_id or getattr(alert, 'assigned_officer_id', '')

        if not target_chat:
            logger.warning(f"No chat_id for alert {alert.alert_id}")
            if self.demo_mode:
                target_chat = "DEMO_CHAT"
            else:
                return {"ok": False, "description": "No chat_id specified"}

        # Format alert message
        severity_emoji = {
            "low": "[!]", "medium": "[!!]", "high": "[!!!]", "critical": "[!!!!]"
        }
        emoji = severity_emoji.get(alert.severity, "[!]")

        message = (
            f"{emoji} <b>DEFORESTATION ALERT</b> {emoji}\n"
            f"\n"
            f"<b>ID:</b> {alert.alert_id}\n"
            f"<b>Cause:</b> {alert.cause}\n"
            f"<b>Severity:</b> {alert.severity.upper()}\n"
            f"<b>Confidence:</b> {alert.confidence:.0%}\n"
            f"<b>Area:</b> {alert.affected_area_hectares:.2f} hectares\n"
            f"<b>Location:</b> {alert.latitude:.4f}N, {alert.longitude:.4f}E\n"
            f"<b>Region:</b> {alert.region or 'Unknown'}\n"
            f"\n"
            f"<b>Officer:</b> {alert.assigned_officer_name or 'Unassigned'}\n"
            f"<b>Time:</b> {alert.timestamp}\n"
            f"\n"
            f"<i>Reply to acknowledge this alert.</i>"
        )

        # Send text message
        result = self.send_text(target_chat, message)

        # Send prediction map if available
        if alert.prediction_map_path and Path(alert.prediction_map_path).exists():
            self.send_photo(target_chat, alert.prediction_map_path, "Prediction Map")

        # Send heatmap if available
        if alert.heatmap_path and Path(alert.heatmap_path).exists():
            self.send_photo(target_chat, alert.heatmap_path, "Confidence Heatmap")

        return result

    def send_batch_summary(self, alerts: list, chat_id: str) -> Dict:
        """Send a summary of multiple alerts."""
        if not alerts:
            return self.send_text(chat_id, "No new alerts.")

        summary = f"<b>Alert Summary - {len(alerts)} new alerts</b>\n\n"
        for i, alert in enumerate(alerts[:10], 1):
            summary += (
                f"{i}. <b>{alert.cause}</b> - "
                f"{alert.affected_area_hectares:.1f}ha - "
                f"{alert.severity.upper()} - "
                f"{alert.confidence:.0%}\n"
            )

        if len(alerts) > 10:
            summary += f"\n... and {len(alerts) - 10} more alerts."

        return self.send_text(chat_id, summary)


if __name__ == "__main__":
    print("Testing Telegram Notifier...")
    notifier = TelegramNotifier()

    print(f"  Demo mode: {notifier.demo_mode}")
    print(f"  Configured: {notifier.is_configured}")

    # Test send in demo mode
    result = notifier.send_text("123456", "Test alert message")
    print(f"  Send result: ok={result.get('ok')}, demo={result.get('demo', False)}")

    bot_info = notifier.get_bot_info()
    print(f"  Bot info: {bot_info}")

    print("[OK] Telegram notifier test complete!")
