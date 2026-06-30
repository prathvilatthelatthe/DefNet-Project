"""
DeforestNet - Email Notifier (Tier 3)
Send deforestation alerts via Gmail SMTP.
100% FREE - 500 emails/day with Gmail.
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from configs.config import NOTIFICATION_CONFIG

logger = get_logger("email_notifier")


class EmailNotifier:
    """
    Send alerts via Gmail SMTP.

    Uses Python's built-in smtplib - no paid services.
    Gmail free tier: 500 emails/day.

    Setup:
        1. Enable 2-Step Verification on Google Account
        2. Create App Password at https://myaccount.google.com/apppasswords
        3. Set EMAIL_SENDER and EMAIL_PASSWORD environment variables
    """

    def __init__(
        self,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None
    ):
        email_config = NOTIFICATION_CONFIG["email"]

        self.sender_email = sender_email or os.environ.get(
            "EMAIL_SENDER", email_config.get("sender_email", "")
        )
        self.sender_password = sender_password or os.environ.get(
            "EMAIL_PASSWORD", email_config.get("sender_password", "")
        )
        self.smtp_server = smtp_server or email_config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = smtp_port or email_config.get("smtp_port", 587)
        self.timeout = email_config.get("timeout_seconds", 30)

        self.demo_mode = (
            not self.sender_email
            or not self.sender_password
            or self.sender_email == "your_email@gmail.com"
            or self.sender_password == "your_app_password_here"
        )

        if self.demo_mode:
            logger.info("Email notifier in DEMO mode (no credentials)")
        else:
            logger.info(f"Email notifier initialized: {self.sender_email}")

    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return not self.demo_mode

    def _create_message(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str = "",
        attachments: List[str] = None
    ) -> MIMEMultipart:
        """Create an email message with HTML body and optional attachments."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = to_email

        # Plain text fallback
        if text_body:
            msg.attach(MIMEText(text_body, "plain"))

        # HTML body
        msg.attach(MIMEText(html_body, "html"))

        # Attachments
        if attachments:
            # Switch to mixed type for attachments
            mixed_msg = MIMEMultipart("mixed")
            mixed_msg["Subject"] = msg["Subject"]
            mixed_msg["From"] = msg["From"]
            mixed_msg["To"] = msg["To"]
            mixed_msg.attach(msg)

            for file_path in attachments:
                if Path(file_path).exists():
                    self._attach_file(mixed_msg, file_path)

            return mixed_msg

        return msg

    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """Attach a file to the message."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".png", ".jpg", ".jpeg", ".gif"):
            with open(path, "rb") as f:
                img = MIMEImage(f.read(), name=path.name)
                img.add_header("Content-Disposition", "attachment", filename=path.name)
                msg.attach(img)
        else:
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", "attachment", filename=path.name)
                msg.attach(part)

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str = "",
        attachments: List[str] = None
    ) -> Dict:
        """
        Send an email.

        Args:
            to_email: Recipient email address
            subject: Email subject line
            html_body: HTML formatted body
            text_body: Plain text fallback
            attachments: List of file paths to attach

        Returns:
            Dict with 'ok' status and details
        """
        if self.demo_mode:
            logger.info(f"[DEMO] Email to {to_email}: {subject}")
            return {
                "ok": True,
                "demo": True,
                "to": to_email,
                "subject": subject,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            msg = self._create_message(to_email, subject, html_body, text_body, attachments)

            # Create secure SSL/TLS connection
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, to_email, msg.as_string())

            logger.info(f"Email sent to {to_email}: {subject}")
            return {
                "ok": True,
                "to": to_email,
                "subject": subject,
                "timestamp": datetime.utcnow().isoformat()
            }

        except smtplib.SMTPAuthenticationError:
            logger.error("Email authentication failed. Check EMAIL_SENDER and EMAIL_PASSWORD.")
            return {"ok": False, "error": "Authentication failed"}
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Email error: {e}")
            return {"ok": False, "error": str(e)}

    def send_alert(self, alert, to_email: str = "") -> Dict:
        """
        Send a deforestation alert via email.

        Args:
            alert: Alert object from src.alerts.models
            to_email: Recipient email (uses officer's email if empty)

        Returns:
            Dict with send status
        """
        # Determine recipient
        recipient = to_email
        if not recipient and hasattr(alert, 'assigned_officer_id'):
            recipient = ""  # Would look up from database

        if not recipient:
            if self.demo_mode:
                recipient = "demo@example.com"
            else:
                return {"ok": False, "error": "No recipient email"}

        # Subject line
        severity_label = alert.severity.upper()
        subject = f"[{severity_label}] Deforestation Alert - {alert.cause} - {alert.alert_id}"

        # HTML body
        severity_color = {
            "low": "#FFC107", "medium": "#FF9800",
            "high": "#F44336", "critical": "#B71C1C"
        }
        color = severity_color.get(alert.severity, "#FF9800")

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">DEFORESTATION ALERT</h1>
                <p style="margin: 5px 0 0 0; font-size: 18px;">{severity_label} Severity</p>
            </div>

            <div style="padding: 20px; background-color: #f5f5f5;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Alert ID</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.alert_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Cause</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.cause}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Confidence</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.confidence:.0%}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Affected Area</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.affected_area_hectares:.2f} hectares</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Location</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.latitude:.4f}N, {alert.longitude:.4f}E</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Region</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.region or 'Unknown'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Assigned Officer</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.assigned_officer_name or 'Unassigned'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Timestamp</td>
                        <td style="padding: 8px;">{alert.timestamp}</td>
                    </tr>
                </table>
            </div>

            <div style="padding: 15px; background-color: #e8e8e8; text-align: center; font-size: 12px;">
                <p>DeforestNet Alert System - Automated Notification</p>
                <p>Please acknowledge this alert by replying to this email.</p>
            </div>
        </body>
        </html>
        """

        # Plain text fallback
        text_body = alert.get_full_summary()

        # Collect attachments
        attachments = []
        for path_attr in ['prediction_map_path', 'heatmap_path', 'gradcam_path']:
            path = getattr(alert, path_attr, "")
            if path and Path(path).exists():
                attachments.append(path)

        return self.send_email(
            to_email=recipient,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            attachments=attachments if attachments else None
        )

    def send_daily_summary(self, alerts: list, to_email: str) -> Dict:
        """Send a daily summary of alerts."""
        if not alerts:
            subject = "DeforestNet Daily Report - No Alerts"
            html_body = """
            <html><body style="font-family: Arial;">
                <h2>Daily Summary</h2>
                <p>No deforestation alerts detected today.</p>
            </body></html>
            """
        else:
            subject = f"DeforestNet Daily Report - {len(alerts)} Alerts"

            rows = ""
            for alert in alerts[:50]:
                color = {"low": "#FFC107", "medium": "#FF9800",
                         "high": "#F44336", "critical": "#B71C1C"}.get(alert.severity, "#999")
                rows += f"""
                <tr>
                    <td style="padding: 6px; border: 1px solid #ddd;">{alert.alert_id}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{alert.cause}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">
                        <span style="color: {color}; font-weight: bold;">{alert.severity.upper()}</span>
                    </td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{alert.affected_area_hectares:.1f} ha</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{alert.confidence:.0%}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{alert.region or '-'}</td>
                </tr>
                """

            html_body = f"""
            <html><body style="font-family: Arial;">
                <h2>Daily Summary - {len(alerts)} Alerts</h2>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #4CAF50; color: white;">
                        <th style="padding: 8px; border: 1px solid #ddd;">ID</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Cause</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Severity</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Area</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Confidence</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Region</th>
                    </tr>
                    {rows}
                </table>
            </body></html>
            """

        return self.send_email(to_email, subject, html_body)

    def test_connection(self) -> Dict:
        """Test SMTP connection without sending."""
        if self.demo_mode:
            return {"ok": True, "demo": True, "message": "Demo mode - no real connection"}

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self.sender_email, self.sender_password)

            return {"ok": True, "message": "SMTP connection successful"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    print("Testing Email Notifier...")
    notifier = EmailNotifier()

    print(f"  Demo mode: {notifier.demo_mode}")
    print(f"  Configured: {notifier.is_configured}")

    # Test connection
    conn_result = notifier.test_connection()
    print(f"  Connection test: {conn_result}")

    # Test send in demo mode
    result = notifier.send_email(
        "test@example.com",
        "Test Alert",
        "<h1>Test</h1><p>This is a test alert.</p>",
        "Test alert plain text"
    )
    print(f"  Send result: ok={result.get('ok')}, demo={result.get('demo', False)}")

    print("[OK] Email notifier test complete!")
