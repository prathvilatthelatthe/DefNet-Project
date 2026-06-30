"""
DeforestNet - Notification API Routes
Trigger and manage notifications.
"""

from flask import Blueprint, jsonify, request, current_app
from src.utils.logger import get_logger

logger = get_logger("api.notifications")

notifications_bp = Blueprint("notifications", __name__)


@notifications_bp.route("/send/<alert_id>", methods=["POST"])
def send_notification(alert_id):
    """
    Send notification for a specific alert.

    Request JSON (optional):
        tiers: list of tiers to use ["fcm", "telegram", "email"]
        mode: "all" (default) or "failover"
    """
    db = current_app.config["ALERT_DB"]
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]

    alert = db.get_alert(alert_id)
    if alert is None:
        return jsonify({"error": "Alert not found"}), 404

    data = request.get_json(silent=True) or {}
    tiers = data.get("tiers")
    mode = data.get("mode", "all")

    if mode == "failover":
        result = notif_manager.send_alert_with_failover(alert)
    else:
        result = notif_manager.send_alert_notification(alert, tiers=tiers)

    return jsonify({
        "success": result.success,
        "successful_tiers": result.successful_tiers,
        "failed_tiers": result.failed_tiers,
        "details": result.tier_results
    })


@notifications_bp.route("/send-batch", methods=["POST"])
def send_batch_notifications():
    """
    Send notifications for multiple alerts.

    Request JSON:
        alert_ids: list of alert IDs to notify
    """
    db = current_app.config["ALERT_DB"]
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]

    data = request.get_json(silent=True) or {}
    alert_ids = data.get("alert_ids", [])

    if not alert_ids:
        return jsonify({"error": "Missing alert_ids"}), 400

    alerts = []
    for aid in alert_ids:
        alert = db.get_alert(aid)
        if alert:
            alerts.append(alert)

    if not alerts:
        return jsonify({"error": "No valid alerts found"}), 404

    results = notif_manager.send_batch_notifications(alerts)

    return jsonify({
        "total": len(results),
        "succeeded": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": [r.to_dict() for r in results]
    })


@notifications_bp.route("/test", methods=["POST"])
def test_notifications():
    """Test all notification tiers."""
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]
    results = notif_manager.test_all_tiers()

    return jsonify({
        "tiers": {
            tier: {
                "configured": info["configured"],
                "test_ok": info["test"].get("ok", False),
                "demo": info["test"].get("demo", False)
            }
            for tier, info in results.items()
        }
    })


@notifications_bp.route("/status", methods=["GET"])
def notification_status():
    """Get notification system status."""
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]

    return jsonify({
        "status": notif_manager.status,
        "statistics": notif_manager.get_statistics()
    })


@notifications_bp.route("/history", methods=["GET"])
def notification_history():
    """Get notification history."""
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]
    limit = request.args.get("limit", 50, type=int)

    history = notif_manager.get_notification_history(limit=limit)
    return jsonify({
        "history": history,
        "count": len(history)
    })


@notifications_bp.route("/daily-summary", methods=["POST"])
def send_daily_summary():
    """
    Send daily summary email and/or Telegram.

    Request JSON (optional):
        email: recipient email address
        telegram_chat_id: Telegram chat ID
    """
    db = current_app.config["ALERT_DB"]
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]

    data = request.get_json(silent=True) or {}
    email = data.get("email", "")
    chat_id = data.get("telegram_chat_id", "")

    # Get today's alerts
    all_alerts = db.get_all_alerts(limit=100)

    results = notif_manager.send_daily_summary(
        all_alerts,
        recipient_email=email,
        telegram_chat_id=chat_id
    )

    return jsonify({
        "alert_count": len(all_alerts),
        "results": {
            channel: {"ok": r.get("ok", False), "demo": r.get("demo", False)}
            for channel, r in results.items()
        }
    })
