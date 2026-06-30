"""
DeforestNet - Alert API Routes
CRUD operations for deforestation alerts.
"""

from flask import Blueprint, jsonify, request, current_app
from src.utils.logger import get_logger

logger = get_logger("api.alerts")

alerts_bp = Blueprint("alerts", __name__)


@alerts_bp.route("", methods=["GET"])
def get_alerts():
    """
    Get all alerts with optional filters.

    Query params:
        status: Filter by status (pending, sent, acknowledged, resolved)
        region: Filter by region
        limit: Max results (default 100)
        offset: Pagination offset (default 0)
    """
    db = current_app.config["ALERT_DB"]

    status = request.args.get("status")
    region = request.args.get("region")
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)

    if region:
        alerts = db.get_alerts_by_region(region)
    else:
        alerts = db.get_all_alerts(status=status, limit=limit, offset=offset)

    return jsonify({
        "alerts": [a.to_dict() for a in alerts],
        "count": len(alerts),
        "limit": limit,
        "offset": offset
    })


@alerts_bp.route("/<alert_id>", methods=["GET"])
def get_alert(alert_id):
    """Get a single alert by ID."""
    db = current_app.config["ALERT_DB"]
    alert = db.get_alert(alert_id)

    if alert is None:
        return jsonify({"error": "Alert not found"}), 404

    return jsonify(alert.to_dict())


@alerts_bp.route("/<alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    manager = current_app.config["ALERT_MANAGER"]
    db = current_app.config["ALERT_DB"]

    alert = db.get_alert(alert_id)
    if alert is None:
        return jsonify({"error": "Alert not found"}), 404

    data = request.get_json(silent=True) or {}
    officer_id = data.get("officer_id", "")

    manager.acknowledge_alert(alert_id, officer_id=officer_id)

    updated = db.get_alert(alert_id)
    return jsonify({
        "message": "Alert acknowledged",
        "alert": updated.to_dict()
    })


@alerts_bp.route("/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    """Resolve an alert."""
    manager = current_app.config["ALERT_MANAGER"]
    db = current_app.config["ALERT_DB"]

    alert = db.get_alert(alert_id)
    if alert is None:
        return jsonify({"error": "Alert not found"}), 404

    data = request.get_json(silent=True) or {}
    notes = data.get("notes", "")
    officer_id = data.get("officer_id", "")

    manager.resolve_alert(alert_id, notes=notes, officer_id=officer_id)

    updated = db.get_alert(alert_id)
    return jsonify({
        "message": "Alert resolved",
        "alert": updated.to_dict()
    })


@alerts_bp.route("/<alert_id>/status", methods=["PUT"])
def update_alert_status(alert_id):
    """Update alert status."""
    db = current_app.config["ALERT_DB"]

    alert = db.get_alert(alert_id)
    if alert is None:
        return jsonify({"error": "Alert not found"}), 404

    data = request.get_json(silent=True) or {}
    new_status = data.get("status")

    if not new_status:
        return jsonify({"error": "Missing 'status' field"}), 400

    valid_statuses = ["pending", "sent", "acknowledged", "investigating", "resolved", "false_alarm"]
    if new_status not in valid_statuses:
        return jsonify({"error": f"Invalid status. Must be one of: {valid_statuses}"}), 400

    db.update_alert_status(
        alert_id, new_status,
        changed_by=data.get("changed_by", ""),
        notes=data.get("notes", "")
    )

    updated = db.get_alert(alert_id)
    return jsonify({
        "message": f"Status updated to {new_status}",
        "alert": updated.to_dict()
    })


@alerts_bp.route("/active", methods=["GET"])
def get_active_alerts():
    """Get all active (non-resolved) alerts."""
    manager = current_app.config["ALERT_MANAGER"]
    alerts = manager.get_active_alerts()

    return jsonify({
        "alerts": [a.to_dict() for a in alerts],
        "count": len(alerts)
    })


@alerts_bp.route("/pending", methods=["GET"])
def get_pending_alerts():
    """Get all pending (unsent) alerts."""
    manager = current_app.config["ALERT_MANAGER"]
    alerts = manager.get_pending_alerts()

    return jsonify({
        "alerts": [a.to_dict() for a in alerts],
        "count": len(alerts)
    })


@alerts_bp.route("/statistics", methods=["GET"])
def get_alert_statistics():
    """Get alert statistics."""
    db = current_app.config["ALERT_DB"]
    stats = db.get_alert_statistics()
    return jsonify(stats)
