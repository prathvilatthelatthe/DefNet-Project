"""
DeforestNet - Dashboard API Routes
Data endpoints for the web dashboard.
"""

from flask import Blueprint, jsonify, request, current_app
from src.utils.logger import get_logger

logger = get_logger("api.dashboard")

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("", methods=["GET"])
def dashboard_overview():
    """
    Get complete dashboard data.

    Returns alert stats, recent alerts, officer info,
    and notification status in a single response.
    """
    db = current_app.config["ALERT_DB"]
    notif_manager = current_app.config["NOTIFICATION_MANAGER"]

    # Alert statistics
    alert_stats = db.get_alert_statistics()

    # Recent alerts (last 10)
    recent_alerts = db.get_all_alerts(limit=10)

    # Active alerts
    manager = current_app.config["ALERT_MANAGER"]
    active_alerts = manager.get_active_alerts()

    # Officers
    officers = db.get_all_officers()

    # Notification status
    notif_status = notif_manager.status
    notif_stats = notif_manager.get_statistics()

    return jsonify({
        "alert_statistics": alert_stats,
        "recent_alerts": [a.to_dict() for a in recent_alerts],
        "active_alerts_count": len(active_alerts),
        "officers": {
            "total": len(officers),
            "list": [o.to_dict() for o in officers]
        },
        "notifications": {
            "status": notif_status,
            "statistics": notif_stats
        }
    })


@dashboard_bp.route("/stats", methods=["GET"])
def dashboard_stats():
    """Get quick statistics for dashboard cards."""
    db = current_app.config["ALERT_DB"]
    manager = current_app.config["ALERT_MANAGER"]

    stats = db.get_alert_statistics()
    active = manager.get_active_alerts()
    pending = manager.get_pending_alerts()
    officers = db.get_all_officers()

    return jsonify({
        "total_alerts": stats.get("total_alerts", 0),
        "active_alerts": len(active),
        "pending_alerts": len(pending),
        "resolved_alerts": stats.get("by_status", {}).get("resolved", 0),
        "total_officers": len(officers),
        "average_confidence": round(stats.get("average_confidence", 0), 3),
        "by_severity": stats.get("by_severity", {}),
        "by_cause": stats.get("by_cause", {}),
        "by_status": stats.get("by_status", {})
    })


@dashboard_bp.route("/alerts-by-cause", methods=["GET"])
def alerts_by_cause():
    """Get alert distribution by deforestation cause."""
    db = current_app.config["ALERT_DB"]
    stats = db.get_alert_statistics()

    return jsonify({
        "by_cause": stats.get("by_cause", {}),
        "total": stats.get("total_alerts", 0)
    })


@dashboard_bp.route("/alerts-by-severity", methods=["GET"])
def alerts_by_severity():
    """Get alert distribution by severity."""
    db = current_app.config["ALERT_DB"]
    stats = db.get_alert_statistics()

    return jsonify({
        "by_severity": stats.get("by_severity", {}),
        "total": stats.get("total_alerts", 0)
    })


@dashboard_bp.route("/alerts-by-status", methods=["GET"])
def alerts_by_status():
    """Get alert distribution by status."""
    db = current_app.config["ALERT_DB"]
    stats = db.get_alert_statistics()

    return jsonify({
        "by_status": stats.get("by_status", {}),
        "total": stats.get("total_alerts", 0)
    })


@dashboard_bp.route("/regions", methods=["GET"])
def get_regions():
    """Get all regions with alert counts."""
    db = current_app.config["ALERT_DB"]
    all_alerts = db.get_all_alerts(limit=1000)

    regions = {}
    for alert in all_alerts:
        r = alert.region or "Unknown"
        if r not in regions:
            regions[r] = {"count": 0, "causes": {}, "severities": {}}
        regions[r]["count"] += 1
        regions[r]["causes"][alert.cause] = regions[r]["causes"].get(alert.cause, 0) + 1
        regions[r]["severities"][alert.severity] = regions[r]["severities"].get(alert.severity, 0) + 1

    return jsonify({
        "regions": regions,
        "total_regions": len(regions)
    })


@dashboard_bp.route("/timeline", methods=["GET"])
def alert_timeline():
    """Get alerts for timeline visualization."""
    db = current_app.config["ALERT_DB"]
    limit = request.args.get("limit", 50, type=int)

    alerts = db.get_all_alerts(limit=limit)

    timeline = []
    for alert in alerts:
        timeline.append({
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp,
            "cause": alert.cause,
            "severity": alert.severity,
            "status": alert.status,
            "area": alert.affected_area_hectares,
            "region": alert.region,
            "latitude": alert.latitude,
            "longitude": alert.longitude
        })

    return jsonify({
        "timeline": timeline,
        "count": len(timeline)
    })
