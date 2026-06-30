"""
DeforestNet - Auto-Monitoring API Routes
Start/stop/status for automatic satellite monitoring.
"""

from flask import Blueprint, jsonify, request, current_app
from src.utils.logger import get_logger

logger = get_logger("api.monitoring")

monitoring_bp = Blueprint("monitoring", __name__)


@monitoring_bp.route("/start", methods=["POST"])
def start_monitoring():
    """Start automatic satellite monitoring."""
    monitor = current_app.config.get("AUTO_MONITOR")
    if not monitor:
        return jsonify({"error": "Auto-monitor not available"}), 503

    data = request.get_json(silent=True) or {}
    interval = data.get("interval", 120)  # Default 2 minutes

    result = monitor.start(interval_seconds=interval)
    return jsonify(result)


@monitoring_bp.route("/stop", methods=["POST"])
def stop_monitoring():
    """Stop automatic satellite monitoring."""
    monitor = current_app.config.get("AUTO_MONITOR")
    if not monitor:
        return jsonify({"error": "Auto-monitor not available"}), 503

    result = monitor.stop()
    return jsonify(result)


@monitoring_bp.route("/status", methods=["GET"])
def monitoring_status():
    """Get auto-monitoring status and scan history."""
    monitor = current_app.config.get("AUTO_MONITOR")
    if not monitor:
        return jsonify({"running": False, "error": "Auto-monitor not available"})

    return jsonify(monitor.get_status())
