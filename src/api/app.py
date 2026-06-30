"""
DeforestNet - Flask Application Factory
Central API server setup with CORS, error handling, and route registration.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

from src.utils.logger import get_logger
from configs.config import API_CONFIG, DATABASE_CONFIG

logger = get_logger("api")


def create_app(testing: bool = False) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        testing: Enable testing mode

    Returns:
        Configured Flask app
    """
    # Get paths
    api_dir = Path(__file__).parent

    app = Flask(
        __name__,
        template_folder=str(api_dir / "templates"),
        static_folder=str(api_dir / "static"),
        static_url_path="/static"
    )

    # Configuration
    app.config["MAX_CONTENT_LENGTH"] = API_CONFIG.get("max_content_length", 50 * 1024 * 1024)
    app.config["TESTING"] = testing

    # Enable CORS for frontend
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ==================== SHARED SERVICES ====================
    # Initialize shared services and attach to app
    _init_services(app)

    # ==================== REGISTER ROUTES ====================
    _register_blueprints(app)

    # ==================== DASHBOARD ROUTES ====================
    @app.route("/", methods=["GET"])
    @app.route("/dashboard", methods=["GET"])
    def serve_dashboard():
        """Serve the main dashboard page."""
        return render_template("dashboard.html")

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)

    # ==================== ERROR HANDLERS ====================
    _register_error_handlers(app)

    # ==================== HEALTH CHECK ====================
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """API health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "DeforestNet API",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "tiers": {
                "fcm": app.config.get("FCM_MODE", "demo"),
                "telegram": app.config.get("TELEGRAM_MODE", "demo"),
                "email": app.config.get("EMAIL_MODE", "demo")
            }
        })

    @app.route("/api", methods=["GET"])
    def api_root():
        """API root - list available endpoints."""
        return jsonify({
            "service": "DeforestNet API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/api/health",
                "alerts": "/api/alerts",
                "officers": "/api/officers",
                "predictions": "/api/predictions",
                "notifications": "/api/notifications",
                "dashboard": "/api/dashboard"
            }
        })

    logger.info("Flask app created successfully")
    return app


def _init_services(app: Flask):
    """Initialize shared services (database, managers, inference engine)."""
    from src.alerts.database import AlertDatabase
    from src.alerts.alert_manager import AlertManager, AlertGenerator
    from src.notifications.notification_manager import create_notification_manager

    # Alert system
    alert_manager = AlertManager()
    app.config["ALERT_MANAGER"] = alert_manager
    app.config["ALERT_DB"] = alert_manager.db
    app.config["ALERT_GENERATOR"] = alert_manager.generator

    # Notification system
    notif_manager = create_notification_manager(db=alert_manager.db)
    app.config["NOTIFICATION_MANAGER"] = notif_manager

    # Inference engine (real U-Net model)
    try:
        from src.inference.engine import InferenceEngine
        from configs.config import CHECKPOINTS_DIR
        checkpoint_path = CHECKPOINTS_DIR / "benchmark" / "best.pt"
        if checkpoint_path.exists():
            inference_engine = InferenceEngine(checkpoint_path=checkpoint_path)
            app.config["INFERENCE_ENGINE"] = inference_engine
            logger.info(f"Inference engine loaded: {checkpoint_path}")
        else:
            # No checkpoint available — use an untrained model so the full
            # pipeline (inference → alert → notification → map) still works
            inference_engine = InferenceEngine()  # fresh untrained model
            app.config["INFERENCE_ENGINE"] = inference_engine
            logger.info("Inference engine loaded with untrained model (demo mode)")
    except Exception as e:
        app.config["INFERENCE_ENGINE"] = None
        logger.warning(f"Could not load inference engine: {e}")

    # Store tier modes for health check
    status = notif_manager.status
    app.config["FCM_MODE"] = status["fcm"]["mode"]
    app.config["TELEGRAM_MODE"] = status["telegram"]["mode"]
    app.config["EMAIL_MODE"] = status["email"]["mode"]

    # Auto-monitoring scheduler
    from src.api.auto_monitor import AutoMonitor
    auto_monitor = AutoMonitor(app)
    app.config["AUTO_MONITOR"] = auto_monitor

    logger.info("Services initialized")


def _register_blueprints(app: Flask):
    """Register route blueprints."""
    from src.api.routes.alerts import alerts_bp
    from src.api.routes.officers import officers_bp
    from src.api.routes.predictions import predictions_bp
    from src.api.routes.notifications import notifications_bp
    from src.api.routes.dashboard import dashboard_bp
    from src.api.routes.monitoring import monitoring_bp

    app.register_blueprint(alerts_bp, url_prefix="/api/alerts")
    app.register_blueprint(officers_bp, url_prefix="/api/officers")
    app.register_blueprint(predictions_bp, url_prefix="/api/predictions")
    app.register_blueprint(notifications_bp, url_prefix="/api/notifications")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(monitoring_bp, url_prefix="/api/monitoring")

    logger.info("Blueprints registered")


def _register_error_handlers(app: Flask):
    """Register global error handlers."""

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": "Resource not found"}), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large", "message": "Max upload size is 50MB"}), 413

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Internal error: {e}")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500
