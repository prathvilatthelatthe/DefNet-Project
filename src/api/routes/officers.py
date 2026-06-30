"""
DeforestNet - Officer API Routes
CRUD operations for field officers.
"""

from flask import Blueprint, jsonify, request, current_app
from src.alerts.models import Officer
from src.utils.logger import get_logger

logger = get_logger("api.officers")

officers_bp = Blueprint("officers", __name__)


@officers_bp.route("", methods=["GET"])
def get_officers():
    """Get all officers."""
    db = current_app.config["ALERT_DB"]
    active_only = request.args.get("active_only", "true").lower() == "true"

    officers = db.get_all_officers(active_only=active_only)
    return jsonify({
        "officers": [o.to_dict() for o in officers],
        "count": len(officers)
    })


@officers_bp.route("/<officer_id>", methods=["GET"])
def get_officer(officer_id):
    """Get a single officer by ID."""
    db = current_app.config["ALERT_DB"]
    officer = db.get_officer(officer_id)

    if officer is None:
        return jsonify({"error": "Officer not found"}), 404

    return jsonify(officer.to_dict())


@officers_bp.route("", methods=["POST"])
def create_officer():
    """
    Create a new officer.

    Request JSON:
        name: str (required)
        phone: str
        email: str
        telegram_chat_id: str
        region: str
    """
    data = request.get_json(silent=True) or {}

    if not data.get("name"):
        return jsonify({"error": "Missing required field: name"}), 400

    officer = Officer(
        name=data["name"],
        phone=data.get("phone", ""),
        email=data.get("email", ""),
        telegram_chat_id=data.get("telegram_chat_id", ""),
        region=data.get("region", "")
    )

    db = current_app.config["ALERT_DB"]
    officer_id = db.insert_officer(officer)

    return jsonify({
        "message": "Officer created",
        "officer": officer.to_dict()
    }), 201


@officers_bp.route("/<officer_id>", methods=["PUT"])
def update_officer(officer_id):
    """
    Update an officer's details.

    Request JSON (all optional):
        name, phone, email, telegram_chat_id, region
    """
    db = current_app.config["ALERT_DB"]
    existing = db.get_officer(officer_id)

    if existing is None:
        return jsonify({"error": "Officer not found"}), 404

    data = request.get_json(silent=True) or {}

    # Update fields
    updated = Officer(
        officer_id=officer_id,
        name=data.get("name", existing.name),
        phone=data.get("phone", existing.phone),
        email=data.get("email", existing.email),
        telegram_chat_id=data.get("telegram_chat_id", existing.telegram_chat_id),
        region=data.get("region", existing.region),
        is_active=data.get("is_active", existing.is_active)
    )

    db.insert_officer(updated)  # Uses INSERT OR REPLACE

    return jsonify({
        "message": "Officer updated",
        "officer": updated.to_dict()
    })


@officers_bp.route("/by-region/<region>", methods=["GET"])
def get_officers_by_region(region):
    """Get officers assigned to a specific region."""
    db = current_app.config["ALERT_DB"]
    officers = db.get_officers_by_region(region)

    return jsonify({
        "officers": [o.to_dict() for o in officers],
        "region": region,
        "count": len(officers)
    })


@officers_bp.route("/setup-demo", methods=["POST"])
def setup_demo_officers():
    """Create demo officers for testing."""
    manager = current_app.config["ALERT_MANAGER"]
    officers = manager.setup_demo_officers()

    return jsonify({
        "message": f"Created {len(officers)} demo officers",
        "officers": [o.to_dict() for o in officers]
    }), 201
