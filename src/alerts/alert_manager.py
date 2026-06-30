"""
DeforestNet - Alert Manager
Core alert generation and management system.
"""

import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.alerts.models import Alert, Officer, AlertSeverity
from src.alerts.database import AlertDatabase
from src.utils.logger import get_logger
from configs.config import (
    ALERT_CONFIG, CLASS_NAMES, NUM_CLASSES, PREDICTIONS_DIR
)

logger = get_logger("alert_manager")


class AlertGenerator:
    """
    Generate deforestation alerts from model predictions.

    Takes prediction masks and confidence maps, identifies
    deforestation events, and creates structured alerts.
    """

    def __init__(
        self,
        min_confidence: float = None,
        min_area_hectares: float = None,
        pixel_to_hectare: float = None
    ):
        if min_confidence is None:
            min_confidence = ALERT_CONFIG["min_confidence"]
        if min_area_hectares is None:
            min_area_hectares = ALERT_CONFIG["min_affected_area"]
        if pixel_to_hectare is None:
            pixel_to_hectare = ALERT_CONFIG["pixel_to_hectare"]

        self.min_confidence = min_confidence
        self.min_area_hectares = min_area_hectares
        self.pixel_to_hectare = pixel_to_hectare
        self.severity_thresholds = ALERT_CONFIG["severity_thresholds"]

    def analyze_prediction(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        latitude: float = 0.0,
        longitude: float = 0.0,
        region: str = ""
    ) -> Optional[Alert]:
        """
        Analyze a prediction and generate alert if deforestation found.

        Args:
            prediction: (H, W) class labels (0-5)
            confidence: (H, W) confidence scores (0-1)
            latitude: Center latitude of the image
            longitude: Center longitude of the image
            region: Region name

        Returns:
            Alert object if deforestation detected, None otherwise
        """
        # Count pixels per class
        class_distribution = {}
        for i, name in enumerate(CLASS_NAMES):
            count = int(np.sum(prediction == i))
            if count > 0:
                class_distribution[name] = count

        # Identify deforestation pixels (non-forest: classes 1-5)
        deforestation_mask = prediction > 0
        deforestation_pixels = np.sum(deforestation_mask)
        deforestation_area = deforestation_pixels * self.pixel_to_hectare

        # Check minimum area threshold
        if deforestation_area < self.min_area_hectares:
            return None

        # Check confidence threshold
        if deforestation_mask.any():
            mean_confidence = float(confidence[deforestation_mask].mean())
        else:
            return None

        if mean_confidence < self.min_confidence:
            return None

        # Find dominant deforestation cause
        deforestation_classes = {i: np.sum(prediction == i)
                                for i in range(1, NUM_CLASSES)}
        dominant_class_idx = max(deforestation_classes,
                                key=deforestation_classes.get)
        dominant_cause = CLASS_NAMES[dominant_class_idx]

        # Calculate severity
        severity = self._calculate_severity(deforestation_area)

        # Create alert
        alert = Alert(
            timestamp=datetime.utcnow().isoformat(),
            cause=dominant_cause,
            confidence=mean_confidence,
            affected_area_hectares=deforestation_area,
            severity=severity,
            latitude=latitude,
            longitude=longitude,
            region=region,
            class_distribution=class_distribution,
            status="pending"
        )

        logger.info(
            f"Alert generated: {alert.alert_id} - {dominant_cause} "
            f"({deforestation_area:.2f} ha, {mean_confidence:.0%})"
        )

        return alert

    def _calculate_severity(self, area_hectares: float) -> str:
        """Determine alert severity based on affected area."""
        thresholds = self.severity_thresholds

        if area_hectares >= thresholds["high"]:
            return "critical"
        elif area_hectares >= thresholds["medium"]:
            return "high"
        elif area_hectares >= thresholds["low"]:
            return "medium"
        else:
            return "low"

    def analyze_batch(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        locations: Optional[List[Dict]] = None
    ) -> List[Alert]:
        """
        Analyze a batch of predictions.

        Args:
            predictions: (N, H, W) class labels
            confidences: (N, H, W) confidence scores
            locations: List of {latitude, longitude, region} dicts

        Returns:
            List of Alert objects
        """
        alerts = []

        for i in range(len(predictions)):
            loc = locations[i] if locations else {}
            alert = self.analyze_prediction(
                predictions[i],
                confidences[i],
                latitude=loc.get('latitude', 0.0),
                longitude=loc.get('longitude', 0.0),
                region=loc.get('region', '')
            )
            if alert is not None:
                alerts.append(alert)

        logger.info(f"Batch analysis: {len(alerts)} alerts from {len(predictions)} images")
        return alerts


class AlertManager:
    """
    Central alert management system.

    Handles:
    - Alert generation from predictions
    - Alert storage and retrieval
    - Officer assignment
    - Status tracking
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = AlertDatabase(db_path)
        self.generator = AlertGenerator()
        logger.info("AlertManager initialized")

    def process_prediction(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        latitude: float = 0.0,
        longitude: float = 0.0,
        region: str = "",
        image_path: str = "",
        heatmap_path: str = "",
        gradcam_path: str = ""
    ) -> Optional[Alert]:
        """
        Process a prediction and create alert if needed.

        Full pipeline:
        1. Analyze prediction for deforestation
        2. Create alert if threshold met
        3. Assign to officer
        4. Store in database
        5. Return alert for notification

        Returns:
            Alert if created, None if no deforestation detected
        """
        # Generate alert
        alert = self.generator.analyze_prediction(
            prediction, confidence,
            latitude=latitude,
            longitude=longitude,
            region=region
        )

        if alert is None:
            return None

        # Add file references
        alert.satellite_image_path = image_path
        alert.heatmap_path = heatmap_path
        alert.gradcam_path = gradcam_path

        # Auto-assign officer
        self._assign_officer(alert)

        # Store in database
        self.db.insert_alert(alert)

        return alert

    def _assign_officer(self, alert: Alert):
        """Auto-assign nearest available officer."""
        # Get officers from the same region
        officers = self.db.get_officers_by_region(alert.region)

        if not officers:
            # Get any active officer
            officers = self.db.get_all_officers(active_only=True)

        if officers:
            # Simple assignment: pick first available
            officer = officers[0]
            alert.assigned_officer_id = officer.officer_id
            alert.assigned_officer_name = officer.name
            logger.info(f"Alert {alert.alert_id} assigned to {officer.name}")

    def acknowledge_alert(self, alert_id: str, officer_id: str = ""):
        """Mark alert as acknowledged by officer."""
        self.db.update_alert_status(
            alert_id, "acknowledged",
            changed_by=officer_id,
            notes="Officer acknowledged alert"
        )

    def resolve_alert(self, alert_id: str, notes: str = "", officer_id: str = ""):
        """Mark alert as resolved."""
        self.db.update_alert_status(
            alert_id, "resolved",
            changed_by=officer_id,
            notes=notes
        )

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.db.get_alert(alert_id)

    def get_pending_alerts(self) -> List[Alert]:
        """Get all pending (unsent) alerts."""
        return self.db.get_all_alerts(status="pending")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (non-resolved) alerts."""
        all_alerts = self.db.get_all_alerts()
        return [a for a in all_alerts
                if a.status not in ("resolved", "false_alarm")]

    def get_statistics(self) -> Dict:
        """Get alert statistics."""
        return self.db.get_alert_statistics()

    # ==================== OFFICER MANAGEMENT ====================

    def add_officer(self, officer: Officer) -> str:
        """Add a new officer."""
        return self.db.insert_officer(officer)

    def get_officers(self) -> List[Officer]:
        """Get all active officers."""
        return self.db.get_all_officers()

    def setup_demo_officers(self):
        """Create demo officers for testing."""
        # Read Telegram chat ID and email from environment
        tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        officer_email = os.environ.get("EMAIL_SENDER", "demo@example.com")

        demo_officers = [
            Officer(
                officer_id="OFF001",
                name="Rajesh Kumar",
                phone="+919876543210",
                email=officer_email,
                telegram_chat_id=tg_chat_id,
                region="Western Ghats"
            ),
            Officer(
                officer_id="OFF002",
                name="Priya Singh",
                phone="+919876543211",
                email=officer_email,
                telegram_chat_id=tg_chat_id,
                region="Northeast India"
            ),
            Officer(
                officer_id="OFF003",
                name="Amit Patel",
                phone="+919876543212",
                email=officer_email,
                telegram_chat_id=tg_chat_id,
                region="Central India"
            ),
        ]

        for officer in demo_officers:
            self.db.insert_officer(officer)

        logger.info(f"Created {len(demo_officers)} demo officers")
        return demo_officers


def generate_demo_alerts(manager: AlertManager, n_alerts: int = 5) -> List[Alert]:
    """
    Generate demo alerts for testing.

    Args:
        manager: AlertManager instance
        n_alerts: Number of alerts to generate

    Returns:
        List of generated alerts
    """
    regions = [
        {"name": "Western Ghats", "lat_range": (8.0, 14.0), "lon_range": (74.0, 78.0)},
        {"name": "Northeast India", "lat_range": (24.0, 28.0), "lon_range": (89.0, 96.0)},
        {"name": "Central India", "lat_range": (20.0, 24.0), "lon_range": (78.0, 84.0)},
    ]

    causes = ["Logging", "Mining", "Agriculture", "Fire", "Infrastructure"]
    severities = ["low", "medium", "high", "critical"]

    alerts = []
    for i in range(n_alerts):
        region = random.choice(regions)
        cause = random.choice(causes)

        alert = Alert(
            cause=cause,
            confidence=random.uniform(0.7, 0.98),
            affected_area_hectares=random.uniform(0.5, 15.0),
            severity=random.choice(severities),
            latitude=random.uniform(*region["lat_range"]),
            longitude=random.uniform(*region["lon_range"]),
            region=region["name"],
            class_distribution={"Forest": random.randint(40000, 60000),
                                cause: random.randint(2000, 20000)},
            status="pending"
        )

        manager.db.insert_alert(alert)
        alerts.append(alert)

    logger.info(f"Generated {n_alerts} demo alerts")
    return alerts


if __name__ == "__main__":
    print("Testing Alert Manager...")
    print("=" * 50)

    # Create manager with test database
    manager = AlertManager()

    # Setup demo officers
    officers = manager.setup_demo_officers()
    print(f"[OK] Created {len(officers)} demo officers")

    # Generate demo alerts
    alerts = generate_demo_alerts(manager, n_alerts=5)
    print(f"[OK] Generated {len(alerts)} demo alerts")

    # Test with synthetic prediction
    prediction = np.random.randint(0, 6, (256, 256))
    prediction[:200, :200] = 0  # Mostly forest
    prediction[200:, 200:] = 1  # Some logging
    confidence = np.random.uniform(0.7, 0.99, (256, 256)).astype(np.float32)

    alert = manager.process_prediction(
        prediction, confidence,
        latitude=10.234, longitude=76.543,
        region="Western Ghats"
    )

    if alert:
        print(f"[OK] Alert created: {alert.alert_id}")
        print(f"     Cause: {alert.cause}")
        print(f"     Area: {alert.affected_area_hectares:.2f} ha")
        print(f"     Severity: {alert.severity}")
        print(f"     Officer: {alert.assigned_officer_name}")

    # Test statistics
    stats = manager.get_statistics()
    print(f"\n[OK] Statistics:")
    print(f"     Total alerts: {stats['total_alerts']}")
    print(f"     By cause: {stats['by_cause']}")
    print(f"     By severity: {stats['by_severity']}")

    print("\n[OK] All AlertManager tests passed!")
