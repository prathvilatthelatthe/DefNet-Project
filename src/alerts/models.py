"""
DeforestNet - Alert Data Models
Data structures for deforestation alerts.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels based on affected area."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    PENDING = "pending"           # Created, not yet sent
    SENT = "sent"                 # Notification sent
    ACKNOWLEDGED = "acknowledged" # Officer acknowledged
    INVESTIGATING = "investigating"  # Officer en route
    RESOLVED = "resolved"         # Investigation complete
    FALSE_ALARM = "false_alarm"   # Marked as false positive


class DeforestationCause(Enum):
    """Deforestation cause classes."""
    FOREST = "Forest"
    LOGGING = "Logging"
    MINING = "Mining"
    AGRICULTURE = "Agriculture"
    FIRE = "Fire"
    INFRASTRUCTURE = "Infrastructure"


@dataclass
class Location:
    """Geographic location."""
    latitude: float
    longitude: float
    region: str = ""

    def to_dict(self) -> Dict:
        return {"latitude": self.latitude, "longitude": self.longitude,
                "region": self.region}

    def to_string(self) -> str:
        return f"{self.latitude:.3f}N, {self.longitude:.3f}E"


@dataclass
class Alert:
    """
    Deforestation alert data structure.

    Contains all information about a detected deforestation event.
    """
    # Core identification
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Detection results
    cause: str = "Unknown"
    confidence: float = 0.0
    affected_area_hectares: float = 0.0
    severity: str = "low"

    # Location
    latitude: float = 0.0
    longitude: float = 0.0
    region: str = ""

    # Class distribution (pixel counts)
    class_distribution: Dict[str, int] = field(default_factory=dict)

    # File references
    satellite_image_path: str = ""
    prediction_map_path: str = ""
    heatmap_path: str = ""
    gradcam_path: str = ""

    # Officer assignment
    assigned_officer_id: str = ""
    assigned_officer_name: str = ""

    # Status tracking
    status: str = "pending"
    notification_tier: str = ""  # tier_1, tier_2, tier_3
    notification_sent_at: str = ""
    acknowledged_at: str = ""
    resolved_at: str = ""

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    @property
    def is_deforestation(self) -> bool:
        """Check if alert represents actual deforestation."""
        return self.cause != "Forest" and self.cause != "Unknown"

    @property
    def location_string(self) -> str:
        """Human-readable location."""
        return f"{self.latitude:.3f}N, {self.longitude:.3f}E"

    def get_sms_text(self) -> str:
        """Generate SMS-compatible alert text (max 160 chars)."""
        text = (f"ALERT: {self.cause} {self.latitude:.3f}N,"
                f"{self.longitude:.3f}E "
                f"{self.affected_area_hectares:.1f}ha "
                f"Conf:{self.confidence:.0%}")
        return text[:160]

    def get_short_summary(self) -> str:
        """Short summary for notifications."""
        return (f"Deforestation Alert: {self.cause} detected at "
                f"{self.location_string}, "
                f"{self.affected_area_hectares:.1f} hectares, "
                f"confidence {self.confidence:.0%}")

    def get_full_summary(self) -> str:
        """Full alert summary."""
        lines = [
            f"{'='*50}",
            f"DEFORESTATION ALERT - {self.alert_id}",
            f"{'='*50}",
            f"Time:       {self.timestamp}",
            f"Cause:      {self.cause}",
            f"Location:   {self.location_string}",
            f"Region:     {self.region}",
            f"Area:       {self.affected_area_hectares:.2f} hectares",
            f"Confidence: {self.confidence:.0%}",
            f"Severity:   {self.severity.upper()}",
            f"Status:     {self.status}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


@dataclass
class Officer:
    """Field officer information."""
    officer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    phone: str = ""
    email: str = ""
    telegram_chat_id: str = ""
    region: str = ""
    is_active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Officer':
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})
