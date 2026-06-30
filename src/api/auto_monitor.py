"""
DeforestNet - Automatic Satellite Monitoring Scheduler

In production, this replaces manual button clicks.
Automatically picks satellite images, runs them through the U-Net model,
creates alerts, and sends notifications — just like a real deployment.

Real-world equivalent:
  - A cron job downloads new Sentinel-1/2 images from Copernicus API
  - This scheduler processes them automatically every N minutes
  - Officers receive alerts without any human triggering
"""

import threading
import time
import random
import numpy as np
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("auto_monitor")

# Indian forest locations for realistic GPS coordinates
MONITOR_REGIONS = [
    {"latitude": 11.32, "longitude": 76.78, "region": "Western Ghats - Nilgiri Hills"},
    {"latitude": 10.15, "longitude": 77.06, "region": "Western Ghats - Palani Hills"},
    {"latitude": 15.42, "longitude": 74.01, "region": "Western Ghats - Goa Forests"},
    {"latitude": 12.49, "longitude": 75.73, "region": "Western Ghats - Kodagu"},
    {"latitude": 26.73, "longitude": 92.82, "region": "Northeast India - Kaziranga"},
    {"latitude": 25.28, "longitude": 91.72, "region": "Northeast India - Meghalaya"},
    {"latitude": 27.09, "longitude": 93.62, "region": "Northeast India - Arunachal"},
    {"latitude": 23.63, "longitude": 80.58, "region": "Central India - Panna"},
    {"latitude": 22.28, "longitude": 80.61, "region": "Central India - Kanha"},
    {"latitude": 21.39, "longitude": 79.55, "region": "Central India - Nagpur Forests"},
    {"latitude": 23.47, "longitude": 86.15, "region": "Jharkhand - Dalma Wildlife"},
    {"latitude": 20.72, "longitude": 78.06, "region": "Maharashtra - Melghat"},
    {"latitude": 17.50, "longitude": 82.05, "region": "Andhra Pradesh - Araku Valley"},
    {"latitude": 30.37, "longitude": 78.48, "region": "Uttarakhand - Dehradun Forests"},
    {"latitude": 29.38, "longitude": 79.45, "region": "Uttarakhand - Jim Corbett"},
]


class AutoMonitor:
    """
    Automatic satellite monitoring scheduler.

    Simulates real-world behavior:
    - Periodically checks for new satellite images (from test dataset)
    - Runs U-Net model inference
    - Creates alerts if deforestation is detected
    - Sends notifications to assigned officers
    - Logs all scan activity
    """

    def __init__(self, app):
        self.app = app
        self.running = False
        self.thread = None
        self.interval_seconds = 120  # Default: scan every 2 minutes
        self.scan_history = []
        self.total_scans = 0
        self.alerts_generated = 0
        self.notifications_sent = 0
        self._used_indices = set()
        self._test_images = None
        self._test_masks = None
        self._n_images = 0
        logger.info("AutoMonitor initialized")

    def _load_test_data(self):
        """Load satellite test images (one-time)."""
        if self._test_images is not None:
            return True
        try:
            from configs.config import SYNTHETIC_DATA_DIR
            self._test_images = np.load(
                SYNTHETIC_DATA_DIR / "test_images.npy", mmap_mode='r'
            )
            self._test_masks = np.load(
                SYNTHETIC_DATA_DIR / "test_masks.npy", mmap_mode='r'
            )
            self._n_images = len(self._test_images)
            logger.info(f"Loaded {self._n_images} satellite test images")
            return True
        except Exception as e:
            logger.error(f"Could not load satellite data: {e}")
            return False

    def _pick_image(self):
        """Pick next satellite image (cycles through all 200)."""
        available = set(range(self._n_images)) - self._used_indices
        if not available:
            self._used_indices.clear()
            available = set(range(self._n_images))
        idx = random.choice(list(available))
        self._used_indices.add(idx)
        return idx

    def _run_scan(self):
        """Run a single satellite scan cycle."""
        with self.app.app_context():
            engine = self.app.config.get("INFERENCE_ENGINE")
            manager = self.app.config.get("ALERT_MANAGER")
            notif_manager = self.app.config.get("NOTIFICATION_MANAGER")

            if not engine or not manager:
                logger.warning("Scan skipped: engine or manager not available")
                return

            if not self._load_test_data():
                return

            # Pick a satellite image
            idx = self._pick_image()
            image = np.array(self._test_images[idx], dtype=np.float32)
            ground_truth = np.array(self._test_masks[idx])
            location = random.choice(MONITOR_REGIONS)

            scan_time = datetime.utcnow().isoformat()
            logger.info(
                f"AUTO-SCAN #{self.total_scans + 1}: "
                f"Analyzing satellite image #{idx} for {location['region']}"
            )

            # Run U-Net model
            result = engine.predict(image, return_probs=True)
            prediction = result["prediction"]
            confidence = result["confidence"]

            # Process through alert manager
            alert = manager.process_prediction(
                prediction, confidence,
                latitude=location["latitude"],
                longitude=location["longitude"],
                region=location["region"]
            )

            self.total_scans += 1
            scan_record = {
                "scan_number": self.total_scans,
                "timestamp": scan_time,
                "image_index": idx,
                "region": location["region"],
                "deforestation_detected": alert is not None,
            }

            if alert:
                self.alerts_generated += 1
                scan_record["alert_id"] = alert.alert_id
                scan_record["cause"] = alert.cause
                scan_record["severity"] = alert.severity
                scan_record["area_hectares"] = alert.affected_area_hectares
                scan_record["confidence"] = alert.confidence

                # Send notifications
                notif_tiers = []
                if notif_manager:
                    notif_result = notif_manager.send_alert_notification(alert)
                    notif_tiers = notif_result.successful_tiers
                    self.notifications_sent += len(notif_tiers)
                    scan_record["notifications"] = notif_tiers

                logger.info(
                    f"AUTO-SCAN ALERT: {alert.cause} detected in {location['region']} "
                    f"({alert.affected_area_hectares:.1f} ha, {alert.severity}) "
                    f"→ Notified via: {', '.join(notif_tiers) if notif_tiers else 'none'}"
                )
            else:
                logger.info(
                    f"AUTO-SCAN: No deforestation in image #{idx} ({location['region']})"
                )

            # Keep last 50 scan records
            self.scan_history.append(scan_record)
            if len(self.scan_history) > 50:
                self.scan_history = self.scan_history[-50:]

    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        logger.info(
            f"Automatic monitoring STARTED — "
            f"scanning every {self.interval_seconds} seconds"
        )
        while self.running:
            try:
                self._run_scan()
            except Exception as e:
                logger.error(f"Auto-scan error: {e}")

            # Wait for next scan (check every second so we can stop quickly)
            for _ in range(self.interval_seconds):
                if not self.running:
                    break
                time.sleep(1)

        logger.info("Automatic monitoring STOPPED")

    def start(self, interval_seconds=None):
        """Start automatic monitoring."""
        if self.running:
            return {"status": "already_running", "interval": self.interval_seconds}

        if interval_seconds:
            self.interval_seconds = max(30, int(interval_seconds))  # Minimum 30s

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        return {
            "status": "started",
            "interval": self.interval_seconds,
            "message": f"Auto-monitoring started. Scanning every {self.interval_seconds}s"
        }

    def stop(self):
        """Stop automatic monitoring."""
        if not self.running:
            return {"status": "already_stopped"}

        self.running = False
        return {
            "status": "stopped",
            "total_scans": self.total_scans,
            "alerts_generated": self.alerts_generated
        }

    def get_status(self):
        """Get monitoring status."""
        return {
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "total_scans": self.total_scans,
            "alerts_generated": self.alerts_generated,
            "notifications_sent": self.notifications_sent,
            "recent_scans": self.scan_history[-10:],
        }
