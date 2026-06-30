"""
DeforestNet - WSGI Entry Point (Production)
Used by Gunicorn for production deployment on Render.

Usage:
    gunicorn wsgi:app --bind 0.0.0.0:$PORT
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from src.api.app import create_app

# Create the Flask app
app = create_app()

# Auto-seed demo data on startup
def seed_demo_data():
    """Pre-populate the database with demo officers and alerts."""
    with app.app_context():
        manager = app.config["ALERT_MANAGER"]
        db = app.config["ALERT_DB"]

        # Check if already seeded
        existing = db.get_all_officers(active_only=False)
        if len(existing) > 0:
            return

        import numpy as np
        from configs.config import CLASS_NAMES

        # Create demo officers
        officers = manager.setup_demo_officers()

        # Create demo alerts
        scenarios = [
            {"cause": "Mining",        "lat": 10.5,  "lon": 76.3,  "region": "Western Ghats",   "area": 0.30},
            {"cause": "Fire",          "lat": 22.1,  "lon": 80.5,  "region": "Central India",    "area": 0.50},
            {"cause": "Logging",       "lat": 26.5,  "lon": 93.2,  "region": "Northeast India",  "area": 0.20},
            {"cause": "Agriculture",   "lat": 15.3,  "lon": 73.9,  "region": "Western Ghats",    "area": 0.40},
            {"cause": "Infrastructure","lat": 19.0,  "lon": 72.8,  "region": "Central India",    "area": 0.15},
        ]

        cause_map = {name: i for i, name in enumerate(CLASS_NAMES)}

        for s in scenarios:
            cls_idx = cause_map.get(s["cause"], 2)
            size = 256
            side = int(size * (s["area"] ** 0.5))
            start = (size - side) // 2

            pred = np.zeros((size, size), dtype=np.int64)
            conf = np.full((size, size), 0.88, dtype=np.float32)
            pred[start:start+side, start:start+side] = cls_idx
            conf += np.random.uniform(-0.03, 0.03, conf.shape).astype(np.float32)
            conf = np.clip(conf, 0.7, 0.99)

            manager.process_prediction(
                pred, conf, latitude=s["lat"], longitude=s["lon"], region=s["region"]
            )

        print("  Demo data seeded successfully")


# Seed on import
seed_demo_data()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
