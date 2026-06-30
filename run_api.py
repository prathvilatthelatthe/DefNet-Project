"""
DeforestNet - API Server Entry Point
Run the Flask API server with auto-seeded demo data.

Usage:
    python run_api.py
    python run_api.py --port 8000
    python run_api.py --no-seed          # Start without demo data
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before anything else
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"  Loaded .env configuration")
except ImportError:
    pass

from src.api.app import create_app
from configs.config import API_CONFIG


def seed_demo_data(app):
    """Pre-populate the database with demo officers and alerts."""
    with app.app_context():
        manager = app.config["ALERT_MANAGER"]
        db = app.config["ALERT_DB"]

        # Check if already seeded
        existing = db.get_all_officers(active_only=False)
        if len(existing) > 0:
            print("  Demo data: Already seeded (skipping)")
            return

        import numpy as np
        from configs.config import CLASS_NAMES

        # 1. Create demo officers
        officers = manager.setup_demo_officers()
        print(f"  Demo data: Created {len(officers)} officers")

        # 2. Create demo alerts from various scenarios
        scenarios = [
            {"cause": "Mining",        "lat": 10.5,  "lon": 76.3,  "region": "Western Ghats",   "area": 0.30},
            {"cause": "Fire",          "lat": 22.1,  "lon": 80.5,  "region": "Central India",    "area": 0.50},
            {"cause": "Logging",       "lat": 26.5,  "lon": 93.2,  "region": "Northeast India",  "area": 0.20},
            {"cause": "Agriculture",   "lat": 15.3,  "lon": 73.9,  "region": "Western Ghats",    "area": 0.40},
            {"cause": "Infrastructure","lat": 19.0,  "lon": 72.8,  "region": "Central India",    "area": 0.15},
        ]

        cause_map = {name: i for i, name in enumerate(CLASS_NAMES)}
        alert_count = 0

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

            alert = manager.process_prediction(
                pred, conf, latitude=s["lat"], longitude=s["lon"], region=s["region"]
            )
            if alert:
                alert_count += 1

        print(f"  Demo data: Created {alert_count} alerts")


def main():
    parser = argparse.ArgumentParser(description="DeforestNet API Server")
    parser.add_argument("--host", default=API_CONFIG.get("host", "0.0.0.0"),
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=API_CONFIG.get("port", 5000),
                        help="Port to bind to")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (do not use in production)")
    parser.add_argument("--no-seed", action="store_true",
                        help="Skip demo data seeding")

    args = parser.parse_args()
    debug = args.debug

    app = create_app()

    # Auto-seed demo data
    if not args.no_seed:
        seed_demo_data(app)

    print()
    print("=" * 50)
    print("  DeforestNet - Satellite Monitoring System")
    print("=" * 50)
    print(f"  Dashboard:  http://localhost:{args.port}")
    print(f"  API:        http://localhost:{args.port}/api")
    print(f"  Health:     http://localhost:{args.port}/api/health")
    print(f"  Debug:      {debug}")
    print("=" * 50)
    print()

    app.run(host=args.host, port=args.port, debug=debug)


if __name__ == "__main__":
    main()
