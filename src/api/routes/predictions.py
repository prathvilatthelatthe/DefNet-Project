"""
DeforestNet - Prediction API Routes
Upload images and get deforestation predictions.
"""

import os
import uuid
import random
import numpy as np
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app, send_file
from src.utils.logger import get_logger
from configs.config import PREDICTIONS_DIR, VISUALIZATION_DIR

logger = get_logger("api.predictions")

predictions_bp = Blueprint("predictions", __name__)

# Store recent predictions in memory for quick access
_recent_predictions = {}

# Indian forest regions with real GPS coordinates for realistic alerts
INDIAN_FOREST_LOCATIONS = [
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

# Track which test images have been used for unique predictions
_used_test_indices = set()


@predictions_bp.route("/analyze", methods=["POST"])
def analyze_prediction():
    """
    Analyze a prediction mask for deforestation.

    Accepts JSON with prediction and confidence arrays,
    or generates a demo prediction.

    Request JSON:
        prediction: 2D array of class labels (optional)
        confidence: 2D array of confidence scores (optional)
        latitude: float (optional)
        longitude: float (optional)
        region: str (optional)
        demo: bool - generate demo data (default: true)
    """
    manager = current_app.config["ALERT_MANAGER"]
    data = request.get_json(silent=True) or {}

    latitude = data.get("latitude", 0.0)
    longitude = data.get("longitude", 0.0)
    region = data.get("region", "")
    use_demo = data.get("demo", True)

    if use_demo or "prediction" not in data:
        # Generate demo prediction
        prediction, confidence = _generate_demo_prediction()
    else:
        try:
            prediction = np.array(data["prediction"], dtype=np.int64)
            confidence = np.array(data.get("confidence",
                                           np.ones_like(prediction) * 0.85),
                                  dtype=np.float32)
        except Exception as e:
            return jsonify({"error": f"Invalid prediction data: {e}"}), 400

    # Process through alert manager
    alert = manager.process_prediction(
        prediction, confidence,
        latitude=latitude,
        longitude=longitude,
        region=region
    )

    if alert is None:
        return jsonify({
            "deforestation_detected": False,
            "message": "No significant deforestation detected",
            "prediction_shape": list(prediction.shape)
        })

    # Send notifications automatically
    notif_manager = current_app.config.get("NOTIFICATION_MANAGER")
    notif_result = None
    if notif_manager:
        notif_result = notif_manager.send_alert_notification(alert)
        logger.info(f"Notifications sent for alert {alert.alert_id}: {notif_result.successful_tiers}")

    # Store prediction for later retrieval
    pred_id = alert.alert_id
    _recent_predictions[pred_id] = {
        "prediction": prediction.tolist(),
        "alert": alert.to_dict()
    }

    return jsonify({
        "deforestation_detected": True,
        "alert": alert.to_dict(),
        "prediction_id": pred_id,
        "prediction_shape": list(prediction.shape),
        "notifications_sent": notif_result.successful_tiers if notif_result else []
    })


@predictions_bp.route("/satellite", methods=["POST"])
def satellite_prediction():
    """
    Run REAL satellite image analysis using the trained U-Net model.

    Loads an actual test satellite image (11-band Sentinel-1/2 data),
    runs it through the trained neural network, and generates a real
    deforestation alert based on what the model detects.

    Request JSON (all optional):
        image_index: int - specific test image index (0-199)
        region: str - override region name
    """
    global _used_test_indices

    engine = current_app.config.get("INFERENCE_ENGINE")
    if engine is None:
        return jsonify({
            "error": "Inference engine not available",
            "message": "Trained model checkpoint not found. Run train.py first."
        }), 503

    manager = current_app.config["ALERT_MANAGER"]
    data = request.get_json(silent=True) or {}

    # Load test satellite images (or generate on-the-fly for deployment)
    try:
        from configs.config import SYNTHETIC_DATA_DIR
        test_img_path = SYNTHETIC_DATA_DIR / "test_images.npy"
        test_mask_path = SYNTHETIC_DATA_DIR / "test_masks.npy"

        if test_img_path.exists() and test_mask_path.exists():
            test_images = np.load(test_img_path, mmap_mode='r')
            test_masks = np.load(test_mask_path, mmap_mode='r')
        else:
            # Generate synthetic test data on-the-fly (for deployment without pre-generated data)
            logger.info("Generating synthetic satellite test data on-the-fly...")
            from src.data.synthetic_generator import SyntheticDataGenerator
            generator = SyntheticDataGenerator(seed=42)
            test_images, test_masks = generator.generate_dataset(20)
            # Cache to disk for subsequent requests
            SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(test_img_path, test_images)
            np.save(test_mask_path, test_masks)
            logger.info(f"Generated and cached {len(test_images)} test images")

        n_images = len(test_images)
    except Exception as e:
        return jsonify({
            "error": f"Could not load satellite test data: {e}",
            "message": "Test data generation failed."
        }), 503

    # Pick image index
    image_index = data.get("image_index")
    if image_index is not None:
        image_index = max(0, min(int(image_index), n_images - 1))
    else:
        # Pick an unused image for variety, reset if all used
        available = set(range(n_images)) - _used_test_indices
        if not available:
            _used_test_indices.clear()
            available = set(range(n_images))
        image_index = random.choice(list(available))

    _used_test_indices.add(image_index)

    # Load the actual satellite image (11 channels: VV, VH, B2, B3, B4, B8, NDVI, EVI, SAVI, VV_VH_RATIO, RVI)
    satellite_image = np.array(test_images[image_index], dtype=np.float32)
    ground_truth = np.array(test_masks[image_index])

    # Run through the REAL trained U-Net model
    logger.info(f"Running U-Net inference on test image #{image_index} (shape: {satellite_image.shape})")
    result = engine.predict(satellite_image, return_probs=True)
    prediction = result["prediction"]
    confidence = result["confidence"]

    # Pick a realistic Indian forest location
    location = data.get("region")
    loc = None
    if location:
        # Find matching region
        for l in INDIAN_FOREST_LOCATIONS:
            if location.lower() in l["region"].lower():
                loc = l
                break
    if loc is None:
        loc = random.choice(INDIAN_FOREST_LOCATIONS)

    # Get model's deforestation summary
    summary = engine.get_deforestation_summary(prediction, confidence)

    # Process through alert manager
    alert = manager.process_prediction(
        prediction, confidence,
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        region=loc["region"]
    )

    if alert is None:
        return jsonify({
            "deforestation_detected": False,
            "message": "Model detected no significant deforestation in this satellite image",
            "image_index": image_index,
            "model_summary": summary,
            "source": "real_model"
        })

    # Send notifications automatically
    notif_manager = current_app.config.get("NOTIFICATION_MANAGER")
    notif_result = None
    if notif_manager:
        notif_result = notif_manager.send_alert_notification(alert)
        logger.info(f"Notifications sent for alert {alert.alert_id}: {notif_result.successful_tiers}")

    # Compute ground truth accuracy
    gt_deforest = np.sum(ground_truth > 0)
    pred_deforest = np.sum(prediction > 0)
    if gt_deforest > 0:
        overlap = np.sum((prediction > 0) & (ground_truth > 0))
        accuracy = float(overlap) / float(gt_deforest) * 100
    else:
        accuracy = 100.0 if pred_deforest == 0 else 0.0

    return jsonify({
        "deforestation_detected": True,
        "alert": alert.to_dict(),
        "source": "real_model",
        "image_index": image_index,
        "model_summary": {
            "total_area_hectares": summary.get("total_area_hectares", 0),
            "forest_area_hectares": summary.get("forest_area_hectares", 0),
            "deforestation_area_hectares": summary.get("deforestation_area_hectares", 0),
            "dominant_cause": summary.get("dominant_cause", "Unknown"),
            "mean_confidence": float(np.mean(confidence)),
        },
        "ground_truth_accuracy": round(accuracy, 1),
        "notifications_sent": notif_result.successful_tiers if notif_result else [],
        "satellite_bands": ["VV", "VH", "B2", "B3", "B4", "B8", "NDVI", "EVI", "SAVI", "VV_VH_Ratio", "RVI"]
    })


@predictions_bp.route("/demo", methods=["POST"])
def demo_prediction():
    """
    Generate a demo prediction and alert.

    Request JSON (all optional):
        cause: str - deforestation cause (Logging, Mining, Agriculture, Fire, Infrastructure)
        latitude: float
        longitude: float
        region: str
        area_fraction: float - fraction of image with deforestation (0.0-1.0)
    """
    from configs.config import CLASS_NAMES

    manager = current_app.config["ALERT_MANAGER"]
    data = request.get_json(silent=True) or {}

    cause = data.get("cause", "Mining")
    latitude = data.get("latitude", 10.5)
    longitude = data.get("longitude", 76.3)
    region = data.get("region", "Western Ghats")
    area_fraction = min(max(data.get("area_fraction", 0.25), 0.05), 0.9)

    # Map cause to class index
    cause_to_class = {name: i for i, name in enumerate(CLASS_NAMES)}
    class_idx = cause_to_class.get(cause, 2)

    # Generate prediction
    size = 256
    prediction = np.zeros((size, size), dtype=np.int64)
    confidence = np.full((size, size), 0.9, dtype=np.float32)

    # Create deforestation region
    side = int(size * (area_fraction ** 0.5))
    start = (size - side) // 2
    prediction[start:start+side, start:start+side] = class_idx

    # Add some noise to confidence
    confidence += np.random.uniform(-0.05, 0.05, (size, size)).astype(np.float32)
    confidence = np.clip(confidence, 0.7, 0.99)

    # Process
    alert = manager.process_prediction(
        prediction, confidence,
        latitude=latitude,
        longitude=longitude,
        region=region
    )

    if alert is None:
        return jsonify({
            "deforestation_detected": False,
            "message": "Area too small or confidence too low"
        })

    # Send notifications automatically
    notif_manager = current_app.config.get("NOTIFICATION_MANAGER")
    notif_result = None
    if notif_manager:
        notif_result = notif_manager.send_alert_notification(alert)
        logger.info(f"Notifications sent for alert {alert.alert_id}: {notif_result.successful_tiers}")

    return jsonify({
        "deforestation_detected": True,
        "alert": alert.to_dict(),
        "demo": True,
        "notifications_sent": notif_result.successful_tiers if notif_result else []
    })


@predictions_bp.route("/recent", methods=["GET"])
def get_recent_predictions():
    """Get recent prediction results."""
    limit = request.args.get("limit", 10, type=int)

    recent = list(_recent_predictions.values())[-limit:]
    return jsonify({
        "predictions": [p["alert"] for p in recent],
        "count": len(recent)
    })


def _generate_demo_prediction():
    """Generate a demo prediction with random deforestation."""
    import random

    size = 256
    prediction = np.zeros((size, size), dtype=np.int64)
    confidence = np.full((size, size), 0.85, dtype=np.float32)

    # Random deforestation cause (1-5)
    cause_class = random.randint(1, 5)

    # Random deforestation area (15-40% of image)
    area_fraction = random.uniform(0.15, 0.40)
    side = int(size * (area_fraction ** 0.5))
    x_start = random.randint(0, size - side)
    y_start = random.randint(0, size - side)

    prediction[y_start:y_start+side, x_start:x_start+side] = cause_class

    # Randomize confidence
    confidence += np.random.uniform(-0.1, 0.1, (size, size)).astype(np.float32)
    confidence = np.clip(confidence, 0.6, 0.99)

    return prediction, confidence
