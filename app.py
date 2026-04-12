# =============================================================================
# FILE: app.py
# PURPOSE: Flask web server — the BRIDGE between your ML model and the UI
#
# This file does 3 things:
#   1. Loads the trained ML model from disk (models/best_model.pkl)
#   2. Exposes API endpoints the UI calls (e.g. /predict, /feedback)
#   3. Serves the HTML UI file to the browser
#
# HOW TO RUN:
#   pip install flask flask-cors
#   python app.py
#   Then open: http://localhost:5000
# =============================================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import json
import os
import threading
import logging

# Import our custom modules
from database import (
    init_db, save_prediction, save_feedback,
    get_db_stats, get_model_history, get_unused_feedback_ids,
    mark_feedback_used, log_retrain, save_model_version
)

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
CORS(app)   # Allow the HTML UI to call these APIs from the browser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── MODEL FILE PATHS (must match train_model.py) ─────────────────────────────
MODELS_DIR      = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl")
META_PATH       = os.path.join(MODELS_DIR, "model_meta.json")

# ─── GLOBAL MODEL VARIABLES ───────────────────────────────────────────────────
# These are loaded once when the server starts, then reused for every prediction.
# Loading from disk on every request would be too slow.
model   = None
scaler  = None
encoder = None
model_meta = {}


# =============================================================================
# PART 1: MODEL LOADING
# =============================================================================

def load_model():
    """
    Load the trained model, scaler, and encoder from disk.
    Called once when the server starts.
    Also called after retraining to reload the updated model.
    """
    global model, scaler, encoder, model_meta

    if not os.path.exists(BEST_MODEL_PATH):
        logger.error(f"❌ Model not found at {BEST_MODEL_PATH}")
        logger.error("   Please run: python train_model.py FIRST!")
        return False

    logger.info("Loading ML model from disk...")
    with open(BEST_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            model_meta = json.load(f)

    logger.info(f"✅ Model loaded — Version: v{model_meta.get('version', '?')} | Accuracy: {model_meta.get('accuracy', '?')}%")
    return True


# =============================================================================
# PART 2: PREDICTION LOGIC
# =============================================================================

def make_prediction(N, P, K, temperature, humidity, ph, rainfall):
    """
    Takes raw input values from the farmer, preprocesses them the SAME WAY
    as during training, and returns the prediction.

    CRITICAL: The scaler used here must be the SAME scaler fitted during training.
    That's why we save and load scaler.pkl.

    Returns:
        predicted_crop  (str)  e.g. "rice"
        confidence      (float) e.g. 0.94
        top_5           (list)  e.g. [("rice", 0.94), ("maize", 0.03), ...]
    """
    if model is None or scaler is None or encoder is None:
        raise RuntimeError("Model not loaded! Run train_model.py first.")

    # ── Step 1: Create numpy array in correct feature order ───────────────
    # ORDER MUST MATCH: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # ── Step 2: Scale the input (same scaler as training!) ────────────────
    input_scaled = scaler.transform(input_array)

    # ── Step 3: Predict ───────────────────────────────────────────────────
    pred_encoded = model.predict(input_scaled)[0]       # e.g. 15 (index)
    predicted_crop = encoder.inverse_transform([pred_encoded])[0]  # e.g. "rice"

    # ── Step 4: Get probabilities for all 22 crops ───────────────────────
    probabilities = model.predict_proba(input_scaled)[0]   # array of 22 values summing to 1

    # Map class names to probabilities
    all_crops_proba = list(zip(encoder.classes_, probabilities))
    all_crops_proba.sort(key=lambda x: x[1], reverse=True)  # sort highest first

    confidence = float(probabilities[pred_encoded])
    top_5 = [(crop, float(prob)) for crop, prob in all_crops_proba[:5]]

    return predicted_crop, confidence, top_5


# =============================================================================
# PART 3: AUTO-RETRAINING (runs in background thread)
# =============================================================================

def trigger_retraining():
    """
    Runs in a background thread when feedback threshold is reached.
    Uses train_model.retrain_with_feedback() to do the actual work.

    After retraining, reloads the model so new predictions use the new model.
    """
    logger.info("🔄 AUTO-RETRAINING TRIGGERED! Starting in background thread...")

    try:
        # Import here to avoid circular imports
        from train_model import retrain_with_feedback

        # Get feedback IDs before retraining so we can mark them as used
        feedback_ids = get_unused_feedback_ids()
        old_accuracy = model_meta.get('accuracy', 0) / 100

        # Run retraining
        updated, new_accuracy = retrain_with_feedback()

        if updated:
            # Mark feedback as used so it doesn't count again
            mark_feedback_used(feedback_ids)

            # Log the retrain event
            log_retrain(
                new_feedback_count=len(feedback_ids),
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                model_updated=True,
                notes="Auto-retrain via feedback threshold"
            )

            # Save version to DB
            save_model_version(
                version=model_meta.get('version', 1) + 1,
                accuracy=new_accuracy * 100,
                total_records=2200 + len(feedback_ids),
                feedback_records=len(feedback_ids),
                notes=f"Auto-retrain #{model_meta.get('version', 1)}"
            )

            # Reload the new model into memory
            load_model()
            logger.info(f"✅ Retraining complete! New accuracy: {new_accuracy*100:.2f}%")
        else:
            log_retrain(
                new_feedback_count=len(feedback_ids),
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                model_updated=False,
                notes="New model was worse, kept old model"
            )
            logger.info("⚠️ Retraining ran but old model was better. Keeping old model.")

    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# PART 4: API ENDPOINTS
# =============================================================================

# ── Serve the HTML UI ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main HTML file (ui.html)"""
    return send_from_directory('.', 'ui.html')


# ── ENDPOINT 1: /predict ──────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    The UI sends soil data here → model predicts → we return result.

    REQUEST (JSON):
    {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.8, "humidity": 82.0,
        "ph": 6.5, "rainfall": 202.9,
        "season": "Kharif"
    }

    RESPONSE (JSON):
    {
        "success": true,
        "predicted_crop": "rice",
        "confidence": 0.94,
        "top_5": [["rice", 0.94], ["maize", 0.03], ...],
        "prediction_id": 123,
        "model_version": 2
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400

        # Extract values (convert to float for safety)
        N           = float(data['N'])
        P           = float(data['P'])
        K           = float(data['K'])
        temperature = float(data['temperature'])
        humidity    = float(data['humidity'])
        ph          = float(data['ph'])
        rainfall    = float(data['rainfall'])
        season      = data.get('season', 'Unknown')

        # Make prediction
        predicted_crop, confidence, top_5 = make_prediction(
            N, P, K, temperature, humidity, ph, rainfall
        )

        # Save prediction to database
        prediction_id = save_prediction(
            N=N, P=P, K=K, temperature=temperature,
            humidity=humidity, ph=ph, rainfall=rainfall,
            predicted_crop=predicted_crop,
            confidence=confidence,
            model_version=model_meta.get('version', 1),
            season=season,
            top_alternatives=[crop for crop, _ in top_5[1:4]]
        )

        return jsonify({
            "success": True,
            "predicted_crop": predicted_crop,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 1),
            "top_5": [[crop, round(prob * 100, 1)] for crop, prob in top_5],
            "prediction_id": prediction_id,
            "model_version": model_meta.get('version', 1),
            "model_accuracy": model_meta.get('accuracy', 0)
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── ENDPOINT 2: /feedback ─────────────────────────────────────────────────────
@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Called when farmer submits the feedback form.

    REQUEST (JSON):
    {
        "prediction_id": 123,
        "star_rating": 5,
        "outcome": "yes",          // or "partial" or "no"
        "correct_crop": "wheat",   // only if outcome = "no"
        "comments": "Great!"
    }

    RESPONSE (JSON):
    {
        "success": true,
        "feedback_id": 45,
        "retraining_triggered": false,
        "pending_count": 12,
        "records_to_retrain": 38
    }
    """
    try:
        data = request.get_json()

        prediction_id = int(data['prediction_id'])
        star_rating   = int(data.get('star_rating', 3))
        outcome       = data.get('outcome', 'yes')
        correct_crop  = data.get('correct_crop', None)
        comments      = data.get('comments', None)

        # Save feedback to DB
        feedback_id, should_retrain = save_feedback(
            prediction_id=prediction_id,
            star_rating=star_rating,
            outcome=outcome,
            correct_crop=correct_crop,
            comments=comments
        )

        # Trigger retraining in background if threshold reached
        if should_retrain:
            logger.info("🚀 Threshold reached! Triggering background retraining...")
            retrain_thread = threading.Thread(target=trigger_retraining, daemon=True)
            retrain_thread.start()

        # Get updated stats for the UI progress bar
        stats = get_db_stats()

        return jsonify({
            "success": True,
            "feedback_id": feedback_id,
            "retraining_triggered": should_retrain,
            "pending_count": stats['pending_feedback'],
            "records_to_retrain": stats['records_to_retrain'],
            "progress_pct": stats['progress_pct']
        })

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── ENDPOINT 3: /stats ────────────────────────────────────────────────────────
@app.route('/stats', methods=['GET'])
def stats():
    """
    Returns live database statistics for the UI dashboard.

    RESPONSE:
    {
        "total_predictions": 247,
        "total_feedback": 189,
        "pending_feedback": 12,
        "records_to_retrain": 38,
        "progress_pct": 24,
        "current_version": 3,
        "current_accuracy": 97.9,
        "total_retrains": 2
    }
    """
    try:
        db_stats = get_db_stats()
        db_stats['model_version'] = model_meta.get('version', 1)
        db_stats['model_accuracy'] = model_meta.get('accuracy', 0)
        return jsonify({"success": True, **db_stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── ENDPOINT 4: /model-history ────────────────────────────────────────────────
@app.route('/model-history', methods=['GET'])
def model_history():
    """
    Returns list of all model versions for the dashboard.
    """
    try:
        history = get_model_history()
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── ENDPOINT 5: /health ───────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": model_meta.get('version', None)
    })


# =============================================================================
# PART 5: START THE SERVER
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AGROSENSE - Starting Web Server")
    print("="*60)

    # Step 1: Initialize database (creates tables if not exist)
    print("\n[1/3] Setting up database...")
    init_db()

    # Step 2: Load the ML model
    print("\n[2/3] Loading ML model...")
    model_loaded = load_model()

    # ✅ ONLY train if model NOT found
    if not model_loaded:
        print("\n⚠️ Model not found. Training model in background...")

        def train_and_reload():
            from train_model import main
            main()
            print("\n🔄 Reloading model after training...")
            load_model()

        threading.Thread(target=train_and_reload, daemon=True).start()

    # Step 3: Start the server
    print("\n[3/3] Starting Flask server...")
    print("\n" + "="*60)
    print("  ✅ SERVER RUNNING!")
    print("  Open your browser at: http://localhost:5000")
    print("  Press CTRL+C to stop")
    print("="*60 + "\n")

    import os
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
