# =============================================================================
# FILE: database.py
# PURPOSE: Create and manage the SQLite database for storing predictions
#          and farmer feedback. This is the "brain" of the data collection.
# HOW TO RUN: python database.py  (to initialize/test the database)
# =============================================================================

import sqlite3
import os
import json
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DB_PATH = "crop_database.db"   # SQLite file - created automatically
RETRAIN_THRESHOLD = 50          # Auto-retrain after this many new feedback records


# =============================================================================
# PART 1: CREATE THE DATABASE AND TABLES
# =============================================================================

def init_db():
    """
    Creates the database file and all tables if they don't exist yet.
    SQLite creates the .db file automatically when you connect to it.

    TABLES:
    ─────────────────────────────────────────────────────────────────────────
    1. predictions_log   → Every prediction the model makes is stored here
    2. user_feedback     → Farmer's response to the prediction (was it right?)
    3. model_versions    → History of all model versions and their accuracy
    4. retrain_log       → Log of when auto-retraining happened
    ─────────────────────────────────────────────────────────────────────────

    You only need to run this ONCE. After that, tables already exist.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── TABLE 1: predictions_log ───────────────────────────────────────────
    # Every time a user requests a crop recommendation, we save the inputs
    # AND the predicted output here. This becomes our training data later.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,

            -- Input features (what the farmer entered)
            N               REAL NOT NULL,
            P               REAL NOT NULL,
            K               REAL NOT NULL,
            temperature     REAL NOT NULL,
            humidity        REAL NOT NULL,
            ph              REAL NOT NULL,
            rainfall        REAL NOT NULL,
            season          TEXT,

            -- Model output
            predicted_crop  TEXT NOT NULL,
            confidence      REAL,           -- e.g. 0.94 means 94% confidence
            model_version   INTEGER,        -- which version of the model predicted this
            top_alternatives TEXT           -- JSON string: ["maize", "wheat", "cotton"]
        )
    """)

    # ── TABLE 2: user_feedback ─────────────────────────────────────────────
    # Farmer tells us if the prediction was correct or not.
    # This is linked to predictions_log via prediction_id (foreign key).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id   INTEGER NOT NULL,   -- links to predictions_log.id
            timestamp       TEXT NOT NULL,

            star_rating     INTEGER,            -- 1 to 5 stars
            outcome         TEXT,               -- 'yes' / 'partial' / 'no'
            correct_crop    TEXT,               -- what crop actually worked (if prediction was wrong)
            comments        TEXT,               -- optional farmer comments

            -- This flag is set to 1 once this feedback is used for retraining
            -- so we don't accidentally use the same data twice
            used_for_training INTEGER DEFAULT 0,

            FOREIGN KEY (prediction_id) REFERENCES predictions_log(id)
        )
    """)

    # ── TABLE 3: model_versions ────────────────────────────────────────────
    # History of every model we've trained. Good for your presentation to
    # show "Model improved from 97.3% to 97.9% after farmer feedback!"
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            version         INTEGER NOT NULL,
            trained_on      TEXT NOT NULL,
            accuracy        REAL NOT NULL,
            total_records   INTEGER,            -- how many rows were used for training
            feedback_records INTEGER,           -- how many feedback rows were included
            notes           TEXT                -- e.g. "initial training" / "after retrain #3"
        )
    """)

    # ── TABLE 4: retrain_log ──────────────────────────────────────────────
    # Log of every time auto-retraining was triggered and whether it succeeded
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_on    TEXT NOT NULL,
            trigger_reason  TEXT,               -- e.g. "50 new feedback records"
            new_feedback_count INTEGER,
            old_accuracy    REAL,
            new_accuracy    REAL,
            model_updated   INTEGER,            -- 1 = updated, 0 = kept old (new model was worse)
            notes           TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {DB_PATH}")
    print(f"   Tables created: predictions_log, user_feedback, model_versions, retrain_log")


# =============================================================================
# PART 2: FUNCTIONS TO INSERT DATA
# =============================================================================

def save_prediction(N, P, K, temperature, humidity, ph, rainfall,
                    predicted_crop, confidence, model_version=1,
                    season=None, top_alternatives=None):
    """
    Called every time the ML model makes a prediction.
    Saves the input + output to predictions_log table.

    Returns: prediction_id (integer) — pass this to save_feedback() later
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alts_json = json.dumps(top_alternatives) if top_alternatives else None

    cursor.execute("""
        INSERT INTO predictions_log
        (timestamp, N, P, K, temperature, humidity, ph, rainfall, season,
         predicted_crop, confidence, model_version, top_alternatives)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, N, P, K, temperature, humidity, ph, rainfall, season,
          predicted_crop, confidence, model_version, alts_json))

    prediction_id = cursor.lastrowid   # get the auto-generated ID
    conn.commit()
    conn.close()

    return prediction_id   # return this so UI can link feedback to prediction


def save_feedback(prediction_id, star_rating, outcome, correct_crop=None, comments=None):
    """
    Called when a farmer submits the feedback form.
    Saves to user_feedback table and checks if retraining should be triggered.

    Returns: (feedback_id, should_retrain: bool)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO user_feedback
        (prediction_id, timestamp, star_rating, outcome, correct_crop, comments)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (prediction_id, timestamp, star_rating, outcome, correct_crop, comments))

    feedback_id = cursor.lastrowid
    conn.commit()

    # Check if we should trigger retraining
    should_retrain = check_retrain_threshold(cursor)

    conn.close()
    return feedback_id, should_retrain


def save_model_version(version, accuracy, total_records, feedback_records, notes=""):
    """
    Called after training to record the model version in the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_versions
        (version, trained_on, accuracy, total_records, feedback_records, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (version, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          accuracy, total_records, feedback_records, notes))

    conn.commit()
    conn.close()


def log_retrain(new_feedback_count, old_accuracy, new_accuracy, model_updated, notes=""):
    """
    Records when auto-retraining happened and what the results were.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO retrain_log
        (triggered_on, trigger_reason, new_feedback_count, old_accuracy, new_accuracy, model_updated, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          f"Threshold of {RETRAIN_THRESHOLD} new feedback records reached",
          new_feedback_count, old_accuracy, new_accuracy, int(model_updated), notes))

    conn.commit()
    conn.close()


# =============================================================================
# PART 3: FUNCTIONS TO READ DATA
# =============================================================================

def check_retrain_threshold(cursor=None):
    """
    Counts how many feedback records have NOT been used for training yet.
    If the count >= RETRAIN_THRESHOLD (50), returns True.
    """
    close_conn = False
    if cursor is None:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        close_conn = True

    cursor.execute("""
        SELECT COUNT(*) FROM user_feedback WHERE used_for_training = 0
    """)
    count = cursor.fetchone()[0]

    if close_conn:
        conn.close()

    return count >= RETRAIN_THRESHOLD


def get_unused_feedback_count():
    """
    Returns how many new feedback records are waiting to be used for training.
    Used by the UI to show the progress bar.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_feedback WHERE used_for_training = 0")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_total_feedback_count():
    """Returns total number of feedback records (all time)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_feedback")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_total_predictions_count():
    """Returns total number of predictions made."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions_log")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_model_history():
    """
    Returns all model versions as a list of dicts.
    Used to show "model improved over time" in the UI.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT version, trained_on, accuracy, total_records, feedback_records, notes
        FROM model_versions ORDER BY version DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "version": r[0],
            "trained_on": r[1],
            "accuracy": r[2],
            "total_records": r[3],
            "feedback_records": r[4],
            "notes": r[5]
        }
        for r in rows
    ]


def mark_feedback_used(feedback_ids):
    """
    After retraining, mark those feedback records as used so they aren't
    counted again in the threshold check.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ','.join(['?' for _ in feedback_ids])
    cursor.execute(f"""
        UPDATE user_feedback SET used_for_training = 1
        WHERE id IN ({placeholders})
    """, feedback_ids)

    conn.commit()
    conn.close()


def get_unused_feedback_ids():
    """Returns IDs of all feedback not yet used for training."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM user_feedback WHERE used_for_training = 0")
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


def get_db_stats():
    """
    Returns a summary dictionary of all database stats.
    Used by the UI dashboard.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions_log")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM user_feedback")
    total_feedback = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM user_feedback WHERE used_for_training = 0")
    pending_feedback = cursor.fetchone()[0]

    cursor.execute("SELECT MAX(version), MAX(accuracy) FROM model_versions")
    row = cursor.fetchone()
    current_version = row[0] or 1
    current_accuracy = row[1] or 0.0

    cursor.execute("SELECT COUNT(*) FROM retrain_log WHERE model_updated = 1")
    retrain_count = cursor.fetchone()[0]

    conn.close()

    return {
        "total_predictions": total_predictions,
        "total_feedback": total_feedback,
        "pending_feedback": pending_feedback,
        "records_to_retrain": max(0, RETRAIN_THRESHOLD - pending_feedback),
        "progress_pct": min(100, (pending_feedback / RETRAIN_THRESHOLD) * 100),
        "current_version": current_version,
        "current_accuracy": current_accuracy,
        "total_retrains": retrain_count
    }


# =============================================================================
# PART 4: TEST / INITIALIZE (run directly)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DATABASE SETUP - AgroSense Crop Recommendation")
    print("="*60)

    # Create all tables
    init_db()

    # Insert a dummy prediction to test
    pred_id = save_prediction(
        N=90, P=42, K=43, temperature=20.8, humidity=82.0,
        ph=6.5, rainfall=202.9,
        predicted_crop='rice', confidence=0.94,
        model_version=1, season='Kharif',
        top_alternatives=['maize', 'jute', 'coconut']
    )
    print(f"\n✅ Test prediction saved with ID: {pred_id}")

    # Insert test feedback
    feedback_id, should_retrain = save_feedback(
        prediction_id=pred_id,
        star_rating=5,
        outcome='yes',
        comments='Great recommendation!'
    )
    print(f"✅ Test feedback saved with ID: {feedback_id}")
    print(f"   Retrain threshold reached: {should_retrain}")

    # Save initial model version
    save_model_version(version=1, accuracy=97.3, total_records=2200,
                       feedback_records=0, notes="Initial training")
    print(f"✅ Model version 1 recorded in DB")

    # Show stats
    stats = get_db_stats()
    print(f"\n📊 Database Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print(f"\n✅ Database is ready! File: {DB_PATH}")
    print(f"   Next step: Run train_model.py to train the ML model")
