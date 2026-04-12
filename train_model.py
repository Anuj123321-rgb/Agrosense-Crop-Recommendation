# =============================================================================
# FILE: train_model.py
# PURPOSE: Train multiple ML models, pick the best one, save it for the UI
# AUTHOR: Anuj - B.Tech Final Year Project
# HOW TO RUN: python train_model.py
# =============================================================================

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Sklearn - preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Sklearn - models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Sklearn - metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# XGBoost and LightGBM (install if missing: pip install xgboost lightgbm)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Skipping. Run: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARNING] LightGBM not installed. Skipping. Run: pip install lightgbm")


# =============================================================================
# STEP 1: CONFIGURATION
# =============================================================================
# Paths - change DATA_PATH if your CSV is in a different location
DATA_PATH      = "Crop_recommendation.csv"   # Original dataset
DB_PATH        = "crop_database.db"           # SQLite database (created by database.py)
MODELS_DIR     = "models"                     # Folder to save trained models
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")   # Final model used by UI
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")        # Scaler used by UI
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl") # Label encoder
META_PATH       = os.path.join(MODELS_DIR, "model_meta.json")   # Metadata (accuracy, version, etc.)

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Features used for training (must match what the UI sends)
FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL   = 'label'

# Random seed for reproducibility
SEED = 42


# =============================================================================
# STEP 2: DATA LOADING & CLEANING
# (This is your existing notebook work, converted to functions)
# =============================================================================

def remove_outlier(col):
    """
    IQR-based outlier clamping (same as your notebook).
    Returns lower and upper bounds.
    """
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


def load_and_clean_data(path):
    """
    Load CSV and apply the same cleaning steps you did in your notebook.
    Returns a cleaned DataFrame.
    """
    print(f"\n{'='*60}")
    print("STEP 1: LOADING AND CLEANING DATA")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null values: {df.isnull().sum().sum()}")

    # Apply IQR outlier clamping to all numeric feature columns
    numeric_cols = ['P', 'K', 'ph', 'rainfall', 'temperature', 'humidity']
    for col in numeric_cols:
        low, high = remove_outlier(df[col])
        df[col] = np.where(df[col] > high, high, df[col])
        df[col] = np.where(df[col] < low, low, df[col])

    print(f"  Outlier clamping done for: {numeric_cols}")
    print(f"  Unique crops: {df[TARGET_COL].nunique()} → {list(df[TARGET_COL].unique())}")
    return df


def load_feedback_from_db():
    """
    Load verified farmer feedback from the database to augment training data.
    Only loads feedback where outcome = 'yes' (farmer confirmed it worked)
    OR where farmer corrected the crop (outcome = 'no' / 'partial' but gave correct crop).
    Returns DataFrame with same columns as main dataset, or None if DB doesn't exist.
    """
    if not os.path.exists(DB_PATH):
        print("  No database found. Training on original data only.")
        return None

    import sqlite3
    conn = sqlite3.connect(DB_PATH)

    # Load feedback that has verified correct crops
    query = """
        SELECT p.N, p.P, p.K, p.temperature, p.humidity, p.ph, p.rainfall,
               CASE
                   WHEN f.outcome = 'yes' THEN p.predicted_crop
                   ELSE f.correct_crop
               END as label
        FROM predictions_log p
        JOIN user_feedback f ON p.id = f.prediction_id
        WHERE f.outcome = 'yes'
           OR (f.correct_crop IS NOT NULL AND f.correct_crop != '')
    """
    try:
        feedback_df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"  Loaded {len(feedback_df)} verified feedback records from DB")
        return feedback_df if len(feedback_df) > 0 else None
    except Exception as e:
        print(f"  DB read error: {e}")
        conn.close()
        return None


# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================

def preprocess(df, fit_scaler=True, scaler=None, encoder=None):
    """
    - Encode the crop label (string → number)
    - Scale the features using StandardScaler

    fit_scaler=True  → fit a NEW scaler (use during training)
    fit_scaler=False → use EXISTING scaler (use during prediction in UI)
    """
    print(f"\n{'='*60}")
    print("STEP 2: PREPROCESSING")
    print(f"{'='*60}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Label Encoding: rice→0, maize→1, etc.
    if encoder is None:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        print(f"  Label Encoding done. Classes: {list(encoder.classes_)}")
    else:
        y_encoded = encoder.transform(y)

    # Feature Scaling: StandardScaler (mean=0, std=1)
    # IMPORTANT: We fit the scaler only on training data, then apply it to test data
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"  Scaler fitted. Feature means: {scaler.mean_.round(2)}")
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y_encoded, scaler, encoder


# =============================================================================
# STEP 4: DEFINE ALL MODELS
# =============================================================================

def get_all_models():
    """
    Returns a dictionary of all models to compare.
    Each model is configured with good default hyperparameters.
    """
    models = {

        # ── Random Forest ──────────────────────────────────────────────────
        # An ensemble of decision trees. Each tree votes and majority wins.
        # n_estimators=200 → 200 trees (more trees = better but slower)
        # max_depth=None → trees can grow fully
        # class_weight='balanced' → handles class imbalance if any
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1   # use all CPU cores
        ),

        # ── Extra Trees ────────────────────────────────────────────────────
        # Like Random Forest but with random split thresholds → faster & sometimes better
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        ),

        # ── Gradient Boosting ──────────────────────────────────────────────
        # Builds trees sequentially. Each tree corrects errors of previous ones.
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=SEED
        ),

        # ── SVM (Support Vector Machine) ───────────────────────────────────
        # Finds the hyperplane that best separates classes in high-dimensional space.
        # probability=True needed so we can get confidence percentages
        "SVM": SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=SEED
        ),

        # ── KNN (K-Nearest Neighbors) ───────────────────────────────────────
        # Predicts based on the 5 most similar data points
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',   # closer neighbors have more influence
            metric='euclidean',
            n_jobs=-1
        ),

        # ── Naive Bayes ────────────────────────────────────────────────────
        # Fast probabilistic classifier. Assumes features are independent.
        "Naive Bayes": GaussianNB(),

        # ── Logistic Regression ────────────────────────────────────────────
        # Linear model extended to multi-class via softmax
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs',
            random_state=SEED
        ),
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=SEED,
            n_jobs=-1
        )

    # Add LightGBM if available
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        )

    return models


# =============================================================================
# STEP 5: COMPARE ALL MODELS WITH CROSS-VALIDATION
# =============================================================================

def compare_models(X_train, y_train, models):
    """
    Uses 5-Fold Stratified Cross-Validation to compare all models.
    StratifiedKFold ensures each fold has the same class distribution.

    Returns a sorted list of (model_name, mean_accuracy, std_accuracy).
    """
    print(f"\n{'='*60}")
    print("STEP 3: COMPARING ALL MODELS (5-Fold Cross Validation)")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'CV Accuracy':>12} {'Std Dev':>10}")
    print(f"  {'-'*46}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean()
        std_acc = scores.std()
        results.append((name, mean_acc, std_acc, model))
        print(f"  {name:<22} {mean_acc*100:>10.2f}%  ±{std_acc*100:.2f}%")

    # Sort by accuracy (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  🏆 WINNER: {results[0][0]} with {results[0][1]*100:.2f}% CV Accuracy")
    return results


# =============================================================================
# STEP 6: BUILD THE VOTING ENSEMBLE (BEST ARCHITECTURE)
# =============================================================================

def build_voting_ensemble(results, X_train, y_train, top_n=4):
    """
    BEST STRATEGY: Take the top N models and combine them into a VotingClassifier.
    Each model votes → the class with most votes wins.
    voting='soft' means models vote by probabilities (better than hard voting).

    This almost always beats any single model.
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: BUILDING VOTING ENSEMBLE (Top {top_n} models)")
    print(f"{'='*60}")

    # Pick top N models
    top_models = results[:top_n]
    estimators = [(name, model) for name, acc, std, model in top_models]

    print(f"  Combining: {[name for name, _, _, _ in top_models]}")

    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # use predicted probabilities
        n_jobs=-1
    )

    print("  Training ensemble on full training data...")
    ensemble.fit(X_train, y_train)
    print("  Ensemble trained successfully!")
    return ensemble


# =============================================================================
# STEP 7: EVALUATE THE FINAL MODEL
# =============================================================================

def evaluate_model(model, X_test, y_test, encoder):
    """
    Test the final model on unseen test data.
    Prints accuracy, classification report, and confusion matrix.
    Returns the test accuracy.
    """
    print(f"\n{'='*60}")
    print("STEP 5: EVALUATING FINAL MODEL ON TEST SET")
    print(f"{'='*60}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  ✅ Test Accuracy: {acc*100:.4f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    return acc


# =============================================================================
# STEP 8: SAVE THE MODEL (so the UI can load it)
# =============================================================================

def save_model(model, scaler, encoder, accuracy, version=None):
    """
    Save the trained model, scaler, and encoder to disk.
    Also saves metadata JSON with version, accuracy, and timestamp.

    The UI will load these files to make predictions.
    """
    print(f"\n{'='*60}")
    print("STEP 6: SAVING MODEL TO DISK")
    print(f"{'='*60}")

    # Auto-increment version number
    if version is None:
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                old_meta = json.load(f)
            version = old_meta.get('version', 0) + 1
        else:
            version = 1

    # Save model (pickle format)
    with open(BEST_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved → {BEST_MODEL_PATH}")

    # Save scaler (must be the same scaler used during training!)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {SCALER_PATH}")

    # Save label encoder (maps number → crop name)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"  Encoder saved → {ENCODER_PATH}")

    # Save metadata as JSON (readable by humans and the UI)
    meta = {
        "version": version,
        "accuracy": round(accuracy * 100, 4),
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": FEATURE_COLS,
        "classes": list(encoder.classes_),
        "n_classes": len(encoder.classes_)
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_PATH}")
    print(f"  Version: v{version} | Accuracy: {accuracy*100:.2f}%")

    return version


# =============================================================================
# STEP 9: RETRAINING FUNCTION (called automatically when 50 new DB records)
# =============================================================================

def retrain_with_feedback():
    """
    This function is called by the auto-retrainer (auto_retrain.py) when
    the database accumulates 50 new feedback records.

    It:
    1. Loads original data
    2. Loads new feedback data from DB
    3. Combines them
    4. Retrains the best model
    5. Only saves if new accuracy >= old accuracy (safety check)
    """
    print("\n" + "="*60)
    print("AUTO-RETRAINING TRIGGERED BY NEW FEEDBACK DATA")
    print("="*60)

    # Load original data
    df_original = load_and_clean_data(DATA_PATH)

    # Load feedback from DB
    df_feedback = load_feedback_from_db()

    # Combine datasets
    if df_feedback is not None:
        df_combined = pd.concat([df_original, df_feedback], ignore_index=True)
        print(f"  Combined: {len(df_original)} original + {len(df_feedback)} feedback = {len(df_combined)} total rows")
    else:
        df_combined = df_original
        print("  Using original data only (no verified feedback yet)")

    # Preprocess
    X_scaled, y_encoded, scaler, encoder = preprocess(df_combined, fit_scaler=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )

    # Get models and compare
    models = get_all_models()
    results = compare_models(X_train, y_train, models)

    # Build ensemble
    ensemble = build_voting_ensemble(results, X_train, y_train)

    # Evaluate
    new_accuracy = evaluate_model(ensemble, X_test, y_test, encoder)

    # Safety check: only update if new model is at least as good
    old_accuracy = 0.0
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            old_meta = json.load(f)
        old_accuracy = old_meta.get('accuracy', 0) / 100

    if new_accuracy >= old_accuracy:
        version = save_model(ensemble, scaler, encoder, new_accuracy)
        print(f"\n✅ Model updated to v{version}! Accuracy: {old_accuracy*100:.2f}% → {new_accuracy*100:.2f}%")
        return True, new_accuracy
    else:
        print(f"\n⚠️  New model ({new_accuracy*100:.2f}%) is worse than old ({old_accuracy*100:.2f}%). Keeping old model.")
        return False, old_accuracy


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    print("\n" + "🌾 "*20)
    print("   AGROSENSE - CROP RECOMMENDATION SYSTEM")
    print("   Model Training Pipeline")
    print("🌾 "*20 + "\n")

    # ── 1. Load & Clean ───────────────────────────────────────────────────────
    df = load_and_clean_data(DATA_PATH)

    # ── 2. Load feedback if available ─────────────────────────────────────────
    df_feedback = load_feedback_from_db()
    if df_feedback is not None:
        df = pd.concat([df, df_feedback], ignore_index=True)
        print(f"  Total rows after adding feedback: {len(df)}")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    X_scaled, y_encoded, scaler, encoder = preprocess(df, fit_scaler=True)

    # ── 4. Train/Test Split (80% train, 20% test) ─────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2b: TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=SEED,
        stratify=y_encoded   # ensures same class ratio in train & test
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    # ── 5. Compare all models ─────────────────────────────────────────────────
    models = get_all_models()
    results = compare_models(X_train, y_train, models)

    # ── 6. Build voting ensemble ──────────────────────────────────────────────
    ensemble = build_voting_ensemble(results, X_train, y_train, top_n=4)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    final_accuracy = evaluate_model(ensemble, X_test, y_test, encoder)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    version = save_model(ensemble, scaler, encoder, final_accuracy)

    # ── 9. Quick sanity check ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 7: SANITY CHECK - PREDICT ONE SAMPLE")
    print(f"{'='*60}")
    sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])  # should predict rice
    sample_scaled = scaler.transform(sample)
    pred_encoded = ensemble.predict(sample_scaled)[0]
    pred_crop = encoder.inverse_transform([pred_encoded])[0]
    proba = ensemble.predict_proba(sample_scaled)[0]
    confidence = proba.max() * 100
    print(f"  Input: N=90, P=42, K=43, Temp=20.8, Humidity=82, pH=6.5, Rainfall=202.9")
    print(f"  Prediction: {pred_crop.upper()} (Confidence: {confidence:.1f}%)")
    print(f"  Expected: RICE ✅" if pred_crop == 'rice' else f"  ⚠️ Got {pred_crop} instead of rice")

    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Model Version: v{version}")
    print(f"   Test Accuracy: {final_accuracy*100:.2f}%")
    print(f"   Saved to: {MODELS_DIR}/")
    print(f"   Next step: Run app.py to start the web server")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
