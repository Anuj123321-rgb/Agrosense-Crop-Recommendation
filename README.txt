================================================================================
   AGROSENSE — CROP RECOMMENDATION SYSTEM
   Complete Setup & Integration Guide
   B.Tech Final Year Major Project
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PROJECT STRUCTURE (what each file does)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  crop_project/
  │
  ├── Crop_recommendation.csv     ← Your original dataset (2200 rows)
  │
  ├── train_model.py              ← STEP 1: Trains all ML models, saves the best
  ├── database.py                 ← STEP 2: Creates SQLite DB and all functions
  ├── app.py                      ← STEP 3: Flask web server (API + serves UI)
  ├── auto_retrain.py             ← STEP 4: Background scheduler for retraining
  │
  ├── ui.html                     ← The beautiful interactive frontend
  │
  ├── requirements.txt            ← All Python packages needed
  │
  └── models/                     ← Created automatically after training
      ├── best_model.pkl          ← The trained ML model (loaded by app.py)
      ├── scaler.pkl              ← The StandardScaler (MUST match training)
      ├── label_encoder.pkl       ← Maps numbers ↔ crop names
      └── model_meta.json         ← Version, accuracy, timestamp metadata

  └── crop_database.db            ← SQLite database (created by database.py)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW EVERYTHING CONNECTS (THE BIG PICTURE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User opens browser
       │
       ▼
  ui.html (the frontend)
       │  sends JSON via HTTP POST
       ▼
  app.py  (Flask server at localhost:5000)
       │  loads model from disk and preprocesses input
       ▼
  best_model.pkl + scaler.pkl + label_encoder.pkl
       │  returns prediction as JSON
       ▼
  ui.html shows result to user
       │  user gives feedback (star rating, outcome)
       ▼
  app.py saves to crop_database.db
       │  checks: is count of new feedback >= 50?
       ▼
  If YES → trigger background thread → train_model.py retrains
       │  saves new model to models/ folder
       ▼
  app.py reloads model → next predictions use improved model


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP-BY-STEP SETUP GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PREREQUISITES: Python 3.8+ installed, pip available

  ────────────────────────────────────────────────────────────────
  STEP 1: Setup your project folder
  ────────────────────────────────────────────────────────────────

    Create a new folder and put ALL files in it:

      mkdir crop_project
      cd crop_project

    Copy these files into crop_project/:
      - Crop_recommendation.csv
      - train_model.py
      - database.py
      - app.py
      - auto_retrain.py
      - ui.html
      - requirements.txt


  ────────────────────────────────────────────────────────────────
  STEP 2: Install required Python packages
  ────────────────────────────────────────────────────────────────

    Open terminal/command prompt INSIDE the crop_project folder:

      pip install -r requirements.txt

    This installs: pandas, numpy, scikit-learn, flask, flask-cors,
                   xgboost, lightgbm, matplotlib, seaborn, apscheduler

    If you get errors with xgboost or lightgbm, install separately:
      pip install xgboost
      pip install lightgbm


  ────────────────────────────────────────────────────────────────
  STEP 3: Setup the database
  ────────────────────────────────────────────────────────────────

    Run this command in your terminal:

      python database.py

    What happens:
      - Creates crop_database.db file in your folder
      - Creates 4 tables: predictions_log, user_feedback,
        model_versions, retrain_log
      - Inserts 1 test record to verify everything works

    You should see:
      ✅ Database initialized at: crop_database.db
      ✅ Test prediction saved with ID: 1
      ✅ Test feedback saved with ID: 1
      ✅ Model version 1 recorded in DB


  ────────────────────────────────────────────────────────────────
  STEP 4: Train the ML models (THE IMPORTANT STEP)
  ────────────────────────────────────────────────────────────────

    Run this command:

      python train_model.py

    This will take 2-5 minutes depending on your computer.

    What happens (in order):
      1. Loads Crop_recommendation.csv
      2. Applies IQR outlier removal (same as your notebook)
      3. Encodes labels (rice=0, maize=1, ... coffee=21)
      4. Scales features with StandardScaler
      5. Splits data: 80% train, 20% test
      6. Trains ALL these models and compares them:
           - Random Forest (200 trees)
           - Extra Trees (200 trees)
           - Gradient Boosting (150 estimators)
           - SVM (RBF kernel)
           - KNN (5 neighbors)
           - Naive Bayes
           - Logistic Regression
           - XGBoost (200 estimators, if installed)
           - LightGBM (200 estimators, if installed)
      7. Picks the TOP 4 by cross-validation accuracy
      8. Combines them into a Voting Ensemble (soft voting)
      9. Tests on held-out 20% test data
      10. Saves to models/ folder:
            - best_model.pkl
            - scaler.pkl
            - label_encoder.pkl
            - model_meta.json

    Expected output (approximate):
      Model                   CV Accuracy    Std Dev
      ──────────────────────────────────────────────
      Random Forest           99.09%  ±0.55%
      XGBoost                 99.05%  ±0.52%
      Extra Trees             98.95%  ±0.61%
      LightGBM                98.80%  ±0.48%
      Gradient Boosting       97.73%  ±0.72%
      SVM                     97.50%  ±0.80%
      ...

      🏆 WINNER: Random Forest with 99.09% CV Accuracy

      ✅ Test Accuracy: 99.09%


  ────────────────────────────────────────────────────────────────
  STEP 5: Start the web server
  ────────────────────────────────────────────────────────────────

    Run this command:

      python app.py

    What happens:
      - Initializes the database (safe to re-run)
      - Loads the trained model from models/best_model.pkl into memory
      - Starts Flask server at http://localhost:5000
      - Waits for requests from the UI

    You should see:
      ✅ SERVER RUNNING!
      Open your browser at: http://localhost:5000


  ────────────────────────────────────────────────────────────────
  STEP 6: Open the UI in your browser
  ────────────────────────────────────────────────────────────────

    Open your browser and go to:
      http://localhost:5000

    The UI will:
      - Load live stats from the /stats API
      - Show your model version and accuracy
      - Let you enter soil parameters using sliders
      - Show crop recommendation with confidence
      - Let you submit farmer feedback
      - Show the retraining progress bar


  ────────────────────────────────────────────────────────────────
  STEP 7 (Optional): Start the auto-retrain scheduler
  ────────────────────────────────────────────────────────────────

    In a SECOND terminal window, run:

      python auto_retrain.py

    This runs a background scheduler that checks the database
    every 6 hours. If 50 new feedback records exist, it retrains.

    To test it immediately (force retrain now):
      python auto_retrain.py --now


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW THE API WORKS (for your understanding)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The UI sends HTTP requests to Flask. Here are all the endpoints:

  ┌─────────────────────────────────────────────────────────────┐
  │  GET  /           → Serves ui.html to the browser           │
  │  POST /predict    → Takes soil data, returns prediction      │
  │  POST /feedback   → Saves farmer feedback, checks retrain    │
  │  GET  /stats      → Returns DB stats (for progress bar)      │
  │  GET  /model-history → Returns all model versions           │
  │  GET  /health     → Simple health check                      │
  └─────────────────────────────────────────────────────────────┘

  Example: What happens when you click "Get Recommendation":

  1. ui.html collects slider values → creates JSON object
  2. Sends POST to http://localhost:5000/predict
  3. app.py receives the JSON
  4. Applies scaler.transform() on the values
  5. Calls model.predict() and model.predict_proba()
  6. Saves to predictions_log table in database
  7. Returns JSON with crop name, confidence, top 5 alternatives
  8. ui.html renders the result on screen


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW THE AUTO-RETRAINING WORKS (explain in your presentation!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Farmer submits feedback → saved to user_feedback table
     with used_for_training = 0

  2. app.py checks:
     SELECT COUNT(*) FROM user_feedback WHERE used_for_training = 0

  3. If count >= 50:
     → Spawn background thread (so UI doesn't freeze)
     → Call retrain_with_feedback()

  4. retrain_with_feedback():
     → Loads original 2200 rows from CSV
     → Queries feedback where outcome='yes' or correct_crop provided
     → Combines both datasets
     → Runs full model training pipeline
     → Tests new model accuracy

  5. Safety check:
     IF new_accuracy >= old_accuracy:
         save new model files
         mark feedback records as used_for_training = 1
     ELSE:
         keep old model (new model was worse)

  6. app.py reloads model from disk
     → All future predictions use the improved model

  This is exactly how production ML systems like Netflix, Spotify
  retrain their recommendation models on user feedback!


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TROUBLESHOOTING COMMON ERRORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ERROR: "Model not found at models/best_model.pkl"
  FIX:   Run python train_model.py FIRST before app.py

  ERROR: "No module named flask"
  FIX:   Run pip install flask flask-cors

  ERROR: "CORS error" in browser console
  FIX:   Make sure you're opening http://localhost:5000 (not the .html file)

  ERROR: "Connection refused" in browser
  FIX:   Make sure app.py is still running in terminal

  ERROR: xgboost import error
  FIX:   pip install xgboost  OR remove XGBoost from train_model.py

  ERROR: "Port 5000 already in use"
  FIX:   Change port in app.py: app.run(port=5001) and update API_BASE in ui.html


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PRESENTATION CHECKLIST (April 16, 2026)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Before presenting:
  [ ] Run python train_model.py and note the accuracy numbers
  [ ] Run python database.py to initialize DB
  [ ] Run python app.py in one terminal (keep it open)
  [ ] Open http://localhost:5000 in browser
  [ ] Pre-fill a few "test" feedback records so progress bar isn't at 0%
  [ ] Take screenshots of training output showing model comparison table

  During demo:
  [ ] Show the model comparison table (all 7-9 models, their accuracy)
  [ ] Show train_model.py code explaining the ensemble voting strategy
  [ ] Show database.py explaining the 4 tables and what they store
  [ ] Live demo: enter soil values → show prediction with confidence
  [ ] Show the 3 charts: radar, bar, donut
  [ ] Submit feedback → show progress bar update
  [ ] Explain: "When 50 records collected, model auto-retrains in background"
  [ ] Show model_meta.json with version number and accuracy

  Key talking points:
  - "We use a Voting Ensemble of the top 4 models, not just 1 model"
  - "Every prediction is stored in SQLite database for future training"
  - "Farmer feedback is verified before being used for retraining"
  - "Safety check: we only update if new model is at least as accurate"
  - "This is how real ML systems like recommendation engines work"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RUN ORDER SUMMARY (quick reference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Terminal 1 (run once):
    pip install -r requirements.txt
    python database.py
    python train_model.py

  Terminal 1 (every time you want to use the app):
    python app.py

  Terminal 2 (optional, for auto-retrain scheduler):
    python auto_retrain.py

  Browser:
    http://localhost:5000

================================================================================
