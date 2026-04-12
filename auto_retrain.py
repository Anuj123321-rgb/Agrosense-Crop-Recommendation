# =============================================================================
# FILE: auto_retrain.py
# PURPOSE: Standalone script that runs in background and checks the DB
#          every 6 hours. If >= 50 new feedback records → retrain the model.
#
# WHY THIS FILE EXISTS:
#   app.py handles real-time retraining triggered when user submits feedback.
#   This file is a BACKUP SCHEDULER that runs independently —
#   useful when app.py is restarted or if feedback was added offline.
#
# HOW TO RUN:
#   python auto_retrain.py          (runs forever, checking every 6 hours)
#   python auto_retrain.py --now    (force retrain RIGHT NOW, for testing)
# =============================================================================

import time
import sys
import os
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AUTO-RETRAIN] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check every 6 hours (in seconds)
CHECK_INTERVAL_SECONDS = 6 * 60 * 60   # = 21600 seconds

# For testing, you can change this to 60 (check every minute)
# CHECK_INTERVAL_SECONDS = 60

MODELS_DIR = "models"
META_PATH  = os.path.join(MODELS_DIR, "model_meta.json")


def get_old_accuracy():
    """Read current model accuracy from metadata file."""
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        return meta.get('accuracy', 0) / 100
    return 0.0


def run_retrain_check():
    """
    Main check: look at DB, retrain if threshold reached.
    """
    logger.info("Running scheduled retrain check...")

    from database import (
        check_retrain_threshold, get_unused_feedback_ids,
        mark_feedback_used, log_retrain, save_model_version, get_db_stats
    )

    stats = get_db_stats()
    pending = stats['pending_feedback']
    logger.info(f"  Pending feedback records: {pending} / 50 threshold")

    if not check_retrain_threshold():
        logger.info(f"  Threshold not reached. Need {50 - pending} more feedback records.")
        return False

    logger.info(f"  ✅ Threshold reached! Starting retraining with {pending} new records...")

    try:
        from train_model import retrain_with_feedback

        feedback_ids = get_unused_feedback_ids()
        old_accuracy = get_old_accuracy()

        updated, new_accuracy = retrain_with_feedback()

        if updated:
            mark_feedback_used(feedback_ids)
            log_retrain(
                new_feedback_count=len(feedback_ids),
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                model_updated=True,
                notes="Scheduled auto-retrain"
            )

            # Save new version to DB
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            save_model_version(
                version=meta['version'],
                accuracy=new_accuracy * 100,
                total_records=2200 + len(feedback_ids),
                feedback_records=len(feedback_ids),
                notes=f"Scheduled retrain at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            logger.info(f"✅ RETRAIN SUCCESS! Accuracy: {old_accuracy*100:.2f}% → {new_accuracy*100:.2f}%")
            logger.info(f"   {len(feedback_ids)} feedback records marked as used.")
            logger.info(f"   app.py will load the new model on next request.")
        else:
            logger.info(f"⚠️  New model was worse ({new_accuracy*100:.2f}% < {old_accuracy*100:.2f}%). Kept old model.")

        return updated

    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main loop: check DB every CHECK_INTERVAL_SECONDS.
    """
    force_now = '--now' in sys.argv

    print("\n" + "="*60)
    print("  AGROSENSE - Auto Retrain Scheduler")
    print("="*60)

    if force_now:
        print("\n  Mode: FORCE RETRAIN NOW (--now flag)")
        print("="*60 + "\n")
        run_retrain_check()
        return

    print(f"  Mode: Scheduled (every {CHECK_INTERVAL_SECONDS // 3600} hours)")
    print(f"  Retrain threshold: 50 new feedback records")
    print("  Press CTRL+C to stop")
    print("="*60 + "\n")

    while True:
        try:
            run_retrain_check()
            logger.info(f"  Next check in {CHECK_INTERVAL_SECONDS // 3600} hours...")
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Retrying in 30 minutes...")
            time.sleep(30 * 60)


if __name__ == "__main__":
    main()
