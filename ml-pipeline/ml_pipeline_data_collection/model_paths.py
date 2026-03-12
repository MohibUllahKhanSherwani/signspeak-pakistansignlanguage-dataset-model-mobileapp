"""
model_paths.py
==============
Single source of truth for model and encoder file locations.

Import this in api_server.py, train_combined.py, or any script
that needs to load / save the production models.

Usage
-----
    from model_paths import (
        MODEL_BASELINE, MODEL_AUGMENTED,
        ENCODER_BASELINE, ENCODER_AUGMENTED,
        ORIGINAL_DIR,
    )
"""

import os

# Root of ml_pipeline_data_collection/
_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Active (production) models ──────────────────────────────────────────────
MODEL_BASELINE    = os.path.join(_DIR, "action_model_baseline.h5")
MODEL_AUGMENTED   = os.path.join(_DIR, "action_model_augmented.h5")
ENCODER_BASELINE  = os.path.join(_DIR, "label_encoder_baseline.pkl")
ENCODER_AUGMENTED = os.path.join(_DIR, "label_encoder_augmented.pkl")

# ── Original / pre-retrain backup folder ───────────────────────────────────
# train_combined.py moves files here before retraining.
# These are never deleted by training scripts.
ORIGINAL_DIR = os.path.join(_DIR, "original")

ORIGINAL_MODEL_BASELINE    = os.path.join(ORIGINAL_DIR, "action_model_baseline.h5")
ORIGINAL_MODEL_AUGMENTED   = os.path.join(ORIGINAL_DIR, "action_model_augmented.h5")
ORIGINAL_ENCODER_BASELINE  = os.path.join(ORIGINAL_DIR, "label_encoder_baseline.pkl")
ORIGINAL_ENCODER_AUGMENTED = os.path.join(ORIGINAL_DIR, "label_encoder_augmented.pkl")
