import os

# Get the directory of this config file
config_dir = os.path.dirname(os.path.abspath(__file__))

# Path to actions file (relative to this config file)
ACTIONS_FILE = os.path.join(config_dir, "actions.txt")

# Where data will be stored (relative to this config file)
DATA_PATH = os.path.join(config_dir, "MP_Data")

# Recording params
SEQUENCE_LENGTH = 30          # number of frames per sequence
NUM_SEQUENCES = 50            # how many sequences per action (50 is sufficient with augmentation)
FRAME_WAIT_MS = 50           # delay between frames during collection

# Model params
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001

# Inference params
PREDICTION_THRESHOLD = 0.5

# Augmentation params
USE_AUGMENTATION = True          # Enable/disable augmentation
AUGMENTATION_MULTIPLIER = 3      # Data expansion factor (e.g., 3x total data)
AUGMENTATION_PROBABILITIES = {
    'time_warp': 0.3,            # Speed variations (reduced)
    'spatial_scale': 0.2,        # Distance variations (reduced)
    'spatial_translate': 0.1,    # Position variations (reduced)
    'spatial_rotate': 0.0,       # DISABLED: Too risky for sign recognition
    'add_noise': 0.1,            # Sensor noise (reduced)
    'temporal_crop': 0.2,        # Start/end variations (reduced)
}

def load_actions():
    """Load actions from actions.txt (one per line). Returns list of strings."""
    if not os.path.exists(ACTIONS_FILE):
        raise FileNotFoundError(f"{ACTIONS_FILE} not found. Create it and put one action per line.")

    with open(ACTIONS_FILE, "r", encoding="utf-8") as f:
        actions = [line.strip() for line in f if line.strip()]

    if not actions:
        raise ValueError("actions.txt is empty. Add one action per line.")

    return actions
