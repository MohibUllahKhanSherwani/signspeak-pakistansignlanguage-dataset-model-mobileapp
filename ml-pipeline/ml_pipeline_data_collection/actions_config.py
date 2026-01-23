import os

# Get the directory of this config file
config_dir = os.path.dirname(os.path.abspath(__file__))

# Path to actions file (relative to this config file)
ACTIONS_FILE = os.path.join(config_dir, "actions.txt")

# Where data will be stored (relative to this config file)
DATA_PATH = os.path.join(config_dir, "MP_Data")

# Recording params
SEQUENCE_LENGTH = 30          # number of frames per sequence
NUM_SEQUENCES = 50            # how many sequences per action
FRAME_WAIT_MS = 50           # delay between frames during collection

# Model params
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001

# Inference params
PREDICTION_THRESHOLD = 0.5

# Augmentation params
USE_AUGMENTATION = True          # Enable/disable augmentation
AUGMENTATION_MULTIPLIER = 3      # How many augmented versions per original (3x recommended)
AUGMENTATION_PROBABILITIES = {
    # 'horizontal_flip': 0.5,    # DISABLED: PSL gestures are non-symmetric
    'time_warp': 0.5,            # Speed variations - ESSENTIAL
    'spatial_scale': 0.5,        # Distance from camera variations
    'spatial_translate': 0.5,    # Position variations
    'spatial_rotate': 0.3,       # Slight rotation (use carefully)
    'add_noise': 0.3,            # Sensor noise simulation
    'temporal_crop': 0.3,        # Partial sign variations
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
