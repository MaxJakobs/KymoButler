"""Default configuration constants for KymoButler."""

from pathlib import Path

# Model URLs (Wolfram Cloud hosted originals - for reference)
WOLFRAM_CLOUD_BASE = (
    "https://www.wolframcloud.com/objects/deepmirror/Projects/KymoButler/networks/"
)

MODEL_NAMES = {
    "binet": "Bidirectional_Segmentation_Module_V1_1",
    "classnet": "Classification_Module_V1_0",
    "uninet": "Unidrectional_Segmentation_Module_V1_0",
    "decnet": "Decision_Module_V1_0",
}

# PyTorch weight filenames
WEIGHT_FILES = {
    "binet": "bidirectional_seg.pt",
    "uninet": "unidirectional_seg.pt",
    "decnet": "decision_module.pt",
    "classnet": "classifier.pt",
}

# Model directories: check repo models/ first, then ~/.kymobutler/models/
_REPO_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_HOME_MODEL_DIR = Path.home() / ".kymobutler" / "models"
DEFAULT_MODEL_DIR = _REPO_MODEL_DIR if _REPO_MODEL_DIR.exists() else _HOME_MODEL_DIR

# Segmentation defaults
DEFAULT_THRESHOLD = 0.2
DEFAULT_VISION_THRESHOLD = 0.5
DEFAULT_MIN_SIZE = 10
DEFAULT_MIN_FRAMES = 10

# Tracking
SEARCH_RADIUS = 1.5
VISION_MODULE_TILE_SIZE = 48
MAX_CANDIDATES = 24
STRADDLER_MAX_ITERATIONS = 500
STRADDLER_PIXEL_THRESHOLD = 5

# UNet
UNET_BASE_CHANNELS = 64
