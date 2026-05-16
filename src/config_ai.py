from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" / "coil_only_ai"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "coil_unetpp_effb4_scratch_v8_best.pt"

VALID_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

ARCHITECTURE = "UnetPlusPlus"
ENCODER_NAME = "efficientnet-b4"
ENCODER_NAME_FALLBACKS = (
    "timm-efficientnet-b4",
    "tu-efficientnet_b4",
)
IN_CHANNELS = 3
CLASSES = 1
ACTIVATION = None
CLASS_NAME = "coil"
INPUT_SIZE = 512
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
PREDICTION_SCORE_PERCENTILE = 99.5

DEVICE = "auto"
CONF_THRESHOLD = 0.50
MASK_THRESHOLD = 0.30
MIN_COMPONENT_AREA = 64
MORPH_OPEN_KERNEL = 0
MORPH_CLOSE_KERNEL = 0
OUTER_RECOVER_KERNEL = 0
KEEP_LARGEST_COMPONENT = True
PRESERVE_INNER_HOLES = True
MIN_HOLE_AREA = 64
SAVE_MASK = False

DATASET_ROOT = ROOT_DIR / "prepared_trainset_v8"
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_MASKS_DIR = DATASET_ROOT / "train" / "masks"
VAL_IMAGES_DIR = DATASET_ROOT / "val" / "images"
VAL_MASKS_DIR = DATASET_ROOT / "val" / "masks"

BATCH_SIZE = 1
NUM_WORKERS = 0
EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
AMP = True
PIN_MEMORY = False
BEST_CHECKPOINT_PATH = MODEL_PATH
LAST_CHECKPOINT_PATH = MODELS_DIR / "coil_unetpp_effb4_scratch_v8_last.pt"
