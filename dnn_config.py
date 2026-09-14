import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive", "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
ARTIFACT_DIR = os.path.join(CLEAN_DIR, "artifacts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "sintetico")
SYNTHETIC_CLEAN_DIR = os.path.join(SYNTHETIC_DATA_DIR, "clean")
SYNTHETIC_ARTIFACT_DIR = os.path.join(SYNTHETIC_CLEAN_DIR, "artifacts")
SYNTHETIC_RESULTS_DIR = os.path.join(BASE_DIR, "result_sintetico")
SYNTHETIC_INPUT_FILES = {
    "train": os.path.join(SYNTHETIC_CLEAN_DIR, "train_features.parquet"),
    "val": os.path.join(CLEAN_DIR, "val_features.parquet"),
    "test": os.path.join(CLEAN_DIR, "test_features.parquet"),
}

INPUT_FILES = {
    "train": os.path.join(CLEAN_DIR, "train_features.parquet"),
    "val":   os.path.join(CLEAN_DIR, "val_features.parquet"),
    "test":  os.path.join(CLEAN_DIR, "test_features.parquet"),
}

from target_config import CLASSIFICATION_TARGET
CAT_COLS = ["category", "subcategory", "platform", "stock_status"]

BATCH_SIZE = 4096
READ_BATCH = 131_072
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
PATIENCE = 5
MAX_BATCHES_PER_EPOCH = None
USE_AMP = True
USE_CLASS_WEIGHTS = True
GRAD_CLIP = 5.0
LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LR = 1e-5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
