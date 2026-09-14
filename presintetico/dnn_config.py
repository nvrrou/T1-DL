import os
import torch


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "archive",
    "data",
)

CLEAN_DIR = os.path.join(
    DATA_DIR,
    "clean",
)

ARTIFACT_DIR = os.path.join(
    CLEAN_DIR,
    "artifacts",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

INPUT_FILES = {
    "train": os.path.join(
        CLEAN_DIR,
        "train_features.parquet",
    ),
    "val": os.path.join(
        CLEAN_DIR,
        "val_features.parquet",
    ),
    "test": os.path.join(
        CLEAN_DIR,
        "test_features.parquet",
    ),
}



from target_config import (
    CLASSIFICATION_TARGET
)


CAT_COLS = [
    "category",
    "subcategory",
    "platform",
    "stock_status",
]


# ============================================================
# DATOS
# ============================================================

BATCH_SIZE = 4096

READ_BATCH = 131_072


# ============================================================
# ARQUITECTURA
# ============================================================

HIDDEN_LAYERS = [
    256,
    128,
    64,
]

# Antes 0.30.
# 0.15 deja aprender mas sin quitar regularizacion.
DROPOUT = 0.15


# ============================================================
# OPTIMIZACION
# ============================================================

LEARNING_RATE = 1e-3

# Antes 1e-5.
WEIGHT_DECAY = 1e-4


# ============================================================
# ENTRENAMIENTO
# ============================================================

EPOCHS = 50

PATIENCE = 7


# ============================================================
# SCHEDULER
# ============================================================

LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-5


# ============================================================
# LOSS
# ============================================================

# Empieza con True porque quieres optimizar Macro-F1.
# Luego puedes hacer una corrida False para comparar.
USE_CLASS_WEIGHTS = True


# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = True


# ============================================================
# ESTABILIDAD
# ============================================================

GRAD_CLIP = 5.0


MAX_BATCHES_PER_EPOCH = None

SEED = 42


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
