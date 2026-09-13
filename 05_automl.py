import os
import sys
import json
import time
import warnings

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from flaml import AutoML

from dnn_config import (
    INPUT_FILES, CLEAN_DIR, ARTIFACT_DIR, RESULTS_DIR,
    CLASSIFICATION_TARGET, CAT_COLS,
)
from dnn_data import sample_parquet
from experiment_results import save_metrics
from target_config import PRICE_DIRECTION_CLASSES, price_direction_display_labels

AUTOML_TRAIN_SAMPLE = None
TIME_BUDGET = 1800
EARLY_STOP = True
MAX_ITER = 1_000_000
ESTIMATORS = ["lgbm", "rf", "extra_tree", "lrl2"]
METRIC = "macro_f1"
RANDOM_STATE = 42
FLAML_LOG = os.path.join(RESULTS_DIR, "automl_flaml.log")

def load_schema():
    import pickle
    with open(os.path.join(ARTIFACT_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    cat_cols = [c for c in CAT_COLS if c in feature_names]
    num_cols = [c for c in feature_names if c not in cat_cols]
    return feature_names, num_cols, cat_cols

def read_split(path, feature_names, cat_cols, sample=None):
    cols = feature_names + [CLASSIFICATION_TARGET]
    if sample:
        df = sample_parquet(path, cols, sample, seed=RANDOM_STATE)
    else:
        df = pq.read_table(path, columns=cols).to_pandas()
    X = df[feature_names].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = df[CLASSIFICATION_TARGET].astype(str).to_numpy()
    return X, y

def main():
    print("=" * 60)
    print("  PASO 3 (2.1): AutoML con FLAML")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feature_names, num_cols, cat_cols = load_schema()
    print(f"  Features: {len(feature_names)} ({len(num_cols)} num + {len(cat_cols)} cat)")

    print("\n  Cargando datos...")
    X_train, y_train = read_split(INPUT_FILES["train"], feature_names, cat_cols, sample=AUTOML_TRAIN_SAMPLE)
    X_val, y_val = read_split(INPUT_FILES["val"], feature_names, cat_cols)
    X_test, y_test = read_split(INPUT_FILES["test"], feature_names, cat_cols)
    print(f"    train (muestra): {X_train.shape}  test: {X_test.shape}")

    classes, counts = np.unique(y_train, return_counts=True)
    freq = dict(zip(classes, counts))
    total, k = len(y_train), len(classes)
    sample_weight = np.array([total / (k * freq[v]) for v in y_train], dtype=np.float64)

    budget_txt = "sin limite (hasta converger)" if TIME_BUDGET == -1 else f"{TIME_BUDGET}s (~{TIME_BUDGET//60} min)"
    print(f"\n  Buscando (tiempo={budget_txt}, early_stop={EARLY_STOP}, estimadores={ESTIMATORS})...")
    print(f"  Progreso en vivo: {FLAML_LOG}")
    automl = AutoML()
    t0 = time.time()
    automl.fit(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val, sample=False, retrain_full=False,
        task="classification",
        metric=METRIC,
        estimator_list=ESTIMATORS,
        time_budget=TIME_BUDGET,
        max_iter=MAX_ITER,
        early_stop=EARLY_STOP,
        sample_weight=sample_weight,
        eval_method="holdout",
        seed=RANDOM_STATE,
        log_file_name=FLAML_LOG,
        verbose=1,
    )
    search_time = time.time() - t0

    print(f"\n  Mejor modelo: {automl.best_estimator}")
    print(f"  Config: {automl.best_config}")

    inference_start = time.perf_counter()
    y_pred = automl.predict(X_test)
    inference_seconds = time.perf_counter() - inference_start
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    train_acc = accuracy_score(y_train, automl.predict(X_train))

    print(f"\n  Test accuracy: {test_acc:.4f} | Test macro-F1: {test_f1:.4f}")
    print("\n" + classification_report(
        y_test, y_pred, labels=PRICE_DIRECTION_CLASSES,
        target_names=price_direction_display_labels(PRICE_DIRECTION_CLASSES), zero_division=0,
    ))

    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred, labels=PRICE_DIRECTION_CLASSES)
    ConfusionMatrixDisplay(
        cm, display_labels=price_direction_display_labels(PRICE_DIRECTION_CLASSES)
    ).plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_xlabel("Predicha", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax.set_title(f"AutoML ({automl.best_estimator}) - Test (Acc={test_acc:.4f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "11_automl_confusion.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    complexity = model_complexity(automl.model.estimator)

    joblib.dump(automl, os.path.join(ARTIFACT_DIR, "automl_model.pkl"))
    metrics = {
        "name": f"AutoML ({automl.best_estimator})",
        "family": "AutoML / ML clasico",
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "train_accuracy": float(train_acc),
        "generalizacion_gap": float(train_acc - test_acc),
        "complejidad_params": complexity,
        "complejidad_unidad": model_complexity_unit(automl.model.estimator),
        "tiempo_s": float(search_time),
        "search_time_s": float(search_time), "fit_time_s": 0.0,
        "inference_time_s": inference_seconds, "device": "cpu",
        "val_accuracy": float(accuracy_score(y_val, automl.predict(X_val))),
        "val_macro_f1": float(f1_score(y_val, automl.predict(X_val), average="macro")),
        "best_estimator": str(automl.best_estimator),
        "best_config": {k2: _jsonable(v) for k2, v in (automl.best_config or {}).items()},
    }
    save_metrics("automl", metrics, y_test, y_pred,
                 os.path.join(ARTIFACT_DIR, "automl_model.pkl"), ARTIFACT_DIR, RESULTS_DIR)

    print(f"\n{'='*60}")
    print(f"  AutoML listo | {automl.best_estimator} | test acc {test_acc:.4f} | {search_time:.0f}s")
    print(f"  Guardado: results/metrics_automl.json")
    print(f"{'='*60}")

def model_complexity(est):
    try:
        if hasattr(est, "coef_"):
            return int(np.asarray(est.coef_).size + np.asarray(est.intercept_).size)
        if hasattr(est, "booster_"):
            return int(sum(tree["num_leaves"] for tree in est.booster_.dump_model()["tree_info"]))
        if hasattr(est, "estimators_"):
            return int(sum(tree.tree_.node_count for tree in np.asarray(est.estimators_).ravel()))
        if hasattr(est, "get_booster"):
            return int(len(est.get_booster().trees_to_dataframe()))
    except Exception:
        pass
    return "n/d"

def model_complexity_unit(est):
    if hasattr(est, "coef_"):
        return "parametros entrenables"
    if hasattr(est, "booster_"):
        return "hojas de arbol"
    if hasattr(est, "estimators_") or hasattr(est, "get_booster"):
        return "nodos de arbol"
    return "n/d"

def _jsonable(v):
    try:
        json.dumps(v); return v
    except Exception:
        return str(v)

if __name__ == "__main__":
    main()
