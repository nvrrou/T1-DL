import os
import sys
import time
import warnings
import json

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score
from flaml import AutoML
from catboost_gpu import CatBoostGPUEstimator, CATBOOST_GPU_SEARCH_SPACE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from dnn_config import (
    INPUT_FILES, ARTIFACT_DIR, RESULTS_DIR,
    CLASSIFICATION_TARGET, CAT_COLS,
)
from dnn_data import sample_parquet

TRAIN_SAMPLE = None
SEEDS = [67, 3429, 42]
TIME_BUDGET_PER_SEED = 600
EARLY_STOP = False
ENSEMBLE = False
HPO_METHOD = "bs"
METRIC = "macro_f1"
MARGIN = 0.005

ESTIMATORS = ["catboost_gpu"]

def load_schema():
    import pickle
    with open(os.path.join(ARTIFACT_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    cat_cols = [c for c in CAT_COLS if c in feature_names]
    return feature_names, cat_cols

def read_split(path, feature_names, cat_cols, sample=None):
    cols = feature_names + [CLASSIFICATION_TARGET]
    df = sample_parquet(path, cols, sample) if sample else pq.read_table(path, columns=cols).to_pandas()
    X = df[feature_names].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = df[CLASSIFICATION_TARGET].astype(str).to_numpy()
    return X, y

def main():
    # Comparar con la nueva corrida de tres clases, no con resultados anteriores.
    with open(os.path.join(RESULTS_DIR, "metrics_automl.json"), encoding="utf-8") as f:
        REFERENCE_ACC = float(json.load(f)["val_accuracy"])
    print("=" * 60)
    print("  05b: Robustez del AutoML (variacion entre semillas en validacion)")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feature_names, cat_cols = load_schema()
    print("\n  Cargando datos (una vez)...")
    X_train, y_train = read_split(INPUT_FILES["train"], feature_names, cat_cols, sample=TRAIN_SAMPLE)
    X_val, y_val = read_split(INPUT_FILES["val"], feature_names, cat_cols)
    print(f"    train (muestra): {X_train.shape}  val: {X_val.shape}")

    classes, counts = np.unique(y_train, return_counts=True)
    freq = dict(zip(classes, counts)); total, k = len(y_train), len(classes)
    sw = np.array([total / (k * freq[v]) for v in y_train], dtype=np.float64)

    print(f"\n  Estimadores: {ESTIMATORS} | HPO: {HPO_METHOD} | {TIME_BUDGET_PER_SEED}s por seed")
    print(f"  early_stop={EARLY_STOP} | ensemble(stacking)={ENSEMBLE}")
    if ENSEMBLE:
        print("  NOTA: con ensemble=True el numero reportado es de un STACKING, no de")
        print("        un modelo unico -> ya no cuenta como 'tecnica de baja complejidad'.")
    print(f"  Referencia (05 base): accuracy = {REFERENCE_ACC:.4f}\n")

    rows = []
    for seed in SEEDS:
        automl = AutoML()
        automl.add_learner(
            learner_name="catboost_gpu",
            learner_class=CatBoostGPUEstimator,
        )
        t0 = time.time()
        automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                   sample=False, retrain_full=False, task="classification",
                   metric=METRIC, estimator_list=ESTIMATORS,
                   time_budget=TIME_BUDGET_PER_SEED, hpo_method=HPO_METHOD,
                   early_stop=EARLY_STOP, ensemble=ENSEMBLE,
                   sample_weight=sw, eval_method="holdout", seed=seed,
                   custom_hp=CATBOOST_GPU_SEARCH_SPACE, verbose=0)
        dt = time.time() - t0
        yp = automl.predict(X_val)
        acc = accuracy_score(y_val, yp)
        f1 = f1_score(y_val, yp, average="macro")
        modelo = "ensemble(stack)" if ENSEMBLE else str(automl.best_estimator)
        rows.append({"seed": seed, "modelo": modelo, "best_estimator": str(automl.best_estimator),
                     "val_accuracy": acc, "val_macro_f1": f1, "tiempo_s": round(dt)})
        marca = "  <-- supera la referencia" if acc > REFERENCE_ACC + MARGIN else ""
        print(f"  seed {seed:>3} | {modelo:<15} | acc {acc:.4f} | macroF1 {f1:.4f} | {dt:.0f}s{marca}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "automl_robustez.csv"), index=False)

    best = df.loc[df["val_accuracy"].idxmax()]
    print(f"\n  Mejor de todas: seed {int(best['seed'])} ({best['best_estimator']}) "
          f"-> accuracy {best['val_accuracy']:.4f}")
    print(f"  Referencia 05 base:                 accuracy {REFERENCE_ACC:.4f}")
    delta = best["val_accuracy"] - REFERENCE_ACC
    print(f"  Diferencia:                         {delta:+.4f}")
    print("Variacion entre semillas en validacion; no demuestra un techo de desempeno.")
    print("\n  Guardado: results/automl_robustez.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
