import os
import time
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

import dnn_config as cfg
from dnn_data import load_schema, sample_parquet
from target_config import price_direction_display_labels


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TRAIN_ROWS = 7_897_446  # 7M

TREE_CHECKPOINTS = [
    200,
    400,
    566,   # ganador anterior
    800,
    1000,
    1200,
]

MAX_TREES = max(TREE_CHECKPOINTS)

USE_SAMPLE_WEIGHT = True


# Config ganadora de AutoML
BEST_CONFIG = {
    "learning_rate": 0.06225267745049986,
    "depth": 9,
    "l2_leaf_reg": 10.82917257232035,
    "random_strength": 1.5976349196702044,
    "border_count": 64,
}


# ============================================================
# HELPERS
# ============================================================

def timer(msg):
    class T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'=' * 70}")
            print(f"  {msg}")
            print(f"{'=' * 70}")
            return self

        def __exit__(self, *args):
            print(
                f"  OK en {time.time() - self.t0:.1f}s"
            )

    return T()


def prepare_categories(df, cat_cols):
    for c in cat_cols:
        if c in df.columns:
            # CatBoost maneja categoricas de forma segura como string.
            df[c] = df[c].astype(str)

    return df


def make_sample_weights(y):
    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    total = len(y)
    k = len(classes)

    class_weights = {
        cls: total / (k * count)
        for cls, count
        in zip(classes, counts)
    }

    sample_weight = np.array(
        [
            class_weights[v]
            for v in y
        ],
        dtype=np.float32,
    )

    return sample_weight, class_weights


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("  CATBOOST GPU - CURVA DE ARBOLES CON 10M FILAS")
    print("=" * 70)

    print(f"  Train objetivo : {TRAIN_ROWS:,}")
    print(f"  Checkpoints    : {TREE_CHECKPOINTS}")
    print(f"  GPU            : NVIDIA GeForce RTX 3060 Ti")
    print(f"  Sample weights : {USE_SAMPLE_WEIGHT}")

    os.makedirs(
        cfg.RESULTS_DIR,
        exist_ok=True,
    )

    # ========================================================
    # SCHEMA
    # ========================================================

    num_cols, cat_cols, _ = load_schema()

    feature_cols = (
        num_cols
        + cat_cols
    )

    target = (
        cfg.CLASSIFICATION_TARGET
    )

    train_path = (
        cfg.INPUT_FILES["train"]
    )

    val_path = (
        cfg.INPUT_FILES["val"]
    )

    test_path = (
        cfg.INPUT_FILES["test"]
    )

    # ========================================================
    # COMPROBAR FILAS DISPONIBLES
    # ========================================================

    available_rows = (
        pq.ParquetFile(
            train_path
        ).metadata.num_rows
    )

    print(
        f"\n  Filas disponibles "
        f"en train: {available_rows:,}"
    )

    if available_rows < TRAIN_ROWS:
        raise RuntimeError(
            f"El parquet train solo tiene "
            f"{available_rows:,} filas. "
            f"No se pueden sacar "
            f"{TRAIN_ROWS:,}."
        )

    # ========================================================
    # CARGAR TRAIN 10M
    # ========================================================

    with timer(
        "Cargando muestra de 10M del TRAIN"
    ):

        train_df = sample_parquet(
            train_path,
            feature_cols + [target],
            TRAIN_ROWS,
            seed=SEED,
        )

        train_df = (
            prepare_categories(
                train_df,
                cat_cols,
            )
        )

        X_train = train_df[
            feature_cols
        ]

        y_train = (
            train_df[target]
            .astype(str)
            .to_numpy()
        )

        print(
            f"    X_train: "
            f"{X_train.shape}"
        )

    # ========================================================
    # CARGAR VAL COMPLETO
    # ========================================================

    with timer(
        "Cargando validation completo"
    ):

        val_df = (
            pq.read_table(
                val_path,
                columns=(
                    feature_cols
                    + [target]
                ),
            )
            .to_pandas()
        )

        val_df = (
            prepare_categories(
                val_df,
                cat_cols,
            )
        )

        X_val = val_df[
            feature_cols
        ]

        y_val = (
            val_df[target]
            .astype(str)
            .to_numpy()
        )

        print(
            f"    X_val: "
            f"{X_val.shape}"
        )

    # ========================================================
    # SAMPLE WEIGHTS
    # ========================================================

    sample_weight = None

    if USE_SAMPLE_WEIGHT:

        (
            sample_weight,
            class_weights,
        ) = make_sample_weights(
            y_train
        )

        print(
            "\n  Pesos calculados "
            "sobre los 10M:"
        )

        for cls, weight in (
            class_weights.items()
        ):
            print(
                f"    {cls:15s} "
                f"{weight:.4f}"
            )

    # ========================================================
    # POOLS
    # ========================================================

    print(
        "\n  Construyendo Pool de CatBoost..."
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_cols,
        weight=sample_weight,
    )

    val_pool = Pool(
        X_val,
        y_val,
        cat_features=cat_cols,
    )

    # Liberamos dataframe original.
    del train_df
    del val_df

    # ========================================================
    # MODELO
    # ========================================================

    model = CatBoostClassifier(
        iterations=MAX_TREES,

        learning_rate=(
            BEST_CONFIG[
                "learning_rate"
            ]
        ),

        depth=(
            BEST_CONFIG["depth"]
        ),

        l2_leaf_reg=(
            BEST_CONFIG[
                "l2_leaf_reg"
            ]
        ),

        random_strength=(
            BEST_CONFIG[
                "random_strength"
            ]
        ),

        border_count=(
            BEST_CONFIG[
                "border_count"
            ]
        ),

        loss_function="MultiClass",

        task_type="GPU",
        devices="0",

        random_seed=SEED,

        allow_writing_files=False,

        verbose=50,
    )

    # ========================================================
    # TRAIN HASTA MAX_TREES
    # ========================================================

    print(
        f"\n  Entrenando una sola vez "
        f"hasta {MAX_TREES} arboles..."
    )

    train_start = time.perf_counter()

    model.fit(
        train_pool,
        use_best_model=False,
    )

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    print(
        f"\n  Entrenamiento total: "
        f"{train_seconds:.1f}s"
    )

    # ========================================================
    # EVALUAR CHECKPOINTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  CURVA VALIDATION"
    )

    print(
        "=" * 70
    )

    results = []

    best_f1 = -1.0
    best_trees = None

    for trees in TREE_CHECKPOINTS:

        t0 = time.perf_counter()

        pred = model.predict(
            val_pool,
            ntree_start=0,
            ntree_end=trees,
            prediction_type="Class",
        )

        pred = (
            np.asarray(pred)
            .reshape(-1)
            .astype(str)
        )

        pred_seconds = (
            time.perf_counter()
            - t0
        )

        acc = accuracy_score(
            y_val,
            pred,
        )

        macro_f1 = f1_score(
            y_val,
            pred,
            average="macro",
        )

        results.append({
            "train_rows":
                TRAIN_ROWS,

            "n_estimators":
                trees,

            "val_accuracy":
                float(acc),

            "val_macro_f1":
                float(macro_f1),

            "prediction_time_s":
                float(pred_seconds),
        })

        marker = ""

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_trees = trees
            marker = "  <-- MEJOR"

        print(
            f"  trees={trees:4d}"
            f" | acc={acc:.4f}"
            f" | macro-F1={macro_f1:.4f}"
            f" | pred={pred_seconds:.2f}s"
            f"{marker}"
        )

    # ========================================================
    # GUARDAR CURVA
    # ========================================================

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "n_estimators"
        )
    )

    csv_path = os.path.join(
        cfg.RESULTS_DIR,
        "catboost_10m_tree_curve.csv",
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"\n  Mejor numero de arboles: "
        f"{best_trees}"
    )

    print(
        f"  Mejor VAL Macro-F1: "
        f"{best_f1:.4f}"
    )

    # ========================================================
    # TEST SOLO CON EL GANADOR
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  EVALUACION FINAL EN TEST"
    )

    print(
        "=" * 70
    )

    with timer(
        "Cargando TEST"
    ):

        test_df = (
            pq.read_table(
                test_path,
                columns=(
                    feature_cols
                    + [target]
                ),
            )
            .to_pandas()
        )

        test_df = (
            prepare_categories(
                test_df,
                cat_cols,
            )
        )

        X_test = test_df[
            feature_cols
        ]

        y_test = (
            test_df[target]
            .astype(str)
            .to_numpy()
        )

        test_pool = Pool(
            X_test,
            y_test,
            cat_features=cat_cols,
        )

    inference_start = (
        time.perf_counter()
    )

    test_pred = model.predict(
        test_pool,
        ntree_start=0,
        ntree_end=best_trees,
        prediction_type="Class",
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    test_pred = (
        np.asarray(test_pred)
        .reshape(-1)
        .astype(str)
    )

    test_acc = accuracy_score(
        y_test,
        test_pred,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    print(
        f"\n  TEST accuracy : "
        f"{test_acc:.4f}"
    )

    print(
        f"  TEST Macro-F1 : "
        f"{test_f1:.4f}"
    )

    print(
        f"  Arboles usados: "
        f"{best_trees}"
    )

    print(
        "\n"
        + classification_report(
            y_test,
            test_pred,
            target_names=(
                price_direction_display_labels(
                    sorted(
                        np.unique(
                            y_test
                        )
                    )
                )
            ),
            zero_division=0,
        )
    )

    # ========================================================
    # GUARDAR RESUMEN
    # ========================================================

    summary = {
        "train_rows":
            TRAIN_ROWS,

        "best_n_estimators":
            best_trees,

        "val_macro_f1":
            float(best_f1),

        "test_accuracy":
            float(test_acc),

        "test_macro_f1":
            float(test_f1),

        "train_time_s":
            float(train_seconds),

        "inference_time_s":
            float(inference_seconds),

        "best_config":
            BEST_CONFIG,
    }

    json_path = os.path.join(
        cfg.RESULTS_DIR,
        "catboost_10m_result.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\n  Curva guardada: "
        f"{csv_path}"
    )

    print(
        f"  Resumen guardado: "
        f"{json_path}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  COMPARACION CONTRA 2.37M"
    )

    print(
        "=" * 70
    )

    print(
        "  Modelo anterior:"
    )

    print(
        "    2.37M rows"
        " | Test Acc = 0.5918"
        " | Test F1 = 0.5620"
    )

    print(
        "\n  Modelo 10M:"
    )

    print(
        f"    10.0M rows"
        f" | Test Acc = {test_acc:.4f}"
        f" | Test F1 = {test_f1:.4f}"
    )

    print(
        "\n  Delta F1:"
        f" {test_f1 - 0.56198106631278:+.4f}"
    )

    print(
        "  Delta Acc:"
        f" {test_acc - 0.5918276433280756:+.4f}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()