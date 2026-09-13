import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from experiment_results import save_metrics
from target_config import (
    PRICE_DIRECTION_CLASSES,
    price_direction_display_labels,
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive", "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
ARTIFACT_DIR = os.path.join(CLEAN_DIR, "artifacts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from target_config import CLASSIFICATION_TARGET
REGRESSION_TARGETS = ["target_price_7d", "target_price_30d"]

TRAIN_SAMPLE = None  # Mismo train completo que DNN, AutoML y NAS.
RANDOM_STATE = 42

SVM_RBF_SAMPLE = 50_000

LEARNING_CURVE_SAMPLE = 100_000

def timer(msg: str):
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  {msg}")
            print(f"{'='*60}")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            print(f"  OK Completado en {elapsed:.1f}s")
    return Timer()

def load_clean_data():
    data = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(CLEAN_DIR, f"{split}_features.parquet")
        if not os.path.exists(path):
            print(f"  Archivo no encontrado: {path}")
            print(f"  Ejecuta primero: python 02_feature_engineering.py")
            sys.exit(1)
        df = pd.read_parquet(path)
        if split == "train" and TRAIN_SAMPLE and len(df) > TRAIN_SAMPLE:
            n_full = len(df)
            df = df.sample(n=TRAIN_SAMPLE, random_state=RANDOM_STATE).reset_index(drop=True)
            print(f"    {split}: muestra de {TRAIN_SAMPLE:,} filas (de {n_full:,})")
        else:
            print(f"    {split}: {df.shape}")
        data[split] = df
    return data

def prepare_xy(data: dict):
    X, y = {}, {}
    for split, df in data.items():
        target_cols = [CLASSIFICATION_TARGET] + [
            c for c in REGRESSION_TARGETS if c in df.columns
        ]
        feature_cols = [c for c in df.columns if c not in target_cols]
        X[split] = df[feature_cols].to_numpy(dtype=np.float32)
        y[split] = df[CLASSIFICATION_TARGET].astype(str).to_numpy()
        print(f"    {split} -> X: {X[split].shape}, y: {y[split].shape}")
    return X, y

def save_figure(fig, name: str):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Guardado: {path}")

def format_confusion_axes(ax):
    """Etiquetas compactas y legibles para matrices de cinco clases."""
    ax.set_xlabel("Predicha", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def plot_logistic_confusion(y_true, y_pred, classes, acc):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    ConfusionMatrixDisplay(
        cm, display_labels=price_direction_display_labels(classes)
    ).plot(ax=ax, cmap="Oranges", values_format="d")
    format_confusion_axes(ax)
    ax.set_title(f"Regresion Logistica - Matriz de Confusion (Acc={acc:.4f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "02_logistic_regression_cm")

def print_metrics(y_true, y_pred, model_name: str, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  --- {model_name} ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    return acc

def run_linear_regression(X, y, target_names):
    with timer("Modelo 1: Regresion Lineal (discretizada)"):
        unique_classes = np.unique(y["train"])
        class_to_int = {c: i for i, c in enumerate(unique_classes)}
        int_to_class = {i: c for c, i in class_to_int.items()}

        y_train_num = np.array([class_to_int[c] for c in y["train"]])
        y_val_num = np.array([class_to_int[c] for c in y["val"]])

        model = LinearRegression()
        model.fit(X["train"], y_train_num)

        y_pred_raw = model.predict(X["val"])
        y_pred_int = np.clip(np.round(y_pred_raw).astype(int), 0, len(unique_classes) - 1)
        y_pred = np.array([int_to_class[i] for i in y_pred_int])

        acc = print_metrics(y["val"], y_pred, "Regresion Lineal", target_names)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(y_pred_raw, bins=100, alpha=0.7, color="#2196F3", edgecolor="white")
        short_labels = price_direction_display_labels(unique_classes)
        for i, label in enumerate(short_labels):
            axes[0].axvline(x=i, color="red", linestyle="--", alpha=0.7, label=f"{label} = {i}")
        axes[0].set_title("Distribucion de predicciones continuas", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Valor predicho")
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend(fontsize=8)

        cm = confusion_matrix(y["val"], y_pred, labels=unique_classes)
        ConfusionMatrixDisplay(cm, display_labels=short_labels).plot(ax=axes[1], cmap="Blues", values_format="d")
        format_confusion_axes(axes[1])
        axes[1].set_title(f"Matriz de Confusion (Acc={acc:.4f})", fontsize=12, fontweight="bold")

        fig.suptitle("Regresion Lineal - Baseline", fontsize=14, fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "01_linear_regression")

        return acc

def run_logistic_regression(X, y, target_names):
    with timer("Modelo 2: Regresion Logistica (multinomial)"):
        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=1,
            verbose=0,
        )
        fit_start = time.perf_counter()
        model.fit(X["train"], y["train"])
        fit_seconds = time.perf_counter() - fit_start

        y_pred = model.predict(X["val"])
        acc = print_metrics(y["val"], y_pred, "Regresion Logistica", target_names)

        unique_classes = np.unique(y["train"])

        plot_logistic_confusion(y["val"], y_pred, unique_classes, acc)

        joblib.dump(model, os.path.join(RESULTS_DIR, "logistic_regression_model.pkl"))

        inference_start = time.perf_counter()
        test_pred = model.predict(X["test"])
        inference_seconds = time.perf_counter() - inference_start
        save_metrics("baseline", {
            "name": "Baseline (Reg. Logistica)", "family": "ML clasico lineal",
            "train_accuracy": float(accuracy_score(y["train"], model.predict(X["train"]))),
            "val_accuracy": float(acc),
            "val_macro_f1": float(f1_score(y["val"], y_pred, average="macro")),
            "complejidad_params": int(model.coef_.size + model.intercept_.size),
            "complejidad_unidad": "parametros entrenables",
            "search_time_s": 0.0, "fit_time_s": fit_seconds, "tiempo_s": fit_seconds,
            "inference_time_s": inference_seconds, "device": "cpu",
        }, y["test"], test_pred, os.path.join(RESULTS_DIR, "logistic_regression_model.pkl"), ARTIFACT_DIR, RESULTS_DIR)
        return model, acc

def run_learning_curves(X, y, model):
    with timer("Curvas de Aprendizaje (Regresion Logistica)"):
        n_sample = min(LEARNING_CURVE_SAMPLE, len(y["train"]))
        # El Parquet esta ordenado temporalmente. Val es siempre el split futuro.
        idx = np.linspace(0, len(y["train"]) - 1, n_sample, dtype=int)
        X_sample, y_sample = X["train"][idx], y["train"][idx]
        train_sizes = np.unique(np.linspace(max(10, n_sample // 10), n_sample, 8, dtype=int))
        records = []
        for size in train_sizes:
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(X_sample[:size], y_sample[:size])
            records.append({"train_rows": int(size),
                "train_accuracy": accuracy_score(y_sample[:size], clf.predict(X_sample[:size])),
                "val_accuracy": accuracy_score(y["val"], clf.predict(X["val"]))})
        pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "learning_curves.csv"), index=False)
        train_mean = np.array([r["train_accuracy"] for r in records])
        val_mean = np.array([r["val_accuracy"] for r in records])
        train_std = np.zeros(len(records))
        val_std = np.zeros(len(records))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="#2196F3")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="#FF5722")
        ax.plot(train_sizes, train_mean, "o-", color="#2196F3", label="Train", linewidth=2)
        ax.plot(train_sizes, val_mean, "o-", color="#FF5722", label="Validacion", linewidth=2)
        ax.set_xlabel("Tamano del conjunto de entrenamiento", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Curvas de Aprendizaje - Regresion Logistica\n"
                     "(evaluacion sobre el mismo split temporal de validacion)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        gap = train_mean[-1] - val_mean[-1]
        ax.annotate(f"Brecha train-val: {gap:.4f}",
                    xy=(train_sizes[-1], val_mean[-1]),
                    xytext=(train_sizes[-3], val_mean[-1] - 0.03),
                    fontsize=10, color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))

        fig.tight_layout()
        save_figure(fig, "03_learning_curves")

        print(f"\n  Train accuracy final: {train_mean[-1]:.4f}")
        print(f"  Val accuracy final:   {val_mean[-1]:.4f}")
        print(f"  Brecha:               {gap:.4f}")


def run_linear_svm(X, y, target_names):
    with timer("Modelo 3: SVM Lineal (LinearSVC)"):
        model = LinearSVC(
            max_iter=5000,
            dual="auto",
            verbose=0,
        )
        model.fit(X["train"], y["train"])

        y_pred = model.predict(X["val"])
        acc = print_metrics(y["val"], y_pred, "SVM Lineal", target_names)

        unique_classes = np.unique(y["train"])

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y["val"], y_pred, labels=unique_classes)
        ConfusionMatrixDisplay(cm, display_labels=price_direction_display_labels(unique_classes)).plot(ax=ax, cmap="Greens", values_format="d")
        format_confusion_axes(ax)
        ax.set_title(f"SVM Lineal - Matriz de Confusion (Acc={acc:.4f})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "04_linear_svm_cm")

        return acc

def run_rbf_svm(X, y, target_names):
    with timer(f"Modelo 4: SVM RBF (muestra de {SVM_RBF_SAMPLE:,} filas)"):
        n_sample = min(SVM_RBF_SAMPLE, len(y["train"]))
        rng = np.random.RandomState(42)
        idx_train = rng.choice(len(y["train"]), n_sample, replace=False)
        idx_val = rng.choice(len(y["val"]), min(n_sample, len(y["val"])), replace=False)

        X_train_s = X["train"][idx_train]
        y_train_s = y["train"][idx_train]
        X_val_s = X["val"][idx_val]
        y_val_s = y["val"][idx_val]

        print(f"    Train sample: {X_train_s.shape}")
        print(f"    Val sample:   {X_val_s.shape}")

        model = SVC(kernel="rbf", gamma="scale", verbose=False)
        model.fit(X_train_s, y_train_s)

        y_pred = model.predict(X_val_s)
        acc = print_metrics(y_val_s, y_pred, "SVM RBF (muestra)", target_names)

        unique_classes = np.unique(y_train_s)

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_val_s, y_pred, labels=unique_classes)
        ConfusionMatrixDisplay(cm, display_labels=price_direction_display_labels(unique_classes)).plot(ax=ax, cmap="Purples", values_format="d")
        format_confusion_axes(ax)
        ax.set_title(f"SVM RBF - Matriz de Confusion (Acc={acc:.4f})\n"
                     f"(Muestra: {n_sample:,} filas)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "05_rbf_svm_cm")

        return acc

def plot_comparison(results: dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    accs = list(results.values())
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    bars = ax.barh(models, accs, color=colors[:len(models)], height=0.5, edgecolor="white",
                   linewidth=1.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Comparacion de Modelos Baseline\n"
                 "Se requiere un modelo no lineal (DNN)?",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(accs) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    ax.axvline(x=1/len(PRICE_DIRECTION_CLASSES), color="red", linestyle="--", alpha=0.5, label="Azar uniforme (1/5 = 20%)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    save_figure(fig, "06_model_comparison")

def plot_error_analysis(X, y, data, model, model_name: str):
    with timer(f"Analisis de errores - {model_name}"):
        y_pred = model.predict(X["val"])
        errors = y["val"] != y_pred

        val_df = data["val"].copy()
        val_df["error"] = errors
        val_df["y_true"] = y["val"]
        val_df["y_pred"] = y_pred

        encoders_path = os.path.join(ARTIFACT_DIR, "encoders.pkl")
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            if "category" in val_df.columns and "category" in encoders:
                le = encoders["category"]
                cat_col = val_df["category"].clip(0, len(le.classes_) - 1)
                val_df["category_name"] = le.inverse_transform(cat_col.astype(int))
                error_by_cat = val_df.groupby("category_name")["error"].mean().sort_values(
                    ascending=False
                )
                error_by_cat.plot(kind="barh", ax=axes[0], color="#e74c3c", edgecolor="white")
                axes[0].set_xlabel("Tasa de Error", fontsize=11)
                axes[0].set_title("Tasa de Error por Categoria", fontsize=12, fontweight="bold")
                axes[0].grid(axis="x", alpha=0.3)

            if "platform" in val_df.columns and "platform" in encoders:
                le = encoders["platform"]
                plat_col = val_df["platform"].clip(0, len(le.classes_) - 1)
                val_df["platform_name"] = le.inverse_transform(plat_col.astype(int))
                error_by_plat = val_df.groupby("platform_name")["error"].mean().sort_values(
                    ascending=False
                )
                error_by_plat.plot(kind="barh", ax=axes[1], color="#3498db", edgecolor="white")
                axes[1].set_xlabel("Tasa de Error", fontsize=11)
                axes[1].set_title("Tasa de Error por Plataforma", fontsize=12, fontweight="bold")
                axes[1].grid(axis="x", alpha=0.3)

            fig.suptitle(f"Analisis de Errores - {model_name}",
                        fontsize=14, fontweight="bold")
            fig.tight_layout()
            save_figure(fig, "07_error_analysis")

        error_rate = errors.mean()
        print(f"\n  Tasa de error global: {error_rate:.4f}")
        print(f"  Errores totales: {errors.sum():,} / {len(errors):,}")

def generate_summary(results: dict):
    print("\n" + "=" * 60)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 60)

    best_model = max(results, key=results.get)
    best_acc = results[best_model]

    print(f"\n  Mejor modelo baseline: {best_model}")
    print(f"  Accuracy: {best_acc:.4f}")

    print(f"\n  Resultados por modelo:")
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(acc * 40)
        print(f"    {model:30s} {acc:.4f}  {bar}")

    linear_models = ["Regresion Lineal", "Regresion Logistica", "SVM Lineal"]
    nonlinear_models = ["SVM RBF"]

    linear_accs = [results[m] for m in linear_models if m in results]
    nonlinear_accs = [results[m] for m in nonlinear_models if m in results]

    if linear_accs and nonlinear_accs:
        best_linear = max(linear_accs)
        best_nonlinear = max(nonlinear_accs)
        improvement = best_nonlinear - best_linear

        print(f"\n  --- Analisis de No-Linealidad ---")
        print(f"  Mejor modelo lineal:     {best_linear:.4f}")
        print(f"  Mejor modelo no lineal:  {best_nonlinear:.4f}")
        print(f"  Mejora:                  {improvement:+.4f}")

        print("    SVM RBF usa una muestra menor: esta diferencia no aisla el efecto del modelo.")

    print(f"\n  Figuras guardadas en: {RESULTS_DIR}")
    print("=" * 60)

def main():
    print("=" * 60)
    print("  MODELOS BASELINE - Clasificacion Multiclase")
    print(f"  Target: {CLASSIFICATION_TARGET} (5 clases de cambio de precio a 7 dias)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with timer("Cargando datos limpios"):
        data = load_clean_data()

    with timer("Preparando features y target"):
        X, y = prepare_xy(data)
        classes = sorted(np.unique(y["train"]).tolist())
        target_names = price_direction_display_labels(classes)
        print(f"    Clases: {target_names}")

    results = {}

    results["Regresion Lineal"] = run_linear_regression(X, y, target_names)

    lr_model, lr_acc = run_logistic_regression(X, y, target_names)
    results["Regresion Logistica"] = lr_acc

    run_learning_curves(X, y, lr_model)

    results["SVM Lineal"] = run_linear_svm(X, y, target_names)

    results["SVM RBF"] = run_rbf_svm(X, y, target_names)

    plot_comparison(results)

    plot_error_analysis(X, y, data, lr_model, "Regresion Logistica")

    generate_summary(results)

    results_df = pd.DataFrame([
        {"Modelo": k, "Accuracy": v} for k, v in results.items()
    ]).sort_values("Accuracy", ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    print(f"\n  Resultados guardados en: {os.path.join(RESULTS_DIR, 'baseline_results.csv')}")

if __name__ == "__main__":
    main()
