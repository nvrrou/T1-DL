"""Grafico de complejidad de baseline, DNN, AutoML y NAS."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import dnn_config as cfg


MODELS = {
    "baseline": "Baseline\nLogistica",
    "dnn": "DNN\nmanual",
    "automl": "AutoML\nCatBoost",
    "nas": "NAS\nOptuna",
}


def load_model_parameters():
    manifest_path = os.path.join(cfg.ARTIFACT_DIR, "dataset_manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        run_id = json.load(handle)["run_id"]

    rows = []
    for key, display_name in MODELS.items():
        path = os.path.join(cfg.RESULTS_DIR, f"metrics_{key}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Falta {path}. Ejecuta el modelo {key} antes de crear el grafico."
            )
        with open(path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        if metrics.get("run_id") != run_id:
            raise ValueError(
                f"{key}: el resultado pertenece a otro procesamiento. Vuelve a entrenarlo."
            )
        complexity = metrics.get("complejidad_params")
        if not isinstance(complexity, (int, float)) or complexity <= 0:
            raise ValueError(f"{key}: complejidad invalida: {complexity!r}")
        unit = metrics.get("complejidad_unidad", "n/d")
        class_count = len(metrics.get("classes", []))
        if unit == "hojas de arbol":
            if class_count < 2:
                raise ValueError(f"{key}: no se pudo determinar el numero de clases.")
            learned_values = int(complexity * class_count)
            detail = f"{int(complexity):,} hojas x {class_count} clases"
            equivalent_unit = "valores de hoja"
        else:
            learned_values = int(complexity)
            detail = f"{int(complexity):,} parametros"
            equivalent_unit = unit
        rows.append({
            "modelo": display_name,
            "valores_aprendidos": learned_values,
            "detalle": detail,
            "unidad": equivalent_unit,
            "tamano_mb": float(metrics["model_size_mb"]),
        })
    return pd.DataFrame(rows)


def add_value_labels(ax, values, formatter):
    for bar, value in zip(ax.patches, values):
        ax.annotate(
            formatter(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_parameters(df):
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(df["modelo"], df["valores_aprendidos"], color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Valores aprendidos (escala logaritmica)")
    axes[0].set_title("Parametros o valores numericos aprendidos")
    axes[0].grid(axis="y", alpha=0.25)
    add_value_labels(
        axes[0], df["valores_aprendidos"], lambda value: f"{int(value):,}"
    )

    axes[1].bar(df["modelo"], df["tamano_mb"], color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MB (escala logaritmica)")
    axes[1].set_title("Tamano del modelo guardado")
    axes[1].grid(axis="y", alpha=0.25)
    add_value_labels(axes[1], df["tamano_mb"], lambda value: f"{value:.3f} MB")

    units = " | ".join(
        f"{name.replace(chr(10), ' ')}: {detail}"
        for name, detail in zip(df["modelo"], df["detalle"])
    )
    fig.text(
        0.5,
        0.01,
        units,
        ha="center",
        fontsize=9,
    )
    fig.suptitle("Parametros y complejidad de las cuatro soluciones", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(cfg.RESULTS_DIR, "15_parametros_modelos.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    df = load_model_parameters()
    csv_path = os.path.join(cfg.RESULTS_DIR, "parametros_modelos.csv")
    df.assign(modelo=df["modelo"].str.replace("\n", " ", regex=False)).to_csv(
        csv_path, index=False
    )
    output_path = plot_parameters(df)
    print(df.to_string(index=False))
    print(f"Grafico guardado: {output_path}")
    print(f"Datos guardados: {csv_path}")
    print("CatBoost: valores aprendidos = hojas x numero de clases.")
    print("La equivalencia es aproximada porque la estructura del arbol tambien forma parte del modelo.")


if __name__ == "__main__":
    main()
