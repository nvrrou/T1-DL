import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dnn_config as cfg


def load_results():
    with open(os.path.join(cfg.SYNTHETIC_ARTIFACT_DIR, "dataset_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    rows = []
    for key in ("dnn", "automl", "nas"):
        path = os.path.join(cfg.SYNTHETIC_RESULTS_DIR, f"metrics_{key}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta {path}. Ejecuta 04, 05 y 06 antes de comparar.")
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        if row.get("run_id") != manifest["run_id"]:
            raise ValueError(f"{key}: resultados de otro procesamiento; vuelve a entrenar.")
        for split in ("train", "val", "test"):
            if row[f"{split}_rows"] != manifest["splits"][split]["rows"]:
                raise ValueError(f"{key}: el conjunto {split} no coincide con el experimento.")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load_results()
    columns = ["name", "test_accuracy", "test_macro_f1", "train_accuracy",
               "generalizacion_gap", "complejidad_params", "complejidad_unidad", "model_size_mb",
               "search_time_s", "fit_time_s", "tiempo_s", "inference_time_s",
               "train_rows", "val_rows", "test_rows", "device", "run_id"]
    df[columns].to_csv(os.path.join(cfg.SYNTHETIC_RESULTS_DIR, "comparacion_modelos.csv"), index=False)
    print(df[columns[:-1]].to_string(index=False))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    df.set_index("name")[["test_accuracy", "test_macro_f1"]].plot.bar(ax=axes[0, 0], rot=15)
    axes[0, 0].set_title("Desempeno en el mismo test temporal")
    df.plot.bar(x="name", y="model_size_mb", ax=axes[0, 1], rot=15, legend=False)
    axes[0, 1].set_title("Tamano serializado (MB); no equivale a RAM de entrenamiento")
    df.set_index("name")[["search_time_s", "fit_time_s"]].plot.bar(stacked=True, ax=axes[1, 0], rot=15)
    axes[1, 0].set_title("Costo de busqueda + entrenamiento final (segundos)")
    df.plot.bar(x="name", y="generalizacion_gap", ax=axes[1, 1], rot=15, legend=False)
    axes[1, 1].set_title("Brecha accuracy train - test")
    fig.suptitle("Train sintetico - Validacion y test reales", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(cfg.SYNTHETIC_RESULTS_DIR, "14_comparacion.png"), dpi=150)
    plt.close(fig)
    print("Resultados completos en result_sintetico/comparacion_modelos.csv y 14_comparacion.png")
    print("Comparacion: DNN manual (04), AutoML CatBoost GPU (05) y NAS (06).")
    print("Los tiempos excluyen lectura, preprocesamiento, reportes y evaluacion de train.")
    print("La interpretacion de resultados y las conclusiones se completan en la presentacion.")


if __name__ == "__main__":
    main()
