"""Evidencia reproducible de cada ejecucion, sin interpretar los resultados."""
import json
import os
import platform
import uuid
from importlib.metadata import version

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, classification_report, f1_score

from target_config import (
    CLASSIFICATION_TARGET,
    PRICE_DIRECTION_CLASSES,
    NEUTRAL_CHANGE_PCT,
    price_direction_display_labels,
)


def write_dataset_manifest(paths, feature_names, artifact_dir):
    if len(feature_names) < 15:
        raise ValueError(f"La rubrica exige al menos 15 features; encontradas {len(feature_names)}.")
    splits = {}
    for split, path in paths.items():
        counts = {}
        for batch in pq.ParquetFile(path).iter_batches(columns=[CLASSIFICATION_TARGET]):
            for label, count in batch.column(0).to_pandas().value_counts().items():
                counts[str(label)] = counts.get(str(label), 0) + int(count)
        if set(counts) != set(PRICE_DIRECTION_CLASSES):
            raise ValueError(f"{split}: se requieren las tres clases; encontradas {counts}")
        splits[split] = {"rows": sum(counts.values()), "classes": counts}
    metadata_path = os.path.join(os.path.dirname(artifact_dir), "split_metadata.json")
    with open(metadata_path, encoding="utf-8") as f:
        temporal = json.load(f)
    manifest = {"run_id": str(uuid.uuid4()), "target": CLASSIFICATION_TARGET,
                "neutral_change_pct": NEUTRAL_CHANGE_PCT,
                "feature_count": len(feature_names), "features": feature_names,
                "splits": splits, "temporal": temporal}
    with open(os.path.join(artifact_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def save_metrics(key, metrics, y_true, y_pred, model_path, artifact_dir, results_dir):
    with open(os.path.join(artifact_dir, "dataset_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    def names(values):
        a = np.asarray(values)
        return np.asarray(PRICE_DIRECTION_CLASSES)[a.astype(int)] if a.dtype.kind in "iu" else a
    yt, yp = names(y_true), names(y_pred)
    counts = manifest["splits"]["train"]["classes"]
    majority = max(counts, key=counts.get)
    metrics.update({"run_id": manifest["run_id"], "classes": list(PRICE_DIRECTION_CLASSES),
                    "feature_count": manifest["feature_count"],
                    "train_rows": manifest["splits"]["train"]["rows"],
                    "val_rows": manifest["splits"]["val"]["rows"], "test_rows": len(yt),
                    "test_accuracy": float(accuracy_score(yt, yp)),
                    "test_macro_f1": float(f1_score(yt, yp, labels=PRICE_DIRECTION_CLASSES, average="macro", zero_division=0)),
                    "majority_test_accuracy": float(np.mean(yt == majority)),
                    "uniform_random_accuracy": 1 / len(PRICE_DIRECTION_CLASSES),
                    "model_size_mb": os.path.getsize(model_path) / 1024**2,
                    "hardware": platform.platform(), "processor": platform.processor(),
                    "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "torch", "flaml", "optuna")}})
    metrics["generalizacion_gap"] = metrics["train_accuracy"] - metrics["test_accuracy"]
    report = classification_report(yt, yp, labels=PRICE_DIRECTION_CLASSES, output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report).T
    report_frame.rename(
        index=dict(zip(PRICE_DIRECTION_CLASSES, price_direction_display_labels(PRICE_DIRECTION_CLASSES))),
        inplace=True,
    )
    report_frame.to_csv(os.path.join(results_dir, f"report_{key}.csv"))
    with open(os.path.join(results_dir, f"metrics_{key}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics
