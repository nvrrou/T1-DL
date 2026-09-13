import os
import sys
import time
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive", "data")

SPLIT_FILES = {
    "train": os.path.join(DATA_DIR, "splits", "ecommerce_train.csv"),
    "val":   os.path.join(DATA_DIR, "splits", "ecommerce_val.csv"),
    "test":  os.path.join(DATA_DIR, "splits", "ecommerce_test.csv"),
}

from target_config import CLASSIFICATION_TARGET, price_direction_labels
CHUNK_SIZE = 500_000

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
            print(f"  ✓ Completado en {elapsed:.1f}s")
    return Timer()

def streaming_eda(filepath: str, name: str):
    print(f"\n  --- EDA: {name} ---")
    print(f"  Archivo: {filepath}")

    total_rows = 0
    dtype_counts = None
    columns = None
    null_counts = None

    target_dist = {}

    welford = {}
    numeric_cols = None

    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk[CLASSIFICATION_TARGET] = price_direction_labels(chunk["price"], chunk["target_price_7d"])
        n_chunk = len(chunk)
        total_rows += n_chunk

        if columns is None:
            columns = chunk.columns.tolist()
            dtype_counts = chunk.dtypes.value_counts()
            null_counts = chunk.isnull().sum()
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(chunk[c])]
            for col in numeric_cols:
                welford[col] = {
                    "n": 0, "mean": 0.0, "M2": 0.0,
                    "col_min": float("inf"), "col_max": float("-inf"),
                }
        else:
            null_counts = null_counts.add(chunk.isnull().sum(), fill_value=0)

        if CLASSIFICATION_TARGET in chunk.columns:
            vc = chunk[CLASSIFICATION_TARGET].value_counts()
            for val, count in vc.items():
                target_dist[val] = target_dist.get(val, 0) + count

        for col in numeric_cols:
            series = chunk[col].dropna()
            if len(series) == 0:
                continue
            state = welford[col]

            vals = series.values.astype(np.float64)
            batch_n = len(vals)
            batch_mean = vals.mean()
            batch_var = vals.var(ddof=0) if batch_n > 1 else 0.0

            n_a = state["n"]
            n_b = batch_n
            n_ab = n_a + n_b

            if n_a == 0:
                state["mean"] = batch_mean
                state["M2"] = batch_var * batch_n
            else:
                delta = batch_mean - state["mean"]
                state["mean"] = (n_a * state["mean"] + n_b * batch_mean) / n_ab
                state["M2"] += batch_var * n_b + delta**2 * n_a * n_b / n_ab

            state["n"] = n_ab
            state["col_min"] = min(state["col_min"], float(vals.min()))
            state["col_max"] = max(state["col_max"], float(vals.max()))

        print(f"    Chunk {i+1}: {total_rows:,} filas procesadas", end="\r")

    print(f"    Total: {total_rows:,} filas procesadas                    ")

    print(f"\n  Shape: ({total_rows:,}, {len(columns)})")
    print(f"  Dtypes:\n{dtype_counts.to_string()}")

    null_pct = (null_counts / total_rows * 100).round(2)
    null_df = pd.DataFrame({"nulos": null_counts.astype(int), "%": null_pct})
    null_df = null_df[null_df["nulos"] > 0].sort_values("nulos", ascending=False)
    if len(null_df) > 0:
        print(f"\n  Columnas con nulos:")
        print(null_df.to_string())
    else:
        print(f"\n  ✓ Sin valores nulos")

    if target_dist:
        print(f"\n  Distribución de '{CLASSIFICATION_TARGET}':")
        total_target = sum(target_dist.values())
        for val in sorted(target_dist.keys(), key=lambda x: target_dist[x], reverse=True):
            pct = target_dist[val] / total_target * 100
            print(f"    {val}: {pct:.2f}% ({target_dist[val]:,})")

    print(f"\n  Estadísticas numéricas (resumen):")
    stats_rows = []
    for col in numeric_cols:
        s = welford[col]
        if s["n"] == 0:
            continue
        variance = s["M2"] / s["n"] if s["n"] > 0 else 0.0
        stats_rows.append({
            "col": col,
            "count": int(s["n"]),
            "mean": round(s["mean"], 4),
            "std": round(math.sqrt(variance), 4),
            "min": round(s["col_min"], 4),
            "max": round(s["col_max"], 4),
        })
    desc = pd.DataFrame(stats_rows).set_index("col")
    print(desc.head(15).to_string())
    if len(desc) > 15:
        print(f"    ... y {len(desc)-15} columnas numéricas más")

    print(f"\n  Columnas ({len(columns)}):")
    for col in columns:
        print(f"    - {col}")

    return total_rows

def main():
    print("=" * 60)
    print("  SCRIPT DE EDA / PROFILING")
    print("  Dataset: E-Commerce Price Tracker")
    print("  ⚠ Solo inspección — no guarda datos procesados")
    print("=" * 60)

    total_all = 0
    for split_name, filepath in SPLIT_FILES.items():
        with timer(f"EDA: {split_name}"):
            if not os.path.exists(filepath):
                print(f"    ⚠ Archivo no encontrado: {filepath}")
                continue
            n = streaming_eda(filepath, split_name)
            total_all += n

    print(f"\n{'='*60}")
    print(f"  Total filas en todos los splits: {total_all:,}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
