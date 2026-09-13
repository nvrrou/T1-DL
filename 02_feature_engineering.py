import os
import gc
import sys
import time
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder, StandardScaler
from experiment_results import write_dataset_manifest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive", "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
ARTIFACT_DIR = os.path.join(CLEAN_DIR, "artifacts")

INPUT_FILES = {
    "train": os.path.join(CLEAN_DIR, "train.parquet"),
    "val":   os.path.join(CLEAN_DIR, "val.parquet"),
    "test":  os.path.join(CLEAN_DIR, "test.parquet"),
}

OUTPUT_FILES = {
    "train": os.path.join(CLEAN_DIR, "train_features.parquet"),
    "val":   os.path.join(CLEAN_DIR, "val_features.parquet"),
    "test":  os.path.join(CLEAN_DIR, "test_features.parquet"),
}

CAT_COLS = ["category", "subcategory", "platform", "stock_status"]

from target_config import CLASSIFICATION_TARGET, price_direction_labels
REGRESSION_TARGETS = ["target_price_7d", "target_price_30d"]

CHUNK_SIZE = 250_000

PARQUET_COMPRESSION = "ZSTD"

UNKNOWN_TOKEN = "__unknown__"

def timer(msg: str):
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  {msg}")
            print(f"{'='*60}")
            return self

        def __exit__(self, *args):
            print(f"  OK Completado en {time.time() - self.t0:.1f}s")

    return Timer()

def iter_parquet_chunks(path: str, chunk_size: int = CHUNK_SIZE):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()

def infer_column_roles(sample: pd.DataFrame):
    all_cols = sample.columns.tolist()
    targets = [CLASSIFICATION_TARGET] + [c for c in REGRESSION_TARGETS if c in all_cols]

    cat_cols = [c for c in CAT_COLS if c in all_cols]

    num_feature_cols = [
        c for c in all_cols
        if c not in targets
        and c not in cat_cols
        and pd.api.types.is_numeric_dtype(sample[c])
    ]

    roles = {
        "all_cols": all_cols,
        "targets": targets,
        "cat_cols": cat_cols,
        "num_feature_cols": num_feature_cols,
        "reg_targets": [c for c in REGRESSION_TARGETS if c in all_cols],
    }
    return roles

def pass1a_accumulate_stats(train_path: str, roles: dict):
    num_cols = roles["num_feature_cols"]
    cat_cols = roles["cat_cols"]

    num_sum = {c: 0.0 for c in num_cols}
    num_count = {c: 0 for c in num_cols}

    cat_vocab = {c: set() for c in cat_cols}
    cat_counter = {c: Counter() for c in cat_cols}

    total_rows = 0
    for i, chunk in enumerate(iter_parquet_chunks(train_path)):
        total_rows += len(chunk)

        for c in num_cols:
            col = pd.to_numeric(chunk[c], errors="coerce")
            num_sum[c] += float(np.nansum(col.values))
            num_count[c] += int(col.notna().sum())

        for c in cat_cols:
            vc = chunk[c].dropna().astype(str).value_counts()
            cat_vocab[c] = cat_vocab[c].union(vc.index)
            cat_counter[c].update({k: int(v) for k, v in vc.items()})

        print(f"    Chunk {i + 1}: {total_rows:,} filas", end="\r")
        del chunk
        gc.collect()

    print(f"    Total: {total_rows:,} filas procesadas               ")

    numeric_means = {
        c: (num_sum[c] / num_count[c]) if num_count[c] > 0 else 0.0
        for c in num_cols
    }

    categorical_modes = {}
    for c in cat_cols:
        if cat_counter[c]:
            categorical_modes[c] = cat_counter[c].most_common(1)[0][0]
        else:
            categorical_modes[c] = UNKNOWN_TOKEN

    imputers = {
        "numeric_means": numeric_means,
        "categorical_modes": categorical_modes,
    }

    encoders = {}
    for c in cat_cols:
        vocab = sorted(cat_vocab[c])
        le = LabelEncoder()
        le.fit(vocab + [UNKNOWN_TOKEN])
        encoders[c] = le

    return imputers, encoders, total_rows

def impute_chunk(chunk: pd.DataFrame, roles: dict, imputers: dict) -> pd.DataFrame:
    numeric_means = imputers["numeric_means"]
    categorical_modes = imputers["categorical_modes"]

    for c in roles["num_feature_cols"]:
        col = pd.to_numeric(chunk[c], errors="coerce")
        chunk[c] = col.fillna(numeric_means.get(c, 0.0))

    for c in roles["cat_cols"]:
        fill = categorical_modes.get(c, UNKNOWN_TOKEN)
        chunk[c] = chunk[c].astype("object").where(chunk[c].notna(), fill)
        chunk[c] = chunk[c].astype(str)

    return chunk

def pass1b_fit_scaler(train_path: str, roles: dict, imputers: dict) -> StandardScaler:
    num_cols = roles["num_feature_cols"]
    scaler = StandardScaler()

    total_rows = 0
    for i, chunk in enumerate(iter_parquet_chunks(train_path)):
        chunk = impute_chunk(chunk, roles, imputers)
        block = chunk[num_cols].astype(np.float64)
        scaler.partial_fit(block)
        total_rows += len(chunk)
        print(f"    Chunk {i + 1}: {total_rows:,} filas", end="\r")
        del chunk, block
        gc.collect()

    print(f"    Total: {total_rows:,} filas procesadas               ")
    return scaler

def encode_chunk(chunk: pd.DataFrame, roles: dict, encoders: dict) -> pd.DataFrame:
    for c in roles["cat_cols"]:
        le = encoders[c]
        known = set(le.classes_)
        vals = chunk[c].astype(str).map(lambda x, k=known: x if x in k else UNKNOWN_TOKEN)
        chunk[c] = le.transform(vals).astype(np.int16)
    return chunk

def build_final_dtypes(roles: dict) -> dict:
    dtypes = {}
    for c in roles["num_feature_cols"]:
        dtypes[c] = np.float32
    for c in roles["cat_cols"]:
        dtypes[c] = np.int16
    for c in roles["reg_targets"]:
        dtypes[c] = np.float32
    return dtypes

def pass2_transform_write(split: str, in_path: str, out_path: str,
                          roles: dict, scaler: StandardScaler,
                          encoders: dict, imputers: dict):
    num_cols = roles["num_feature_cols"]
    final_dtypes = build_final_dtypes(roles)
    final_cols = (
        num_cols
        + roles["cat_cols"]
        + roles["reg_targets"]
        + ([CLASSIFICATION_TARGET] if CLASSIFICATION_TARGET in roles["all_cols"] else [])
    )

    writer = None
    total_rows = 0
    try:
        for i, chunk in enumerate(iter_parquet_chunks(in_path)):
            # El target se calcula con precios reales; nunca con precios escalados.
            chunk[CLASSIFICATION_TARGET] = price_direction_labels(
                chunk["price"], chunk["target_price_7d"]
            )
            chunk = chunk.loc[chunk[CLASSIFICATION_TARGET].notna()].copy()
            if chunk.empty:
                continue
            chunk = impute_chunk(chunk, roles, imputers)
            chunk = encode_chunk(chunk, roles, encoders)
            block = chunk[num_cols].astype(np.float64)
            chunk[num_cols] = scaler.transform(block).astype(np.float32)
            chunk = chunk[final_cols]
            for c, dt in final_dtypes.items():
                if c in chunk.columns:
                    chunk[c] = chunk[c].astype(dt)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema,
                                          compression=PARQUET_COMPRESSION)
            writer.write_table(table)

            total_rows += len(chunk)
            print(f"    {split} chunk {i + 1}: {total_rows:,} filas", end="\r")

            del chunk, block, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"    {split}: {total_rows:,} filas -> {size_mb:.1f} MB          ")
    return total_rows

def save_artifacts(scaler, encoders, imputers, feature_names):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifacts = {
        "scaler": scaler,
        "encoders": encoders,
        "imputers": imputers,
        "feature_names": feature_names,
    }
    for name, obj in artifacts.items():
        path = os.path.join(ARTIFACT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"    Guardado: {name}.pkl")

def main():
    print("=" * 60)
    print("  FASE 3: FEATURE ENGINEERING & PREPROCESAMIENTO")
    print("  Arquitectura: Two-Pass Streaming (out-of-core)")
    print("=" * 60)

    for split, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"  ERROR: no se encontro {path}")
            print(f"  Ejecuta primero: python 01_limpieza_datos.py")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  {split}: {path} ({size_mb:.1f} MB)")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    first_chunk = next(iter_parquet_chunks(INPUT_FILES["train"]))
    roles = infer_column_roles(first_chunk)
    del first_chunk
    gc.collect()
    print(f"\n  Columnas numericas (features): {len(roles['num_feature_cols'])}")
    print(f"  Columnas categoricas:          {len(roles['cat_cols'])} {roles['cat_cols']}")
    print(f"  Targets:                       {roles['targets']}")

    with timer("PASADA 1a: stats incrementales (imputacion + vocabulario)"):
        imputers, encoders, n_train = pass1a_accumulate_stats(INPUT_FILES["train"], roles)
        print(f"    Medias numericas calculadas: {len(imputers['numeric_means'])}")
        for c in roles["cat_cols"]:
            print(f"    '{c}': {len(encoders[c].classes_)} categorias (incl. {UNKNOWN_TOKEN})")

    with timer("PASADA 1b: StandardScaler.partial_fit bloque a bloque"):
        scaler = pass1b_fit_scaler(INPUT_FILES["train"], roles, imputers)
        print(f"    Scaler ajustado sobre {scaler.n_samples_seen_:,} filas, "
              f"{scaler.n_features_in_} features")

    feature_names = roles["num_feature_cols"] + roles["cat_cols"]

    with timer("Guardando artefactos (.pkl)"):
        save_artifacts(scaler, encoders, imputers, feature_names)

    with timer("PASADA 2: transform & write streaming (ParquetWriter)"):
        for split in ["train", "val", "test"]:
            pass2_transform_write(
                split, INPUT_FILES[split], OUTPUT_FILES[split],
                roles, scaler, encoders, imputers,
            )

    with timer("Verificacion de salida"):
        for split in ["train", "val", "test"]:
            pf = pq.ParquetFile(OUTPUT_FILES[split])
            meta = pf.metadata
            print(f"    {split}: {meta.num_rows:,} filas x {meta.num_columns} cols")
        manifest = write_dataset_manifest(OUTPUT_FILES, feature_names, ARTIFACT_DIR)
        print(f"    Dataset: {manifest['feature_count']} features; distribuciones en artifacts/dataset_manifest.json")

    print(f"\n{'='*60}")
    print("  OK FEATURE ENGINEERING COMPLETADO")
    print(f"{'='*60}")
    print(f"  Salidas en: {CLEAN_DIR}")
    print(f"  Artefactos en: {ARTIFACT_DIR}")
    print(f"  Siguiente paso: python 03_baselines.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
