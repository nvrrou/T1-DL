import os
import sys
import time
import json

import duckdb
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive", "data", "sintetico")

SPLIT_FILES = {
    "synthetic": os.path.join(DATA_DIR, "train_real.csv"),
}

CLEAN_DIR = os.path.join(DATA_DIR, "clean")
REAL_CLEAN_DIR = os.path.join(BASE_DIR, "archive", "data", "clean")

DROP_COLS = ["product_id", "title", "brand", "asin", "timestamp"]

from target_config import CLASSIFICATION_TARGET

DATASET_ROWS = 3_000_000
TRAIN_PCT = 80
VAL_PCT = 10
TEST_PCT = 10
HORIZON_DAYS = 7

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

def main():
    print("=" * 60)
    print("  SCRIPT DE LIMPIEZA DE DATOS (DuckDB out-of-core)")
    print("  Dataset: E-Commerce Price Tracker - DATOS SINTETICOS")
    print("=" * 60)

    # Los holdouts originales son de solo lectura.
    for split in ("val", "test"):
        path = os.path.join(REAL_CLEAN_DIR, f"{split}_features.parquet")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    with open(os.path.join(REAL_CLEAN_DIR, "split_metadata.json"), encoding="utf-8") as f:
        real_metadata = json.load(f)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for name, path in SPLIT_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠ Archivo no encontrado: {path}")
            sys.exit(1)
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  {name}: {path} ({size_gb:.2f} GB)")

    db_path = os.path.join(CLEAN_DIR, "_temp_cleaning.duckdb")
    con = duckdb.connect(db_path)

    try:
        con.execute("SET memory_limit = '4GB';")
        con.execute("SET threads TO 4;")
        con.execute(f"SET temp_directory = '{CLEAN_DIR}';")
        con.execute("SET preserve_insertion_order = false;")

        with timer("Registrando CSVs en DuckDB"):
            csv_paths = [path.replace("\\", "/") for path in SPLIT_FILES.values()]
            csv_list = ", ".join(f"'{p}'" for p in csv_paths)
            con.execute(f"""
                CREATE OR REPLACE VIEW raw_data AS
                SELECT * FROM read_csv_auto([{csv_list}],
                    header=true,
                    ignore_errors=true
                );
            """)
            total = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
            print(f"    Total filas de entrenamiento sintetico: {total:,}")

        with timer("Inspeccionando columnas"):
            cols_info = con.execute("DESCRIBE raw_data").fetchall()
            all_cols = [row[0] for row in cols_info]
            keep_cols = [c for c in all_cols if c not in DROP_COLS]
            print(f"    Columnas originales: {len(all_cols)}")
            print(f"    Columnas eliminadas: {[c for c in DROP_COLS if c in all_cols]}")
            print(f"    Columnas conservadas: {len(keep_cols)}")

        with timer("Limpieza: filtrar precios invalidos para el target + deduplicar (DISTINCT)"):
            select_cols = ", ".join(f'"{c}"' for c in keep_cols)

            con.execute(f"""
                CREATE OR REPLACE TABLE clean_data AS
                SELECT DISTINCT {select_cols},
                    TRY_CAST(timestamp AS TIMESTAMP) AS _observed_at,
                    product_id AS _product_id
                FROM raw_data
                WHERE price > 0 AND isfinite(price)
                  AND target_price_7d >= 0 AND isfinite(target_price_7d)
                  AND TRY_CAST(timestamp AS TIMESTAMP) IS NOT NULL;
            """)

            n_clean = con.execute("SELECT COUNT(*) FROM clean_data").fetchone()[0]
            n_removed = total - n_clean
            print(f"    Filas después de limpieza: {n_clean:,}")
            print(f"    Filas eliminadas (nulos + duplicados): {n_removed:,}")

        with timer("Seleccion de train sintetico; val y test reales se conservan"):
            con.execute(f"""
                CREATE OR REPLACE TABLE partitioned AS
                SELECT *, 'train' AS _split FROM clean_data
                ORDER BY hash(_product_id, platform, _observed_at),
                         _product_id, platform, _observed_at
                LIMIT {DATASET_ROWS};
            """)
            n_train, start, end = con.execute(
                "SELECT COUNT(*), MIN(_observed_at), MAX(_observed_at) FROM partitioned"
            ).fetchone()
            if not n_train:
                raise ValueError("El entrenamiento sintetico quedo vacio.")
            metadata = {
                "mode": "synthetic_train_real_holdouts",
                "source_files": SPLIT_FILES,
                "source_rows": total, "clean_rows": n_clean,
                "selected_rows": n_train,
                "splits": {"train": {"rows": n_train, "start": str(start),
                                     "end": str(end), "source": "synthetic"}},
            }
            for split in ("val", "test"):
                path = os.path.join(REAL_CLEAN_DIR, f"{split}_features.parquet")
                metadata["splits"][split] = {
                    **real_metadata["splits"][split],
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "source": "real", "path": path,
                }
            with open(os.path.join(CLEAN_DIR, "split_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"    Train sintetico: {n_train:,} filas; sin nueva division 80/10/10")

        with timer("Exportando a Parquet (float32, categorías en texto)"):
            col_types = con.execute("DESCRIBE clean_data").fetchall()
            cast_exprs = []
            for col_name, col_type, *_ in col_types:
                if col_name.startswith("_"):
                    continue
                if col_name in ("price", "target_price_7d"):
                    cast_exprs.append(f'CAST("{col_name}" AS DOUBLE) AS "{col_name}"')
                    continue
                if "DOUBLE" in col_type.upper() or "FLOAT" in col_type.upper():
                    cast_exprs.append(f'CAST("{col_name}" AS FLOAT) AS "{col_name}"')
                elif "BIGINT" in col_type.upper():
                    cast_exprs.append(f'CAST("{col_name}" AS INTEGER) AS "{col_name}"')
                else:
                    cast_exprs.append(f'"{col_name}"')
            select_cast = ", ".join(cast_exprs)

            for split in ["train"]:
                outpath = os.path.join(CLEAN_DIR, f"{split}.parquet").replace("\\", "/")
                con.execute(f"""
                    COPY (
                        SELECT {select_cast}
                        FROM partitioned
                        WHERE _split = '{split}'
                        ORDER BY _observed_at, _product_id, platform
                    )
                    TO '{outpath}'
                    (FORMAT PARQUET, COMPRESSION 'ZSTD');
                """)
                size_mb = os.path.getsize(outpath.replace("/", os.sep)) / (1024**2)
                n_rows = con.execute(f"""
                    SELECT COUNT(*) FROM partitioned WHERE _split = '{split}'
                """).fetchone()[0]
                print(f"    {split}: {n_rows:,} filas → {size_mb:.1f} MB")

    finally:
        con.close()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                wal_path = db_path + ".wal"
                if os.path.exists(wal_path):
                    os.remove(wal_path)
            except OSError:
                pass

    with timer("Verificación de salida"):
        for split in ["train"]:
            pf = pq.ParquetFile(os.path.join(CLEAN_DIR, f"{split}.parquet"))
            meta = pf.metadata
            schema = pf.schema_arrow
            print(f"    {split}: {meta.num_rows:,} filas × {meta.num_columns} cols")
            col_names = list(schema.names)
            for drop_col in DROP_COLS:
                assert drop_col not in col_names, f"Columna {drop_col} no debería existir"
            print(f"      ✓ Columnas prohibidas ausentes")

            float_cols = [schema.field(i) for i in range(len(schema))
                         if "float" in str(schema.field(i).type).lower()]
            if float_cols:
                sample_type = str(float_cols[0].type)
                print(f"      ✓ Tipo float: {sample_type}")

    print(f"\n{'='*60}")
    print("  ✓ LIMPIEZA COMPLETADA")
    print(f"{'='*60}")
    print(f"  Archivos guardados en: {CLEAN_DIR}")
    print(f"  Siguiente paso: python 02_feature_engineering.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
