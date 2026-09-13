import pickle

import numpy as np
import pyarrow.parquet as pq
import torch

from dnn_config import (
    ARTIFACT_DIR, CAT_COLS, CLASSIFICATION_TARGET, READ_BATCH, BATCH_SIZE,
)

def load_schema():
    import os
    with open(os.path.join(ARTIFACT_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)

    cat_cols = [c for c in CAT_COLS if c in feature_names]
    num_cols = [c for c in feature_names if c not in cat_cols]
    cat_cardinalities = [len(encoders[c].classes_) for c in cat_cols]
    return num_cols, cat_cols, cat_cardinalities

def scan_classes_and_weights(train_path):
    counts = {}
    pf = pq.ParquetFile(train_path)
    for batch in pf.iter_batches(batch_size=READ_BATCH, columns=[CLASSIFICATION_TARGET]):
        s = batch.column(0).to_pandas().astype(str)
        for val, n in s.value_counts().items():
            counts[val] = counts.get(val, 0) + int(n)
    classes = sorted(counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    total = sum(counts.values())
    k = len(classes)
    weights = np.array([total / (k * counts[c]) for c in classes], dtype=np.float32)
    return classes, class_to_idx, counts, total, weights

class ParquetStream(torch.utils.data.IterableDataset):
    def __init__(self, path, num_cols, cat_cols, class_to_idx,
                 batch_size=BATCH_SIZE, read_batch=READ_BATCH,
                 shuffle=True, max_batches=None):
        self.path = path
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.read_batch = read_batch
        self.shuffle = shuffle
        self.max_batches = max_batches
        self.cols = num_cols + cat_cols + [CLASSIFICATION_TARGET]

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        emitted = 0
        for rb in pf.iter_batches(batch_size=self.read_batch, columns=self.cols):
            df = rb.to_pandas()
            Xn = df[self.num_cols].to_numpy(dtype=np.float32)
            Xc = df[self.cat_cols].to_numpy(dtype=np.int64)
            y = df[CLASSIFICATION_TARGET].astype(str).map(self.class_to_idx).to_numpy(dtype=np.int64)

            idx = np.arange(len(df))
            if self.shuffle:
                np.random.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                if start and len(idx) - start == 1:
                    break  # La fila final ya se incluyo en el batch anterior.
                end = min(start + self.batch_size, len(idx))
                if len(idx) - end == 1:
                    end += 1  # BatchNorm necesita al menos dos filas durante train.
                sel = idx[start:end]
                yield (torch.from_numpy(Xn[sel]),
                       torch.from_numpy(Xc[sel]),
                       torch.from_numpy(y[sel]))
                emitted += 1
                if self.max_batches is not None and emitted >= self.max_batches:
                    return

def sample_parquet(path, columns, n, seed=42, read_batch=262_144):
    import pyarrow.parquet as _pq
    pf = _pq.ParquetFile(path)
    total = pf.metadata.num_rows
    if not n or total <= n:
        return _pq.read_table(path, columns=columns).to_pandas()
    frac = n / total
    rng = np.random.default_rng(seed)
    parts = []
    for rb in pf.iter_batches(batch_size=read_batch, columns=columns):
        df = rb.to_pandas()
        mask = rng.random(len(df)) < frac
        if mask.any():
            parts.append(df.loc[mask])
    import pandas as pd
    return pd.concat(parts, ignore_index=True) if parts else _pq.read_table(
        path, columns=columns).to_pandas().head(0)

def load_full(path, num_cols, cat_cols, class_to_idx):
    tbl = pq.read_table(path, columns=num_cols + cat_cols + [CLASSIFICATION_TARGET])
    df = tbl.to_pandas()
    Xn = torch.tensor(df[num_cols].to_numpy(dtype=np.float32))
    Xc = torch.tensor(df[cat_cols].to_numpy(dtype=np.int64))
    y = torch.tensor(df[CLASSIFICATION_TARGET].astype(str).map(class_to_idx).to_numpy(dtype=np.int64))
    return Xn, Xc, y
