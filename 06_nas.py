import os
import sys
import json
import time
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import dnn_config as cfg
from dnn_data import load_schema, scan_classes_and_weights, load_full, sample_parquet
from dnn_model import TabularDNN, emb_dim
from experiment_results import save_metrics
from target_config import price_direction_display_labels

NAS_TRAIN_SAMPLE = 0  # 0: usar el mismo train completo que las otras soluciones.
N_TRIALS = 25
NAS_EPOCHS_PER_TRIAL = 4
NAS_FINAL_EPOCHS = 15
SEED = cfg.SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_sample(path, num_cols, cat_cols, class_to_idx, n):
    df = sample_parquet(path, num_cols + cat_cols + [cfg.CLASSIFICATION_TARGET], n, seed=SEED)
    Xn = torch.tensor(df[num_cols].to_numpy(dtype=np.float32))
    Xc = torch.tensor(df[cat_cols].to_numpy(dtype=np.int64))
    y = torch.tensor(df[cfg.CLASSIFICATION_TARGET].astype(str).map(class_to_idx).to_numpy(dtype=np.int64))
    return Xn, Xc, y

@torch.no_grad()
def predict(model, Xn, Xc, chunk=16384):
    model.eval()
    out = torch.empty(len(Xn), dtype=torch.long)
    for s in range(0, len(Xn), chunk):
        e = min(s + chunk, len(Xn))
        logits = model(Xn[s:e].to(cfg.DEVICE), Xc[s:e].to(cfg.DEVICE))
        out[s:e] = logits.argmax(1).cpu()
    return out.numpy()

def train_model(hidden, dropout, lr, wd, bs, epochs, data, weights,
                cat_cards, n_num, n_classes, trial=None):
    Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va = data
    model = TabularDNN(n_num, cat_cards, hidden, n_classes, dropout).to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=weights)
    n = len(y_tr)
    y_va_np = y_va.numpy()
    val_f1 = 0.0
    best_f1, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xn = Xn_tr[idx].to(cfg.DEVICE); xc = Xc_tr[idx].to(cfg.DEVICE); yb = y_tr[idx].to(cfg.DEVICE)
            opt.zero_grad()
            loss = crit(model(xn, xc), yb)
            loss.backward()
            opt.step()
        val_f1 = f1_score(y_va_np, predict(model, Xn_va, Xc_va), average="macro")
        history.append({"epoch": epoch + 1, "val_macro_f1": float(val_f1)})
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    model.training_history = history
    return model, best_f1

def main():
    print("=" * 60)
    print("  PASO 3 (2.2): NAS con Optuna")
    print(f"  Dispositivo: {cfg.DEVICE}")
    print("=" * 60)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    num_cols, cat_cols, cat_cards = load_schema()
    classes, class_to_idx, counts, total, w = scan_classes_and_weights(cfg.INPUT_FILES["train"])
    n_classes, n_num = len(classes), len(num_cols)
    weights = torch.from_numpy(w).to(cfg.DEVICE)

    print("\n  Cargando el train comun y validacion temporal...")
    Xn_tr, Xc_tr, y_tr = load_sample(cfg.INPUT_FILES["train"], num_cols, cat_cols, class_to_idx, NAS_TRAIN_SAMPLE)
    Xn_va, Xc_va, y_va = load_full(cfg.INPUT_FILES["val"], num_cols, cat_cols, class_to_idx)
    data = (Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va)
    print(f"    train muestra: {tuple(Xn_tr.shape)}  val: {tuple(Xn_va.shape)}")

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden = [trial.suggest_categorical(f"units_l{i}", [64, 128, 256, 512]) for i in range(n_layers)]
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
        _, val_f1 = train_model(hidden, dropout, lr, wd, bs, NAS_EPOCHS_PER_TRIAL,
                                data, weights, cat_cards, n_num, n_classes, trial)
        return val_f1

    print(f"\n  Buscando {N_TRIALS} arquitecturas ({NAS_EPOCHS_PER_TRIAL} epocas c/u)...")
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=SEED),
                                pruner=MedianPruner(n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_time = time.time() - t0

    study.trials_dataframe().to_csv(os.path.join(cfg.RESULTS_DIR, "nas_trials.csv"), index=False)
    best = study.best_params
    n_layers = best["n_layers"]
    best_hidden = [best[f"units_l{i}"] for i in range(n_layers)]
    print(f"\n  Mejor arquitectura: capas={best_hidden} dropout={best['dropout']:.3f} "
          f"lr={best['lr']:.1e} bs={best['batch_size']}")
    print(f"  Mejor val macro-F1 en busqueda: {study.best_value:.4f}")

    print(f"\n  Reentrenando la mejor arquitectura ({NAS_FINAL_EPOCHS} epocas)...")
    final_start = time.perf_counter()
    model, final_val_f1 = train_model(best_hidden, best["dropout"], best["lr"], best["weight_decay"],
                           best["batch_size"], NAS_FINAL_EPOCHS, data, weights,
                           cat_cards, n_num, n_classes, trial=None)

    final_seconds = time.perf_counter() - final_start
    pd.DataFrame(model.training_history).to_csv(os.path.join(cfg.RESULTS_DIR, "nas_final_history.csv"), index=False)
    Xn_te, Xc_te, y_te = load_full(cfg.INPUT_FILES["test"], num_cols, cat_cols, class_to_idx)
    y_true = y_te.numpy()
    inference_start = time.perf_counter()
    y_pred = predict(model, Xn_te, Xc_te)
    inference_seconds = time.perf_counter() - inference_start
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    train_acc = accuracy_score(y_tr.numpy(), predict(model, Xn_tr, Xc_tr))
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Test accuracy: {test_acc:.4f} | Test macro-F1: {test_f1:.4f} | params: {n_params:,}")
    print("\n" + classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=price_direction_display_labels(classes), zero_division=0,
    ))

    plot_history(study)
    plot_confusion(y_true, y_pred, classes, test_acc)

    torch.save({"state_dict": model.state_dict(), "hidden": best_hidden,
                "dropout": best["dropout"], "num_cols": num_cols, "cat_cols": cat_cols,
                "cat_cardinalities": cat_cards, "classes": classes},
               os.path.join(cfg.ARTIFACT_DIR, "nas_model.pt"))
    metrics = {
        "name": "NAS (Optuna)",
        "family": "DNN con arquitectura optimizada",
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "train_accuracy": float(train_acc),
        "generalizacion_gap": float(train_acc - test_acc),
        "complejidad_params": int(n_params),
        "complejidad_unidad": "parametros entrenables",
        "tiempo_s": float(search_time + final_seconds),
        "search_time_s": float(search_time), "fit_time_s": final_seconds,
        "inference_time_s": inference_seconds, "val_macro_f1": float(final_val_f1),
        "device": torch.cuda.get_device_name() if cfg.DEVICE.type == "cuda" else "cpu",
        "n_trials": N_TRIALS,
        "best_arch": {"hidden": best_hidden, "dropout": best["dropout"],
                      "lr": best["lr"], "weight_decay": best["weight_decay"],
                      "batch_size": best["batch_size"]},
    }
    save_metrics("nas", metrics, y_true, y_pred,
                 os.path.join(cfg.ARTIFACT_DIR, "nas_model.pt"), cfg.ARTIFACT_DIR, cfg.RESULTS_DIR)

    print(f"\n{'='*60}")
    print(f"  NAS listo | arq {best_hidden} | test acc {test_acc:.4f} | {search_time:.0f}s")
    print(f"  Guardado: results/metrics_nas.json")
    print(f"{'='*60}")

def plot_history(study):
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(vals) + 1), vals, "o", alpha=0.5, label="Trial (val macro-F1)")
    ax.plot(range(1, len(vals) + 1), best_so_far, "-", color="#FF5722", linewidth=2, label="Mejor hasta ahora")
    ax.set_xlabel("Trial"); ax.set_ylabel("Val macro-F1")
    ax.set_title("NAS - Historia de la busqueda (Optuna)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.RESULTS_DIR, "12_nas_historia.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_confusion(y_true, y_pred, classes, acc):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    ConfusionMatrixDisplay(cm, display_labels=price_direction_display_labels(classes)).plot(
        ax=ax, cmap="Purples", values_format="d"
    )
    ax.set_xlabel("Predicha", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax.set_title(f"NAS - Test (Acc={acc:.4f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.RESULTS_DIR, "13_nas_confusion.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    main()
