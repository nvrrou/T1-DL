import os
import time
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

import dnn_config as cfg
from dnn_data import load_schema, scan_classes_and_weights, ParquetStream, load_full
from dnn_model import TabularDNN
from dnn_plots import plot_convergence, plot_confusion
from experiment_results import save_metrics
from target_config import price_direction_display_labels

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

def timer(msg):
    class T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*60}\n  {msg}\n{'='*60}")
            return self
        def __exit__(self, *a):
            print(f"  OK en {time.time()-self.t0:.1f}s")
    return T()

@torch.no_grad()
def evaluate(model, Xn, Xc, y, criterion, chunk=8192):
    model.eval()
    n = len(y)
    total_loss, correct, loss_weight = 0.0, 0, 0.0
    preds = torch.empty(n, dtype=torch.long)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xn = Xn[s:e].to(cfg.DEVICE)
        xc = Xc[s:e].to(cfg.DEVICE)
        yb = y[s:e].to(cfg.DEVICE)
        logits = model(xn, xc)
        denominator = criterion.weight[yb].sum().item() if criterion.weight is not None else len(yb)
        total_loss += criterion(logits, yb).item() * denominator
        loss_weight += denominator
        p = logits.argmax(1)
        correct += (p == yb).sum().item()
        preds[s:e] = p.cpu()
    return total_loss / loss_weight, correct / n, preds

def train():
    print("=" * 60)
    print("  PASO 2: DNN (PyTorch) - Clasificacion multiclase")
    print(f"  Dispositivo: {cfg.DEVICE}")
    print("=" * 60)

    os.makedirs(cfg.SYNTHETIC_RESULTS_DIR, exist_ok=True)
    for split, path in cfg.SYNTHETIC_INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"  ERROR: falta {path}. Ejecuta antes 02_feature_engineering.py")
            return

    num_cols, cat_cols, cat_cards = load_schema(cfg.SYNTHETIC_ARTIFACT_DIR)
    classes, class_to_idx, counts, total, weights = scan_classes_and_weights(cfg.SYNTHETIC_INPUT_FILES["train"])
    n_classes = len(classes)
    print(f"\n  Numericas: {len(num_cols)} | Categoricas: {len(cat_cols)} {list(zip(cat_cols, cat_cards))}")
    print(f"  Clases {classes} - distribucion:")
    for c in classes:
        print(f"    {c}: {counts[c]/total*100:.2f}% ({counts[c]:,})  peso={weights[class_to_idx[c]]:.3f}")

    with timer("Cargando val y test en memoria"):
        Xn_val, Xc_val, y_val = load_full(cfg.SYNTHETIC_INPUT_FILES["val"], num_cols, cat_cols, class_to_idx)
        Xn_test, Xc_test, y_test = load_full(cfg.SYNTHETIC_INPUT_FILES["test"], num_cols, cat_cols, class_to_idx)
        print(f"    val: {tuple(Xn_val.shape)}  test: {tuple(Xn_test.shape)}")

    model = TabularDNN(len(num_cols), cat_cards, cfg.HIDDEN_LAYERS, n_classes, cfg.DROPOUT).to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Arquitectura: {len(num_cols)}(num)+emb -> {cfg.HIDDEN_LAYERS} -> {n_classes}")
    print(f"  Parametros entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(cfg.DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_state, epochs_no_improve = float("inf"), None, 0

    fit_start = time.perf_counter()
    with timer("Entrenamiento (streaming por mini-batches)"):
        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            stream = ParquetStream(cfg.SYNTHETIC_INPUT_FILES["train"], num_cols, cat_cols, class_to_idx,
                                   shuffle=True, max_batches=cfg.MAX_BATCHES_PER_EPOCH)
            run_loss, run_correct, run_n, nb = 0.0, 0, 0, 0
            run_loss_weight = 0.0
            t0 = time.time()
            for xn, xc, yb in stream:
                xn, xc, yb = xn.to(cfg.DEVICE), xc.to(cfg.DEVICE), yb.to(cfg.DEVICE)
                optimizer.zero_grad()
                logits = model(xn, xc)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                bs = yb.size(0)
                denominator = criterion.weight[yb].sum().item()
                run_loss += loss.item() * denominator
                run_loss_weight += denominator
                run_correct += (logits.argmax(1) == yb).sum().item()
                run_n += bs
                nb += 1
                if nb % 200 == 0:
                    print(f"    epoch {epoch} batch {nb}: loss {run_loss/run_loss_weight:.4f}", end="\r")

            tr_loss, tr_acc = run_loss / run_loss_weight, run_correct / run_n
            va_loss, va_acc, _ = evaluate(model, Xn_val, Xc_val, y_val, criterion)
            scheduler.step(va_loss)
            history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
            print(f"    Epoch {epoch:02d}/{cfg.EPOCHS} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
                  f"{time.time()-t0:.0f}s | {run_n:,} filas")

            if va_loss < best_val - 1e-4:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.PATIENCE:
                    print(f"    Early stopping en epoch {epoch} (sin mejora en {cfg.PATIENCE} epocas)")
                    break

    fit_seconds = time.perf_counter() - fit_start
    pd.DataFrame(history).rename_axis("epoch_index").to_csv(os.path.join(cfg.SYNTHETIC_RESULTS_DIR, "dnn_history.csv"))
    if best_state is not None:
        model.load_state_dict(best_state)

    plot_convergence(history)

    with timer("Evaluacion en TEST"):
        inference_start = time.perf_counter()
        te_loss, te_acc, y_pred = evaluate(model, Xn_test, Xc_test, y_test, criterion)
        inference_seconds = time.perf_counter() - inference_start
        y_true = y_test.numpy()
        y_pred = y_pred.numpy()
        print(f"    Test loss {te_loss:.4f} | Test accuracy {te_acc:.4f}")
        print("\n" + classification_report(y_true, y_pred,
              labels=list(range(n_classes)), target_names=price_direction_display_labels(classes), zero_division=0))
        plot_confusion(y_true, y_pred, classes)

    ckpt_path = os.path.join(cfg.SYNTHETIC_ARTIFACT_DIR, "dnn_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "num_cols": num_cols, "cat_cols": cat_cols, "cat_cardinalities": cat_cards,
        "hidden": cfg.HIDDEN_LAYERS, "dropout": cfg.DROPOUT, "classes": classes,
    }, ckpt_path)
    print(f"\n  Modelo guardado: {ckpt_path}")
    correct, evaluated = 0, 0
    model.eval()
    with torch.no_grad():
        for xn, xc, yb in ParquetStream(cfg.SYNTHETIC_INPUT_FILES["train"], num_cols, cat_cols, class_to_idx, shuffle=False):
            predicted = model(xn.to(cfg.DEVICE), xc.to(cfg.DEVICE)).argmax(1).cpu()
            correct += int((predicted == yb).sum())
            evaluated += len(yb)
    _, val_acc, val_pred = evaluate(model, Xn_val, Xc_val, y_val, criterion)
    save_metrics("dnn", {
        "name": "DNN manual", "family": "DNN disenada a mano",
        "train_accuracy": correct / evaluated, "val_accuracy": float(val_acc),
        "val_macro_f1": float(f1_score(y_val.numpy(), val_pred.numpy(), average="macro")),
        "complejidad_params": int(n_params), "search_time_s": 0.0,
        "complejidad_unidad": "parametros entrenables",
        "fit_time_s": fit_seconds, "tiempo_s": fit_seconds,
        "inference_time_s": inference_seconds, "epochs": len(history["train_loss"]),
        "device": torch.cuda.get_device_name() if cfg.DEVICE.type == "cuda" else "cpu",
        "hidden": cfg.HIDDEN_LAYERS, "dropout": cfg.DROPOUT,
    }, y_true, y_pred, ckpt_path, cfg.SYNTHETIC_ARTIFACT_DIR, cfg.SYNTHETIC_RESULTS_DIR)

    print(f"\n{'='*60}\n  RESUMEN DNN\n{'='*60}")
    print(f"  Test accuracy: {te_acc:.4f}")
    print(f"  Parametros: {n_params:,}")
    print(f"  Curvas: {os.path.join(cfg.SYNTHETIC_RESULTS_DIR, '08_dnn_convergencia.png')}")
    print("  Comparar con result_sintetico/baseline_results.csv (03_baselines.py)")
    print(f"{'='*60}")

if __name__ == "__main__":
    train()
