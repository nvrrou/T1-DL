import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, f1_score

import dnn_config as cfg
from dnn_data import (
    load_schema,
    scan_classes_and_weights,
    ParquetStream,
    load_full,
)
from dnn_model import TabularDNN
from dnn_plots import plot_convergence, plot_confusion
from experiment_results import save_metrics
from target_config import price_direction_display_labels


# ============================================================
# SEED / GPU
# ============================================================

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.set_float32_matmul_precision("high")


def timer(msg):
    class T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")
            return self

        def __exit__(self, *a):
            print(f"  OK en {time.time() - self.t0:.1f}s")

    return T()


def loss_denominator(criterion, yb):
    if criterion.weight is not None:
        return criterion.weight[yb].sum().item()
    return len(yb)


# ============================================================
# EVALUACION
# ============================================================

@torch.no_grad()
def evaluate(model, Xn, Xc, y, criterion, chunk=8192):
    model.eval()

    n = len(y)
    total_loss = 0.0
    loss_weight = 0.0
    correct = 0
    preds = torch.empty(n, dtype=torch.long)

    amp_enabled = cfg.USE_AMP and cfg.DEVICE.type == "cuda"

    for s in range(0, n, chunk):
        e = min(s + chunk, n)

        xn = Xn[s:e].to(
            cfg.DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )
        xc = Xc[s:e].to(
            cfg.DEVICE,
            dtype=torch.long,
            non_blocking=True,
        )
        yb = y[s:e].to(
            cfg.DEVICE,
            dtype=torch.long,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=cfg.DEVICE.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(xn, xc)
            loss = criterion(logits, yb)

        denominator = loss_denominator(criterion, yb)

        total_loss += loss.item() * denominator
        loss_weight += denominator

        p = logits.argmax(1)

        correct += (p == yb).sum().item()
        preds[s:e] = p.cpu()

    return (
        total_loss / loss_weight,
        correct / n,
        preds,
    )


# ============================================================
# TRAIN
# ============================================================

def train():
    print("=" * 60)
    print("  PASO 2: DNN mejorada - Clasificacion multiclase")
    print(f"  Dispositivo: {cfg.DEVICE}")

    if cfg.DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"  Mixed precision: {cfg.USE_AMP}")
    print("=" * 60)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.ARTIFACT_DIR, exist_ok=True)

    for split, path in cfg.INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"  ERROR: falta {path}")
            print("  Ejecuta antes 02_feature_engineering.py")
            return

    # ========================================================
    # DATOS
    # ========================================================

    num_cols, cat_cols, cat_cards = load_schema(cfg.ARTIFACT_DIR)

    (
        classes,
        class_to_idx,
        counts,
        total,
        weights,
    ) = scan_classes_and_weights(
        cfg.INPUT_FILES["train"]
    )

    n_classes = len(classes)

    print(
        f"\n  Numericas: {len(num_cols)}"
        f" | Categoricas: {len(cat_cols)}"
        f" {list(zip(cat_cols, cat_cards))}"
    )

    print(f"  Clases: {classes}")

    for c in classes:
        idx = class_to_idx[c]

        print(
            f"    {c}: "
            f"{counts[c] / total * 100:.2f}% "
            f"({counts[c]:,}) "
            f"peso={weights[idx]:.3f}"
        )

    # ========================================================
    # VAL / TEST
    # ========================================================

    with timer("Cargando val y test en memoria"):
        Xn_val, Xc_val, y_val = load_full(
            cfg.INPUT_FILES["val"],
            num_cols,
            cat_cols,
            class_to_idx,
        )

        Xn_test, Xc_test, y_test = load_full(
            cfg.INPUT_FILES["test"],
            num_cols,
            cat_cols,
            class_to_idx,
        )

        print(
            f"    val: {tuple(Xn_val.shape)}"
            f" | test: {tuple(Xn_test.shape)}"
        )

    # ========================================================
    # MODELO
    # ========================================================

    model = TabularDNN(
        len(num_cols),
        cat_cards,
        cfg.HIDDEN_LAYERS,
        n_classes,
        cfg.DROPOUT,
    ).to(cfg.DEVICE)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\n  Arquitectura: "
        f"{len(num_cols)}(num)+emb "
        f"-> {cfg.HIDDEN_LAYERS} "
        f"-> {n_classes}"
    )

    print(f"  Dropout: {cfg.DROPOUT}")
    print(f"  Parametros entrenables: {n_params:,}")

    # ========================================================
    # LOSS
    # ========================================================

    if cfg.USE_CLASS_WEIGHTS:
        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=cfg.DEVICE,
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print(
            f"  Class weights: SI "
            f"{class_weights.cpu().numpy()}"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Class weights: NO")

    # ========================================================
    # OPTIMIZER / SCHEDULER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE,
        min_lr=cfg.MIN_LR,
    )

    amp_enabled = (
        cfg.USE_AMP
        and cfg.DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # ========================================================
    # HISTORY / EARLY STOPPING
    # ========================================================

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    fit_start = time.perf_counter()

    with timer("Entrenamiento (streaming por mini-batches)"):

        for epoch in range(1, cfg.EPOCHS + 1):

            model.train()

            stream = ParquetStream(
                cfg.INPUT_FILES["train"],
                num_cols,
                cat_cols,
                class_to_idx,
                shuffle=True,
                max_batches=cfg.MAX_BATCHES_PER_EPOCH,
            )

            run_loss = 0.0
            run_loss_weight = 0.0
            run_correct = 0
            run_n = 0
            nb = 0

            t0 = time.time()

            for xn, xc, yb in stream:

                xn = xn.to(
                    cfg.DEVICE,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                xc = xc.to(
                    cfg.DEVICE,
                    dtype=torch.long,
                    non_blocking=True,
                )

                yb = yb.to(
                    cfg.DEVICE,
                    dtype=torch.long,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # Mixed precision
                with torch.autocast(
                    device_type=cfg.DEVICE.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(xn, xc)
                    loss = criterion(logits, yb)

                # Backprop
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.GRAD_CLIP,
                )

                scaler.step(optimizer)
                scaler.update()

                # Metricas train
                denominator = loss_denominator(
                    criterion,
                    yb,
                )

                run_loss += (
                    loss.item()
                    * denominator
                )

                run_loss_weight += denominator

                run_correct += (
                    logits.argmax(1) == yb
                ).sum().item()

                run_n += yb.size(0)
                nb += 1

                if nb % 200 == 0:
                    print(
                        f"    epoch {epoch} "
                        f"batch {nb}: "
                        f"loss "
                        f"{run_loss / run_loss_weight:.4f}",
                        end="\r",
                    )

            # =================================================
            # METRICAS EPOCH
            # =================================================

            tr_loss = (
                run_loss
                / run_loss_weight
            )

            tr_acc = (
                run_correct
                / run_n
            )

            va_loss, va_acc, va_pred = evaluate(
                model,
                Xn_val,
                Xc_val,
                y_val,
                criterion,
            )

            va_f1 = f1_score(
                y_val.numpy(),
                va_pred.numpy(),
                average="macro",
            )

            # Scheduler ahora maximiza Macro-F1
            scheduler.step(va_f1)

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            history["val_macro_f1"].append(va_f1)
            history["learning_rate"].append(current_lr)

            print(
                f"\n    Epoch {epoch:02d}/{cfg.EPOCHS}"
                f" | train loss {tr_loss:.4f}"
                f" acc {tr_acc:.4f}"
                f" | val loss {va_loss:.4f}"
                f" acc {va_acc:.4f}"
                f" F1 {va_f1:.4f}"
                f" | lr {current_lr:.2e}"
                f" | {time.time() - t0:.0f}s"
                f" | {run_n:,} filas"
            )

            # =================================================
            # MEJOR MODELO SEGUN MACRO F1
            # =================================================

            if va_f1 > best_val_f1 + 1e-4:

                best_val_f1 = va_f1
                best_val_acc = va_acc
                best_epoch = epoch

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v
                    in model.state_dict().items()
                }

                epochs_no_improve = 0

                print(
                    f"      -> Nuevo mejor modelo"
                    f" | F1 {best_val_f1:.4f}"
                    f" | Acc {best_val_acc:.4f}"
                )

            else:
                epochs_no_improve += 1

                print(
                    f"      -> Sin mejora "
                    f"{epochs_no_improve}/{cfg.PATIENCE}"
                )

                if (
                    epochs_no_improve
                    >= cfg.PATIENCE
                ):
                    print(
                        f"    Early stopping "
                        f"en epoch {epoch}"
                    )
                    break

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    # ========================================================
    # GUARDAR HISTORIAL
    # ========================================================

    pd.DataFrame(history).rename_axis(
        "epoch_index"
    ).to_csv(
        os.path.join(
            cfg.RESULTS_DIR,
            "dnn_history.csv",
        )
    )

    # Recuperar mejor epoch, no el ultimo.
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(cfg.DEVICE)

    print(
        f"\n  Mejor epoch: {best_epoch}"
        f" | Val F1: {best_val_f1:.4f}"
        f" | Val Acc: {best_val_acc:.4f}"
    )

    # Mantiene compatibilidad con tu plot actual.
    plot_convergence({
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
    })

    # ========================================================
    # TEST
    # ========================================================

    with timer("Evaluacion en TEST"):

        inference_start = time.perf_counter()

        te_loss, te_acc, y_pred = evaluate(
            model,
            Xn_test,
            Xc_test,
            y_test,
            criterion,
        )

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        y_true = y_test.numpy()
        y_pred = y_pred.numpy()

        test_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )

        print(
            f"    Test loss {te_loss:.4f}"
            f" | Test accuracy {te_acc:.4f}"
            f" | Macro-F1 {test_f1:.4f}"
        )

        print(
            "\n"
            + classification_report(
                y_true,
                y_pred,
                labels=list(range(n_classes)),
                target_names=(
                    price_direction_display_labels(
                        classes
                    )
                ),
                zero_division=0,
            )
        )

        plot_confusion(
            y_true,
            y_pred,
            classes,
        )

    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    ckpt_path = os.path.join(
        cfg.ARTIFACT_DIR,
        "dnn_model.pt",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "cat_cardinalities": cat_cards,
            "hidden": cfg.HIDDEN_LAYERS,
            "dropout": cfg.DROPOUT,
            "classes": classes,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
        },
        ckpt_path,
    )

    print(
        f"\n  Modelo guardado: "
        f"{ckpt_path}"
    )

    # ========================================================
    # TRAIN ACCURACY
    # ========================================================

    correct = 0
    evaluated = 0

    model.eval()

    with torch.no_grad():

        for xn, xc, yb in ParquetStream(
            cfg.INPUT_FILES["train"],
            num_cols,
            cat_cols,
            class_to_idx,
            shuffle=False,
        ):

            xn = xn.to(
                cfg.DEVICE,
                dtype=torch.float32,
            )

            xc = xc.to(
                cfg.DEVICE,
                dtype=torch.long,
            )

            with torch.autocast(
                device_type=cfg.DEVICE.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predicted = model(
                    xn,
                    xc,
                ).argmax(1).cpu()

            correct += int(
                (predicted == yb).sum()
            )

            evaluated += len(yb)

    train_acc = (
        correct
        / evaluated
    )

    # ========================================================
    # VAL FINAL
    # ========================================================

    _, val_acc, val_pred = evaluate(
        model,
        Xn_val,
        Xc_val,
        y_val,
        criterion,
    )

    val_f1 = f1_score(
        y_val.numpy(),
        val_pred.numpy(),
        average="macro",
    )

    # ========================================================
    # METRICAS
    # ========================================================

    save_metrics(
        "dnn",
        {
            "name": "DNN manual mejorada",
            "family": "DNN disenada a mano",
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
            "test_accuracy": float(te_acc),
            "test_macro_f1": float(test_f1),
            "generalizacion_gap": float(
                train_acc - te_acc
            ),
            "complejidad_params": int(n_params),
            "complejidad_unidad": (
                "parametros entrenables"
            ),
            "search_time_s": 0.0,
            "fit_time_s": float(fit_seconds),
            "tiempo_s": float(fit_seconds),
            "inference_time_s": float(
                inference_seconds
            ),
            "epochs": len(
                history["train_loss"]
            ),
            "best_epoch": int(best_epoch),
            "device": (
                torch.cuda.get_device_name(0)
                if cfg.DEVICE.type == "cuda"
                else "cpu"
            ),
            "hidden": cfg.HIDDEN_LAYERS,
            "dropout": cfg.DROPOUT,
            "use_class_weights": (
                cfg.USE_CLASS_WEIGHTS
            ),
            "use_amp": cfg.USE_AMP,
        },
        y_true,
        y_pred,
        ckpt_path,
        cfg.ARTIFACT_DIR,
        cfg.RESULTS_DIR,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        f"\n{'=' * 60}"
        f"\n  RESUMEN DNN"
        f"\n{'=' * 60}"
    )

    print(f"  Mejor epoch:   {best_epoch}")
    print(f"  Train acc:     {train_acc:.4f}")
    print(f"  Val acc:       {val_acc:.4f}")
    print(f"  Val Macro-F1:  {val_f1:.4f}")
    print(f"  Test acc:      {te_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1:.4f}")
    print(f"  Parametros:    {n_params:,}")
    print(f"  Tiempo:        {fit_seconds:.1f}s")

    print("=" * 60)


if __name__ == "__main__":
    train()
