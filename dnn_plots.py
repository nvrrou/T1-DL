import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from dnn_config import SYNTHETIC_RESULTS_DIR
from target_config import price_direction_display_labels


def format_confusion_axes(ax):
    ax.set_xlabel("Predicha", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def plot_convergence(history, filename="08_dnn_convergencia.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"], "o-", label="Validacion", color="#FF5722")
    axes[0].set_title("Convergencia - Perdida (CrossEntropy)", fontweight="bold")
    axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train", color="#2196F3")
    axes[1].plot(epochs, history["val_acc"], "o-", label="Validacion", color="#FF5722")
    axes[1].set_title("Convergencia - Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("DNN - Curvas de convergencia | Datos sinteticos", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(SYNTHETIC_RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"    Guardado: {path}")

def plot_confusion(y_true, y_pred, classes, filename="09_dnn_confusion.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    ConfusionMatrixDisplay(cm, display_labels=price_direction_display_labels(classes)).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    format_confusion_axes(ax)
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"DNN - Matriz de confusion - Datos sinteticos (Test, Acc={acc:.4f})", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(SYNTHETIC_RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"    Guardado: {path}")
