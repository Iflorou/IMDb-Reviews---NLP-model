"""Evaluation metrics and plots: loss/accuracy curves, confusion matrix, precision/recall/f1."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(
    model: tf.keras.Model, train_ds: tf.data.Dataset, test_ds: tf.data.Dataset, name: str = "Model"
) -> dict:
    print(f"Evaluating {name} on training data:")
    loss_train, accuracy_train = model.evaluate(train_ds)
    print(f"Training Loss: {loss_train:.4f}")
    print(f"Training Accuracy: {accuracy_train:.4f}")

    print(f"\nEvaluating {name} on test data:")
    loss_test, accuracy_test = model.evaluate(test_ds)
    print(f"Test Loss: {loss_test:.4f}")
    print(f"Test Accuracy: {accuracy_test:.4f}")

    return {
        "train_loss": loss_train,
        "train_accuracy": accuracy_train,
        "test_loss": loss_test,
        "test_accuracy": accuracy_test,
    }


def plot_history(history: tf.keras.callbacks.History, name: str = "Model") -> None:
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    has_val = "val_loss" in hist

    fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(12, 5) if has_val else (6, 5))
    ax_loss = axes[0] if has_val else axes

    ax_loss.plot(epochs, hist["loss"], "bo-", label="Training loss")
    if has_val:
        ax_loss.plot(epochs, hist["val_loss"], "ro-", label="Validation loss")
    ax_loss.set_title(f"{name} Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    if has_val and "accuracy" in hist:
        ax_acc = axes[1]
        ax_acc.plot(epochs, hist["accuracy"], "bo-", label="Training accuracy")
        ax_acc.plot(epochs, hist["val_accuracy"], "ro-", label="Validation accuracy")
        ax_acc.set_title(f"{name} Accuracy")
        ax_acc.set_xlabel("Epochs")
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    model: tf.keras.Model, test_ds: tf.data.Dataset, name: str = "Model"
) -> dict:
    y_pred_probabilities = model.predict(test_ds)
    y_pred = (y_pred_probabilities > 0.5).astype(int)
    y_true = np.concatenate([y for _, y in test_ds], axis=0)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({name})")
    plt.show()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Precision ({name}): {precision:.4f}")
    print(f"Recall ({name}): {recall:.4f}")
    print(f"F1-Score ({name}): {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1}
