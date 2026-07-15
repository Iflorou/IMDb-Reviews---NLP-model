"""CLI entrypoint: wires data -> pipeline -> model -> fit -> evaluate.

Run from the src/ directory as:
    python train.py --model dense_v2
"""

import argparse

from config import EPOCHS
from data import load_train_test
from evaluate import evaluate_model, plot_confusion_matrix, plot_history
from models import build_dense_model, build_dense_v2_model, build_lstm_model
from pipeline import build_vectorizer, make_datasets, vectorize_datasets

MODEL_CHOICES = ("dense", "dense_v2", "lstm")


def train(model_name: str):
    train_df, test_df = load_train_test()
    train_ds, test_ds = make_datasets(train_df, test_df)
    vectorizer = build_vectorizer(train_ds)

    if model_name == "dense":
        train_vec, test_vec = vectorize_datasets(train_ds, test_ds, vectorizer)
        model = build_dense_model()
        history = model.fit(train_vec, validation_data=test_vec, epochs=EPOCHS)
        evaluate_model(model, train_vec, test_vec, name="Dense")
        plot_history(history, name="Dense")
    elif model_name == "dense_v2":
        model = build_dense_v2_model(vectorizer)
        history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
        evaluate_model(model, train_ds, test_ds, name="Dense V2")
        plot_history(history, name="Dense V2")
        plot_confusion_matrix(model, test_ds, name="Dense V2")
    elif model_name == "lstm":
        train_vec, test_vec = vectorize_datasets(train_ds, test_ds, vectorizer)
        model = build_lstm_model()
        history = model.fit(train_vec, validation_data=test_vec, epochs=EPOCHS)
        evaluate_model(model, train_vec, test_vec, name="LSTM")
        plot_history(history, name="LSTM")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train an IMDb sentiment model")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="dense_v2")
    args = parser.parse_args()
    train(args.model)


if __name__ == "__main__":
    main()
