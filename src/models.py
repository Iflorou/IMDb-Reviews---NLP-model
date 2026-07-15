"""Model architectures for IMDb sentiment classification."""

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from src.config import EMBEDDING_DIM, MAX_VOCAB


def build_dense_model() -> tf.keras.Model:
    """Baseline: Embedding + GlobalAveragePooling1D. Expects pre-vectorized int input."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=MAX_VOCAB, output_dim=EMBEDDING_DIM),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_dense_v2_model(vectorizer: TextVectorization) -> tf.keras.Model:
    """Same as the baseline but with the vectorizer and dropout baked into the model,
    so it accepts raw text input directly."""
    model = tf.keras.Sequential(
        [
            vectorizer,
            tf.keras.layers.Embedding(input_dim=MAX_VOCAB, output_dim=EMBEDDING_DIM),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model() -> tf.keras.Model:
    """Sequence model: Embedding + Bidirectional LSTM. Expects pre-vectorized int input."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=MAX_VOCAB, output_dim=EMBEDDING_DIM),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
