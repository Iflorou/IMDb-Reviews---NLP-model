"""tf.data pipeline construction and text vectorization."""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from config import BATCH_SIZE, MAX_VOCAB


def make_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_df["review"], train_df["sentiment"])
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_df["review"], test_df["sentiment"])
    )

    train_ds = (
        train_ds.shuffle(buffer_size=len(train_df))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds


def build_vectorizer(train_ds: tf.data.Dataset) -> TextVectorization:
    vectorizer = TextVectorization(max_tokens=MAX_VOCAB, output_mode="int")
    vectorizer.adapt(train_ds.map(lambda text, label: text))
    return vectorizer


def vectorize_datasets(
    train_ds: tf.data.Dataset, test_ds: tf.data.Dataset, vectorizer: TextVectorization
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    def vectorize(text, label):
        return vectorizer(text), label

    return train_ds.map(vectorize), test_ds.map(vectorize)
