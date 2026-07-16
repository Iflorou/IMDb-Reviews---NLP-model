"""Dataset download, loading, and train/test splitting."""

import os
import re

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET_FILENAME, DATASET_SLUG, RANDOM_STATE, TEST_SIZE

SENTIMENT_MAP = {"negative": 0, "positive": 1}

# Strips stray C0/C1 control characters (keeping \t \n \r) left over from
# mojibake in the source CSV. Left in place, they end up in the
# TextVectorization vocabulary and break saving the model on Windows,
# whose default file encoding (cp1252) can't represent them.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def download_dataset() -> str:
    return kagglehub.dataset_download(DATASET_SLUG)


def load_dataframe(dataset_path: str) -> pd.DataFrame:
    csv_path = os.path.join(dataset_path, DATASET_FILENAME)
    df = pd.read_csv(csv_path)
    df["review"] = df["review"].str.replace(_CONTROL_CHARS_RE, "", regex=True)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["sentiment"] = train_df["sentiment"].map(SENTIMENT_MAP)
    test_df["sentiment"] = test_df["sentiment"].map(SENTIMENT_MAP)
    return train_df, test_df


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = download_dataset()
    df = load_dataframe(dataset_path)
    return split_dataset(df)
