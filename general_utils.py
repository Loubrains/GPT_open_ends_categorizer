import pandas as pd
import re
import random
from typing import Any
from pandas._libs.missing import NAType
import sys


def preprocess_text(text: Any) -> str | NAType:
    if pd.isna(text):
        return pd.NA

    text = str(text).lower()
    # Convert one or more of any kind of space to single space
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()
    return text


def get_random_sample_from_series(series: pd.Series, sample_size: int) -> pd.Series:
    if sample_size > len(series):
        raise ValueError("Sample size n cannot be greater than the length of the series")
    return series.sample(sample_size, random_state=random.randint(1, 10000))


def load_csv_to_dict(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
        return dict(zip(df["key"], df["value"]))

    except Exception as e:
        print(f"\nError while reading CSV: {e}")
        sys.exit(1)


def load_csv_to_dict_of_lists(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
        df["value"] = df["value"].map(lambda x: x.split(", ") if isinstance(x, str) else [])
        return dict(zip(df["key"], df["value"]))

    except Exception as e:
        print(f"\nError while reading CSV: {e}")
        sys.exit(1)


def export_dataframe_to_csv(file_path: str, export_df: pd.DataFrame, header: bool = True) -> None:
    try:
        if export_df.empty:
            raise pd.errors.EmptyDataError

        export_df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")


def export_dict_to_csv(file_path: str, export_dict: dict, header: bool = True) -> None:
    try:
        if not export_dict:
            raise ValueError("Data is empty")

        df = pd.DataFrame(list(export_dict.items()), columns=["key", "value"])
        df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")


def export_dict_of_lists_to_csv(file_path: str, dict_to_export: dict, header: bool = True) -> None:
    try:
        if not dict_to_export:
            raise ValueError("Data is empty")

        formatted_data = [(key, ", ".join(value)) for key, value in dict_to_export.items()]
        df = pd.DataFrame(formatted_data, columns=["key", "value"])
        df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")
