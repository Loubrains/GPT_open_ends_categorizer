"""
Utilities for handling and processing text data.

Functions:
    `preprocess_text`: Preprocesses input text by converting to lowercase, normalizing whitespace, and removing special characters.
    `get_random_sample_from_series`: Retrieves a random sample of specified size from a pandas Series.
    `create_batches`: Yields consecutive batches of data from a list.
    `load_csv_to_dict`: Loads a CSV file with 'key' and 'value' columns into a dictionary.
    `load_csv_to_dict_of_lists`: Loads a CSV file with 'key' and 'value' columns into a dictionary, where the 'value' column contains lists in string representation.
    `export_dataframe_to_csv`: Exports a pandas DataFrame to a CSV file.
    `export_dict_to_csv`: Exports a dictionary to a CSV file with 'key' and 'value' columns.
"""

from pathlib import Path
import pandas as pd
import re
import random
import ast
from typing import Any
from pandas._libs.missing import NAType
import sys


def preprocess_text(text: Any) -> str | NAType:
    """
    Preprocesses the input text by converting it to lowercase, normalizing whitespace,
    and removing special characters.

    Args:
        text (Any): The text to preprocess. If text value is NaN, returns pd.NA.

    Returns:
        str | NAType: The preprocessed text as a string, or pd.NA if the input was pd.NA.
    """

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
    """
    Retrieves a random sample of specified size from a pandas Series.

    Args:
        series (pd.Series): The series from which to sample.
        sample_size (int): The number of items to retrieve.

    Returns:
        pd.Series: A Series containing the randomly sampled items.

    Raises:
        ValueError: If the sample_size is greater than the length of the series.
    """

    if sample_size > len(series):
        raise ValueError("Sample size n cannot be greater than the length of the series")
    return series.sample(sample_size, random_state=random.randint(1, 10000))


def create_batches(data: list[str], batch_size: int = 3):
    """
    Yields consecutive batches of data from the list.

    Args:
        data (list[str]): The list of data to be batched.
        batch_size (int): The size of each batch. Defaults to 3.

    Yields:
        list[str]: A batch of data.
    """

    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def load_csv_to_dict(file_path: Path) -> dict:
    """
    Loads a CSV file into a dictionary. Expects the CSV to have two columns named 'key' and 'value'.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary whose keys and values come from the loaded CSV's 'key' and 'value' columns.

    Raises:
        Prints an error message and exits the program if the CSV cannot be read or other errors occur.
    """

    try:
        df = pd.read_csv(file_path)
        return dict(zip(df["key"], df["value"]))

    except Exception as e:
        print(f"\nError while reading CSV: {e}")
        sys.exit(1)


def load_csv_to_dict_of_lists(file_path: Path) -> dict:
    """
    Loads a CSV file into a dictionary. Expects the CSV to have two columns named 'key' and 'value',
    where 'value' is a string representation of a list.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary whose keys and values come from the loaded CSV's 'key' and 'value' columns (the latter converted to lists).

    Raises:
        Prints an error message and exits the program if the CSV cannot be read or other errors occur.
    """

    try:
        df = pd.read_csv(file_path)
        df["value"] = df["value"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        return dict(zip(df["key"], df["value"]))

    except Exception as e:
        print(f"\nError while reading CSV: {e}")
        sys.exit(1)


def export_dataframe_to_csv(file_path: Path, export_df: pd.DataFrame, header: bool = True) -> None:
    """
    Exports a pandas DataFrame to a CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
        export_df (pd.DataFrame): The DataFrame to export.
        header (bool): If True, include the header row in the CSV. Defaults to True.

    Raises:
        Prints an error message if the DataFrame is empty or if there's an error during file writing.
    """

    try:
        if export_df.empty:
            raise pd.errors.EmptyDataError

        export_df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")


def export_dict_to_csv(file_path: Path, export_dict: dict, header: bool = True) -> None:
    """
    Exports a dictionary to a CSV file with 'key' and 'value' columns.

    Args:
        file_path (str): The path where the CSV file will be saved.
        export_dict (dict): The dictionary to export.
        header (bool): If True, include the header row in the CSV. Defaults to True.

    Raises:
        Prints an error message if the dictionary is empty or if there's an error during file writing.
    """

    try:
        if not export_dict:
            raise ValueError("Data is empty")

        df = pd.DataFrame(list(export_dict.items()), columns=["key", "value"])
        df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")
