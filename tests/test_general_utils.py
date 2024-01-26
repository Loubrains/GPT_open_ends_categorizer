import pytest
import pandas as pd
from src.utils import general_utils


@pytest.mark.parametrize(
    "test_data, expected_output",
    [
        # Integers convert to strings
        (1, "1"),
        # Text converted to lowercase
        ("TEST", "test"),
        # Double spaces replaced with single space
        ("test  doublespace", "test doublespace"),
        # Tabs replaced with single space
        ("test\ttab", "test tab"),
        # Multiple occurances of doublespaces/tabs are handled
        ("test  multiple  occurances", "test multiple occurances"),
        # Special characters removed
        ("tes§t, sp#ecialchars!", "test specialchars"),
        # Text trimmed
        (" test trim ", "test trim"),
    ],
)
def test_preprocess_text(test_data, expected_output):
    output = general_utils.preprocess_text(test_data)
    assert output == expected_output


@pytest.mark.parametrize(
    "test_data, sample_length, expect_exception",
    [
        # Happy path
        (pd.Series(["a", "b", "c", "d", "e", "f"]), 3, False),
        # Sample length larger than Series length
        (pd.Series(["a", "b", "c", "d", "e", "f"]), 100, True),
    ],
)
def test_get_random_sample_from_series(test_data, sample_length, expect_exception):
    if expect_exception:
        with pytest.raises(ValueError):
            general_utils.get_random_sample_from_series(test_data, sample_length)

    else:
        sample = general_utils.get_random_sample_from_series(test_data, sample_length)
        assert len(sample) == sample_length
        assert sample.isin(test_data).all()


@pytest.mark.parametrize(
    "test_data, batch_size, expected_output",
    [
        # Batch size divides list size equally
        (["a", "b", "c", "d", "e", "f"], 2, [["a", "b"], ["c", "d"], ["e", "f"]]),
        # Batch size divides list size unequally: final batch size is smaller
        (["a", "b", "c", "d", "e", "f"], 4, [["a", "b", "c", "d"], ["e", "f"]]),
        # Batch size = list size: just one phat batch
        (["a", "b", "c", "d", "e", "f"], 6, [["a", "b", "c", "d", "e", "f"]]),
        # Batch size > list size: just one phat batch
        (["a", "b", "c", "d", "e", "f"], 100, [["a", "b", "c", "d", "e", "f"]]),
        # Empty list: empty batch
        ([], 3, []),
    ],
)
def test_create_batches(test_data, batch_size, expected_output):
    batches = list(general_utils.create_batches(test_data, batch_size))

    assert batches == expected_output


@pytest.mark.parametrize(
    "csv_content, expected_dict, expect_exception",
    [
        # CSV correct format
        ("key,value\na,x\nb,y\nc,z", {"a": "x", "b": "y", "c": "z"}, False),
        # CSV wrong headers
        ("wrong_key,wrong_value\na,x\nb,y\nc,z", None, True),
    ],
)
def test_load_csv_to_dict(monkeypatch, tmp_path, csv_content, expected_dict, expect_exception):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    if expect_exception:
        # Mock sys.exit using monkeypatch
        monkeypatch.setattr("sys.exit", lambda x: (_ for _ in ()).throw(SystemExit(x)))
        with pytest.raises(SystemExit):
            general_utils.load_csv_to_dict(str(csv_file))

    else:
        result_dict = general_utils.load_csv_to_dict(str(csv_file))
        assert result_dict == expected_dict


@pytest.mark.parametrize(
    "csv_content, expected_dict, expect_exception",
    [
        (
            # Correct CSV format
            "key,value\na,\"['w', 'x', 'y']\"\nb,['z']",
            {"a": ["w", "x", "y"], "b": ["z"]},
            False,
        ),
        # CSV value column incorrect format, doesn't contain only lists or stringified lists
        ("key,value\na,'['w', 'x', 'y']'\nb,xyz", None, True),
        # CSV wrong headers
        ("wrong_key,wrong_value\na,x\nb,y\nc,z", None, True),
    ],
)
def test_load_csv_to_dict_of_lists(
    monkeypatch, tmp_path, csv_content, expected_dict, expect_exception
):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    if expect_exception:
        # Mock sys.exit using monkeypatch
        monkeypatch.setattr("sys.exit", lambda x: (_ for _ in ()).throw(SystemExit(x)))
        with pytest.raises(SystemExit):
            general_utils.load_csv_to_dict_of_lists(str(csv_file))

    else:
        result_dict = general_utils.load_csv_to_dict_of_lists(str(csv_file))
        assert result_dict == expected_dict
