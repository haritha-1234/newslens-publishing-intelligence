# Tests for the data loading and simulation module.
# We test with small samples so tests run fast
# and do not depend on the full dataset being present.

import pandas as pd
import numpy as np
import pytest
from src.data import simulate_readers
from src.utils import clean_title


def test_simulate_readers_shape():
    # check we get the right number of rows and columns
    df = simulate_readers(n=100)
    assert len(df) == 100
    assert "reader_id" in df.columns
    assert "churn" in df.columns


def test_simulate_readers_churn_is_binary():
    # churn column should only contain 0 and 1
    df = simulate_readers(n=100)
    assert df["churn"].isin([0, 1]).all()


def test_simulate_readers_no_nulls():
    # no missing values allowed in reader profiles
    df = simulate_readers(n=100)
    assert df.isnull().sum().sum() == 0


def test_simulate_readers_churn_rate():
    # churn rate should be roughly 30% with large sample
    df = simulate_readers(n=2000)
    churn_rate = df["churn"].mean()
    assert 0.20 < churn_rate < 0.40


def test_clean_title_lowercase():
    # output should always be lowercase
    result = clean_title("Apple LAUNCHES New iPhone")
    assert result == result.lower()


def test_clean_title_removes_special_chars():
    # numbers and punctuation should be removed
    result = clean_title("iPhone 15 Pro: Best phone of 2024!")
    assert all(c.isalpha() or c.isspace() for c in result)


def test_clean_title_strips_whitespace():
    # no leading or trailing whitespace
    result = clean_title("  Apple news  ")
    assert result == result.strip()