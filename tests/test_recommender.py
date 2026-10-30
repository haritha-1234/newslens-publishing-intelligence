# Tests for the article recommendation engine.
# We use a small fake article dataset so tests
# run quickly without the full news corpus.

import pandas as pd
import pytest
from src.recommender import (
    build_recommender,
    recommend_by_category,
    recommend_for_reader
)
from src.data import simulate_readers


@pytest.fixture
def sample_articles():
    # small fake article dataset for testing
    return pd.DataFrame({
        "title": [
            "Stock market rises on good earnings",
            "Federal reserve cuts interest rates",
            "Business profits hit record high",
            "Apple launches new AI powered iPhone",
            "Google releases new search algorithm",
            "Scientists discover new cancer treatment",
            "Hospital study shows diet improves health",
            "Taylor Swift breaks streaming record",
            "New movie tops box office worldwide",
            "Tech startup raises million in funding"
        ],
        "title_clean": [
            "stock market rises good earnings",
            "federal reserve cuts interest rates",
            "business profits hit record high",
            "apple launches new ai powered iphone",
            "google releases new search algorithm",
            "scientists discover new cancer treatment",
            "hospital study shows diet improves health",
            "taylor swift breaks streaming record",
            "new movie tops box office worldwide",
            "tech startup raises million funding"
        ],
        "category": ["b", "b", "b", "t", "t", "m", "m", "e", "e", "t"],
        "category_name": [
            "Business", "Business", "Business",
            "Science & Tech", "Science & Tech",
            "Health", "Health",
            "Entertainment", "Entertainment",
            "Science & Tech"
        ],
        "publisher": ["Reuters"] * 10
    })


def test_build_recommender_returns_two_objects(sample_articles):
    # should return tfidf vectorizer and similarity matrix
    tfidf, sim_matrix = build_recommender(
        sample_articles, max_features=50
    )
    assert tfidf is not None
    assert sim_matrix is not None


def test_similarity_matrix_shape(sample_articles):
    # similarity matrix should be square n x n
    tfidf, sim_matrix = build_recommender(
        sample_articles, max_features=50
    )
    n = len(sample_articles)
    assert sim_matrix.shape == (n, n)


def test_similarity_matrix_diagonal_is_one(sample_articles):
    # every article is perfectly similar to itself
    tfidf, sim_matrix = build_recommender(
        sample_articles, max_features=50
    )
    import numpy as np
    diagonal = sim_matrix.diagonal()
    assert all(abs(v - 1.0) < 0.01 for v in diagonal)


def test_recommend_by_category_returns_list(sample_articles):
    # should return a list of article titles
    tfidf, sim_matrix = build_recommender(
        sample_articles, max_features=50
    )
    recs = recommend_by_category(sample_articles, sim_matrix, "b", n=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2


def test_recommend_by_category_wrong_code(sample_articles):
    # unknown category code should return empty list
    tfidf, sim_matrix = build_recommender(
        sample_articles, max_features=50
    )
    recs = recommend_by_category(
        sample_articles, sim_matrix, "z", n=2
    )
    assert recs == []


def test_recommend_for_reader_returns_dict(sample_articles):
    # should return a dictionary of category to titles
    tfidf, sim_matrix = build_recommender(
        sample_articles, max_features=50
    )
    readers = simulate_readers(n=10)
    reader = readers.iloc[0]
    recs = recommend_for_reader(reader, sample_articles, sim_matrix)
    assert isinstance(recs, dict)