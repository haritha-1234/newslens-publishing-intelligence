# Tests for the NLP topic classification module.
# We create a small fake dataset so tests run fast
# without needing the full 422K article dataset.

import pandas as pd
import pytest
from src.nlp import build_topic_model, classify_article
from src.utils import CATEGORY_MAP


@pytest.fixture
def sample_news_df():
    # two example headlines per category, repeated 5 times so there
    # are 40 rows. This is enough for the 80/20 stratified split
    # to put at least one article of every category in the test set.
    base = pd.DataFrame({
        "title_clean": [
            "stock market rises federal reserve rate",
            "apple iphone new technology launch",
            "celebrity awards show hollywood film",
            "cancer treatment health study results",
            "business profits quarterly earnings report",
            "software ai machine learning model",
            "music concert singer album release",
            "hospital patient medical research trial"
        ],
        "category": ["b", "t", "e", "m", "b", "t", "e", "m"]
    })
    return pd.concat([base] * 5, ignore_index=True)


def test_build_topic_model_returns_three_objects(sample_news_df):
    # build_topic_model should return tfidf, model, accuracy
    tfidf, model, accuracy = build_topic_model(
        sample_news_df, max_features=100
    )
    assert tfidf is not None
    assert model is not None
    assert isinstance(accuracy, float)


def test_topic_model_accuracy_in_range(sample_news_df):
    # accuracy should always be between 0 and 1
    tfidf, model, accuracy = build_topic_model(
        sample_news_df, max_features=100
    )
    assert 0.0 <= accuracy <= 1.0


def test_classify_article_returns_valid_category(sample_news_df):
    # the predicted category should be one of our known categories
    tfidf, model, accuracy = build_topic_model(
        sample_news_df, max_features=100
    )
    result = classify_article(
        "stock market crash", tfidf, model, CATEGORY_MAP
    )
    assert result in CATEGORY_MAP.values()


def test_classify_article_output_is_string(sample_news_df):
    # the output should always be a string
    tfidf, model, accuracy = build_topic_model(
        sample_news_df, max_features=100
    )
    result = classify_article(
        "new phone launch", tfidf, model, CATEGORY_MAP
    )
    assert isinstance(result, str)