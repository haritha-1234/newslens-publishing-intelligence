# Tests for the churn prediction module.
# We simulate a small reader dataset so tests
# are fast and self-contained.

import pandas as pd
import pytest
from src.data import simulate_readers
from src.churn import build_churn_model, predict_churn


@pytest.fixture
def small_readers():
    # small reader dataset just for testing
    return simulate_readers(n=200)


def test_build_churn_model_returns_three_objects(small_readers):
    # should return scaler, model, roc_auc
    scaler, model, roc_auc = build_churn_model(small_readers)
    assert scaler is not None
    assert model is not None
    assert isinstance(roc_auc, float)


def test_churn_roc_auc_in_range(small_readers):
    # roc auc should always be between 0 and 1
    scaler, model, roc_auc = build_churn_model(small_readers)
    assert 0.0 <= roc_auc <= 1.0


def test_predict_churn_returns_probability(small_readers):
    # churn probability should be between 0 and 1
    scaler, model, roc_auc = build_churn_model(small_readers)
    reader = small_readers.iloc[0]
    prob = predict_churn(reader, scaler, model)
    assert 0.0 <= prob <= 1.0


def test_predict_churn_high_risk_reader(small_readers):
    # a reader who hasnt visited in a long time
    # and reads nothing should have high churn risk
    scaler, model, _ = build_churn_model(small_readers)

    high_risk = small_readers.iloc[0].copy()
    high_risk["days_since_last_visit"] = 30
    high_risk["articles_per_week"]     = 0
    high_risk["newsletter_opens"]      = 0
    high_risk["avg_session_minutes"]   = 0

    prob = predict_churn(high_risk, scaler, model)
    assert prob > 0.5


def test_predict_churn_low_risk_reader(small_readers):
    # a very engaged reader should have low churn risk
    scaler, model, _ = build_churn_model(small_readers)

    low_risk = small_readers.iloc[0].copy()
    low_risk["days_since_last_visit"] = 0
    low_risk["articles_per_week"]     = 20
    low_risk["newsletter_opens"]      = 10
    low_risk["avg_session_minutes"]   = 30

    prob = predict_churn(low_risk, scaler, model)
    assert prob < 0.5