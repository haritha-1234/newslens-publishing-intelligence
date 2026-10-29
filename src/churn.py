# Reader churn prediction module.
# Predicts which readers are likely to stop visiting
# the platform so publishers can re-engage them in time.
# This mirrors Upscore's Convert product.

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CHURN_FEATURES = [
    "articles_per_week",
    "avg_session_minutes",
    "days_since_last_visit",
    "weeks_subscribed",
    "topic_diversity",
    "newsletter_opens",
    "uses_app"
]


def build_churn_model(df_readers, test_size=0.2, random_state=42):
    """
    Train a churn prediction model and log to MLflow.

    Args:
        df_readers  : reader profiles dataframe
        test_size   : fraction for testing
        random_state: for reproducibility

    Returns:
        fitted scaler, trained model, roc_auc score
    """

    X = df_readers[CHURN_FEATURES]
    y = df_readers["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    params = {
        "n_estimators": 100,
        "max_depth"   : 4,
        "random_state": random_state
    }

    with mlflow.start_run(run_name="churn-model"):

        model = GradientBoostingClassifier(**params)
        model.fit(X_train_s, y_train)

        roc_auc = roc_auc_score(
            y_test,
            model.predict_proba(X_test_s)[:, 1]
        )

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, "churn_model")

    print(f"[Churn] Model ROC AUC: {roc_auc:.4f}")
    return scaler, model, roc_auc


def predict_churn(reader_row, scaler, model):
    """
    Predict churn probability for a single reader.

    Args:
        reader_row: pandas Series with reader features
        scaler    : fitted StandardScaler
        model     : trained churn model

    Returns:
        churn probability as float between 0 and 1
    """

    features = reader_row[CHURN_FEATURES].values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    return prob