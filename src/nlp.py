# Topic classification using TF-IDF and Logistic Regression.
# This module mirrors Upscore's Insight product which
# helps editorial teams understand what topics perform best.

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def build_topic_model(df, text_col="title_clean", label_col="category",
                      max_features=10000, test_size=0.2, random_state=42):
    """
    Train a TF-IDF topic classifier and log to MLflow.

    Args:
        df          : dataframe with text and labels
        text_col    : column with cleaned article titles
        label_col   : column with category labels
        max_features: vocabulary size for TF-IDF
        test_size   : fraction of data for testing
        random_state: for reproducibility

    Returns:
        tfidf vectorizer, trained model, accuracy score
    """

    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # build TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # train and log to MLflow
    with mlflow.start_run(run_name="topic-classifier"):

        model = LogisticRegression(
            C=1.0, max_iter=1000, multi_class="multinomial"
        )
        model.fit(X_train_tfidf, y_train)

        accuracy = accuracy_score(
            y_test, model.predict(X_test_tfidf)
        )

        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "topic_model")

    print(f"[NLP] Topic classifier accuracy: {accuracy:.4f}")
    return tfidf, model, accuracy


def classify_article(title, tfidf, model, category_map):
    """
    Classify a single article title into a topic category.

    Args:
        title       : raw article title string
        tfidf       : fitted TF-IDF vectorizer
        model       : trained classifier
        category_map: dict mapping codes to full names

    Returns:
        predicted category name as string
    """

    from src.utils import clean_title
    cleaned = clean_title(title)
    vector  = tfidf.transform([cleaned])
    code    = model.predict(vector)[0]
    return category_map.get(code, code)