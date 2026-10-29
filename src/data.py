# Handles loading and preparing the news dataset
# and simulating reader behaviour data.
# In a real production system the reader data would
# come from a database or analytics platform like
# the one Upscore provides to publishers.

import pandas as pd
import numpy as np
from src.utils import clean_title


def load_news(path, sample_per_category=12500, random_state=42):
    """
    Load cleaned news data and return a balanced sample.

    Args:
        path                : path to news_clean.csv
        sample_per_category : how many articles per category
        random_state        : for reproducibility

    Returns:
        sampled dataframe with clean titles
    """

    df = pd.read_csv(path)
    df["title_clean"] = df["title"].apply(clean_title)

    # balanced sample so no category dominates
    df_sample = df.groupby("category").apply(
        lambda x: x.sample(
            min(len(x), sample_per_category),
            random_state=random_state
        )
    ).reset_index(drop=True)

    print(f"[Data] Loaded {len(df_sample):,} articles")
    return df_sample


def load_rec_articles(path, sample_size=5000, random_state=42):
    """
    Load a sample of articles for the recommender.

    Args:
        path        : path to news_clean.csv
        sample_size : number of articles to use
        random_state: for reproducibility

    Returns:
        sampled dataframe with clean titles
    """

    df = pd.read_csv(path)
    df["title_clean"] = df["title"].apply(clean_title)

    df_sample = df.sample(
        sample_size, random_state=random_state
    ).reset_index(drop=True)

    print(f"[Data] Loaded {len(df_sample):,} articles for recommender")
    return df_sample


def simulate_readers(n=5000, random_state=42):
    """
    Simulate reader behaviour data for churn modelling.
    Features mirror real engagement metrics used by
    publishing platforms like Upscore.

    Args:
        n           : number of readers to simulate
        random_state: for reproducibility

    Returns:
        dataframe of reader profiles with churn label
    """

    np.random.seed(random_state)

    articles_per_week    = np.random.poisson(lam=7, size=n)
    avg_session_minutes  = np.random.exponential(scale=8, size=n)
    days_since_last      = np.random.exponential(scale=5, size=n)
    weeks_subscribed     = np.random.randint(1, 104, size=n)
    topic_diversity      = np.random.randint(1, 5, size=n)
    newsletter_opens     = np.random.poisson(lam=3, size=n)
    uses_app             = np.random.binomial(1, 0.4, size=n)

    churn_score = (
        - 0.3 * articles_per_week
        - 0.2 * avg_session_minutes
        + 0.4 * days_since_last
        - 0.1 * weeks_subscribed
        - 0.1 * newsletter_opens
        + np.random.normal(0, 1, size=n)
    )

    churn = (
        churn_score > np.percentile(churn_score, 70)
    ).astype(int)

    df = pd.DataFrame({
        "reader_id"            : [f"reader_{i}" for i in range(n)],
        "articles_per_week"    : articles_per_week,
        "avg_session_minutes"  : avg_session_minutes.round(1),
        "days_since_last_visit": days_since_last.round(1),
        "weeks_subscribed"     : weeks_subscribed,
        "topic_diversity"      : topic_diversity,
        "newsletter_opens"     : newsletter_opens,
        "uses_app"             : uses_app,
        "churn"                : churn
    })

    print(f"[Data] Simulated {len(df):,} reader profiles")
    print(f"[Data] Churn rate: {churn.mean():.1%}")
    return df