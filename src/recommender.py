# Article recommendation engine using content-based filtering.
# Recommends articles similar to a reader's interests
# to re-engage readers who are at risk of churning.
# This mirrors Upscore's Engage product.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_recommender(df_articles, text_col="title_clean",
                      max_features=5000):
    """
    Build a TF-IDF similarity matrix for article recommendations.

    Args:
        df_articles : dataframe of articles
        text_col    : column with cleaned titles
        max_features: vocabulary size

    Returns:
        fitted TF-IDF vectorizer, similarity matrix
    """

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix     = tfidf.fit_transform(df_articles[text_col])
    sim_matrix = cosine_similarity(matrix, matrix)

    print(f"[Recommender] Built on {len(df_articles):,} articles")
    print(f"[Recommender] Similarity matrix: {sim_matrix.shape}")

    return tfidf, sim_matrix


def recommend_by_category(df_articles, sim_matrix,
                           category_code, n=5):
    """
    Return the most representative articles from a category.

    Args:
        df_articles    : articles dataframe
        sim_matrix     : precomputed similarity matrix
        category_code  : b, t, e, or m
        n              : number of articles to return

    Returns:
        list of article titles
    """

    cat_df = df_articles[
        df_articles["category"] == category_code
    ].copy()

    if len(cat_df) == 0:
        return []

    indices    = cat_df.index.tolist()
    cat_sim    = sim_matrix[np.ix_(indices, indices)]
    avg_sim    = cat_sim.mean(axis=1)
    cat_df     = cat_df.copy()
    cat_df["score"] = avg_sim

    top = cat_df.nlargest(n, "score")
    return top["title"].tolist()


def recommend_for_reader(reader_row, df_articles,
                          sim_matrix, n=3):
    """
    Recommend articles for a specific reader based on
    their topic diversity score.

    Args:
        reader_row  : pandas Series with reader features
        df_articles : articles dataframe
        sim_matrix  : precomputed similarity matrix
        n           : articles per category

    Returns:
        dict of category name to list of article titles
    """

    # readers with high diversity get a mix of categories
    # readers with low diversity get focused recommendations
    if reader_row["topic_diversity"] >= 3:
        categories = ["b", "t", "e", "m"]
    else:
        categories = ["b", "t"]

    category_map = {
        "b": "Business",
        "t": "Science & Tech",
        "e": "Entertainment",
        "m": "Health"
    }

    recommendations = {}
    for code in categories:
        recs = recommend_by_category(
            df_articles, sim_matrix, code, n=n
        )
        if recs:
            recommendations[category_map[code]] = recs

    return recommendations