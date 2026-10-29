# main.py runs the full NewsLens pipeline from the terminal.
# It loads data, trains all models, runs the re-engagement
# pipeline for a sample of readers, and saves all artifacts.
# Run with: python main.py

import os
import pickle
import mlflow

from src.data import load_news, load_rec_articles, simulate_readers
from src.nlp import build_topic_model, classify_article
from src.churn import build_churn_model, predict_churn
from src.recommender import build_recommender, recommend_for_reader
from src.utils import CATEGORY_MAP

NEWS_PATH   = "data/processed/news_clean.csv"
OUTPUT_PATH = "data/processed"


def save_artifact(obj, filename):
    """Save a Python object to disk using pickle."""
    path = os.path.join(OUTPUT_PATH, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Main] Saved {filename}")


if __name__ == "__main__":

    print("="*60)
    print("  NewsLens — Publishing Intelligence Pipeline")
    print("="*60)

    mlflow.set_experiment("newslens")

    # load data
    print("\n--- Loading data ---")
    df_news      = load_news(NEWS_PATH)
    df_rec       = load_rec_articles(NEWS_PATH)
    df_readers   = simulate_readers()

    # topic classification
    print("\n--- Topic Classification ---")
    topic_tfidf, topic_model, topic_acc = build_topic_model(df_news)

    # churn prediction
    print("\n--- Churn Prediction ---")
    scaler, churn_model, churn_auc = build_churn_model(df_readers)

    # recommender
    print("\n--- Building Recommender ---")
    rec_tfidf, sim_matrix = build_recommender(df_rec)

    # save all artifacts
    print("\n--- Saving Artifacts ---")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    save_artifact(topic_tfidf,  "topic_tfidf.pkl")
    save_artifact(topic_model,  "topic_model.pkl")
    save_artifact(scaler,       "churn_scaler.pkl")
    save_artifact(churn_model,  "churn_model.pkl")
    save_artifact(rec_tfidf,    "rec_tfidf.pkl")
    save_artifact(sim_matrix,   "sim_matrix.pkl")
    df_readers.to_csv(f"{OUTPUT_PATH}/readers.csv", index=False)
    df_rec.to_csv(f"{OUTPUT_PATH}/rec_articles.csv", index=False)

    # run re-engagement pipeline for sample readers
    print("\n--- Re-engagement Pipeline ---")
    for reader_id in ["reader_0", "reader_50", "reader_200"]:

        reader = df_readers[
            df_readers["reader_id"] == reader_id
        ].iloc[0]

        churn_prob = predict_churn(reader, scaler, churn_model)
        is_at_risk = churn_prob > 0.5

        print(f"\n{'='*50}")
        print(f"Reader     : {reader_id}")
        print(f"Churn risk : {churn_prob:.1%}")
        print(f"Status     : {'AT RISK' if is_at_risk else 'ENGAGED'}")

        if is_at_risk:
            recs = recommend_for_reader(reader, df_rec, sim_matrix)
            print("Recommendations:")
            for category, titles in recs.items():
                print(f"  {category}:")
                for title in titles:
                    print(f"    • {title}")

    # final summary
    print("\n" + "="*60)
    print("  Pipeline Complete!")
    print("="*60)
    print(f"  Topic accuracy : {topic_acc:.4f}")
    print(f"  Churn ROC AUC  : {churn_auc:.4f}")
    print(f"  Rec articles   : {len(df_rec):,}")