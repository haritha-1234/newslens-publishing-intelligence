# NewsLens Streamlit dashboard for editorial teams.
# It brings together all three pipeline components —
# topic classification, churn prediction and article
# recommendations — into one interface that a publisher
# could use in their daily workflow.
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from src.utils import clean_title, CATEGORY_MAP
from src.churn import predict_churn, CHURN_FEATURES
from src.recommender import recommend_for_reader

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NewsLens — Publishing Intelligence",
    page_icon="📰",
    layout="wide"
)

# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================
# We cache everything so it loads once per session.
# Without caching the models would reload on every
# interaction which would make the app very slow.

@st.cache_resource
def load_artifacts():
    """Load all saved models and data from disk."""

    artifacts = {}

    required_files = [
        "topic_tfidf.pkl",
        "topic_model.pkl",
        "churn_model.pkl",
        "churn_scaler.pkl",
        "rec_tfidf.pkl",
        "sim_matrix.pkl"
    ]

    # check all files exist before loading
    missing = [
        f for f in required_files
        if not os.path.exists(f"data/processed/{f}")
    ]

    if missing:
        return None, missing

    with open("data/processed/topic_tfidf.pkl", "rb") as f:
        artifacts["topic_tfidf"] = pickle.load(f)

    with open("data/processed/topic_model.pkl", "rb") as f:
        artifacts["topic_model"] = pickle.load(f)

    with open("data/processed/churn_model.pkl", "rb") as f:
        artifacts["churn_model"] = pickle.load(f)

    with open("data/processed/churn_scaler.pkl", "rb") as f:
        artifacts["churn_scaler"] = pickle.load(f)

    with open("data/processed/sim_matrix.pkl", "rb") as f:
        artifacts["sim_matrix"] = pickle.load(f)

    artifacts["rec_articles"] = pd.read_csv(
        "data/processed/rec_articles.csv"
    )
    artifacts["readers"] = pd.read_csv(
        "data/processed/readers.csv"
    )

    return artifacts, []


# ============================================================
# HEADER
# ============================================================
st.title("📰 NewsLens")
st.subheader("Publishing Intelligence Platform")
st.markdown(
    "Topic classification · Churn prediction · "
    "Article recommendations"
)
st.markdown("---")

# load artifacts
artifacts, missing = load_artifacts()

if artifacts is None:
    st.error(
        "Models not found. Please run `python main.py` first "
        "to train and save the models."
    )
    st.code("python main.py")
    st.stop()

# ============================================================
# THREE TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📌 Topic Classifier",
    "📉 Churn Predictor",
    "🎯 Recommendations"
])


# ============================================================
# TAB 1 — TOPIC CLASSIFIER
# ============================================================
with tab1:

    st.header("Article Topic Classifier")
    st.write(
        "Paste any news headline and the model will "
        "automatically classify it into a topic category."
    )

    # text input for headline
    headline = st.text_input(
        "Enter a news headline:",
        placeholder="e.g. Apple announces new iPhone with AI features"
    )

    # classify button
    if st.button("Classify", key="classify_btn"):

        if not headline.strip():
            st.warning("Please enter a headline first.")
        else:
            # clean and vectorize
            cleaned = clean_title(headline)
            vector  = artifacts["topic_tfidf"].transform([cleaned])
            code    = artifacts["topic_model"].predict(vector)[0]
            proba   = artifacts["topic_model"].predict_proba(vector)[0]

            category_name = CATEGORY_MAP.get(code, code)

            # show result
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Category", category_name)

            with col2:
                confidence = max(proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")

            # show all probabilities
            st.markdown("**Probability per category:**")
            classes = artifacts["topic_model"].classes_
            prob_df = pd.DataFrame({
                "Category"   : [CATEGORY_MAP.get(c, c) for c in classes],
                "Probability": [f"{p*100:.1f}%" for p in proba]
            })
            st.dataframe(prob_df, hide_index=True)

    # try some examples
    st.markdown("---")
    st.markdown("**Try these examples:**")

    examples = [
        "Federal Reserve raises interest rates again",
        "New cancer treatment shows promising results",
        "Apple announces new MacBook with AI features",
        "Taylor Swift breaks streaming record on Spotify"
    ]

    for example in examples:
        if st.button(example, key=f"ex_{example[:20]}"):
            cleaned  = clean_title(example)
            vector   = artifacts["topic_tfidf"].transform([cleaned])
            code     = artifacts["topic_model"].predict(vector)[0]
            category = CATEGORY_MAP.get(code, code)
            st.success(f"**{example}** → {category}")


# ============================================================
# TAB 2 — CHURN PREDICTOR
# ============================================================
with tab2:

    st.header("Reader Churn Predictor")
    st.write(
        "Look up a reader by ID or enter their engagement "
        "metrics manually to predict their churn risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Look up existing reader")

        reader_id = st.text_input(
            "Reader ID:",
            placeholder="e.g. reader_42"
        )

        if st.button("Look up reader", key="lookup_btn"):

            readers_df = artifacts["readers"]
            match = readers_df[
                readers_df["reader_id"] == reader_id
            ]

            if len(match) == 0:
                st.error(f"Reader '{reader_id}' not found.")
            else:
                reader = match.iloc[0]
                churn_prob = predict_churn(
                    reader,
                    artifacts["churn_scaler"],
                    artifacts["churn_model"]
                )

                st.markdown("---")
                st.markdown("**Reader profile:**")

                profile_df = pd.DataFrame({
                    "Feature": CHURN_FEATURES,
                    "Value"  : [reader[f] for f in CHURN_FEATURES]
                })
                st.dataframe(profile_df, hide_index=True)

                # churn risk display
                st.markdown("---")
                risk_pct = churn_prob * 100

                if churn_prob > 0.7:
                    st.error(f"⚠️ HIGH CHURN RISK: {risk_pct:.1f}%")
                elif churn_prob > 0.4:
                    st.warning(f"⚡ MEDIUM CHURN RISK: {risk_pct:.1f}%")
                else:
                    st.success(f"✅ LOW CHURN RISK: {risk_pct:.1f}%")

                st.progress(churn_prob)

    with col2:
        st.subheader("Enter metrics manually")

        articles    = st.slider("Articles per week", 0, 30, 5)
        session     = st.slider("Avg session minutes", 0, 60, 8)
        days_lapsed = st.slider("Days since last visit", 0, 30, 3)
        weeks_sub   = st.slider("Weeks subscribed", 1, 104, 12)
        diversity   = st.slider("Topic diversity (1-4)", 1, 4, 2)
        newsletter  = st.slider("Newsletter opens last month", 0, 10, 3)
        app_use     = st.selectbox("Uses mobile app?", ["No", "Yes"])

        if st.button("Predict churn", key="manual_btn"):

            manual_reader = pd.Series({
                "articles_per_week"    : articles,
                "avg_session_minutes"  : session,
                "days_since_last_visit": days_lapsed,
                "weeks_subscribed"     : weeks_sub,
                "topic_diversity"      : diversity,
                "newsletter_opens"     : newsletter,
                "uses_app"             : 1 if app_use == "Yes" else 0
            })

            churn_prob = predict_churn(
                manual_reader,
                artifacts["churn_scaler"],
                artifacts["churn_model"]
            )

            risk_pct = churn_prob * 100
            st.markdown("---")

            if churn_prob > 0.7:
                st.error(f"⚠️ HIGH CHURN RISK: {risk_pct:.1f}%")
            elif churn_prob > 0.4:
                st.warning(f"⚡ MEDIUM CHURN RISK: {risk_pct:.1f}%")
            else:
                st.success(f"✅ LOW CHURN RISK: {risk_pct:.1f}%")

            st.progress(churn_prob)


# ============================================================
# TAB 3 — RECOMMENDATIONS
# ============================================================
with tab3:

    st.header("Article Recommendations")
    st.write(
        "Enter a reader ID to get personalised article "
        "recommendations. Most useful for at-risk readers."
    )

    rec_reader_id = st.text_input(
        "Reader ID:",
        placeholder="e.g. reader_0",
        key="rec_reader_input"
    )

    n_recs = st.slider(
        "Articles per category", 1, 5, 3,
        key="n_recs_slider"
    )

    if st.button("Get recommendations", key="rec_btn"):

        readers_df = artifacts["readers"]
        match = readers_df[
            readers_df["reader_id"] == rec_reader_id
        ]

        if len(match) == 0:
            st.error(f"Reader '{rec_reader_id}' not found.")
        else:
            reader     = match.iloc[0]
            churn_prob = predict_churn(
                reader,
                artifacts["churn_scaler"],
                artifacts["churn_model"]
            )

            # show churn risk first
            st.markdown("---")
            risk_pct = churn_prob * 100

            if churn_prob > 0.7:
                st.error(f"⚠️ Churn risk: {risk_pct:.1f}%")
            elif churn_prob > 0.4:
                st.warning(f"⚡ Churn risk: {risk_pct:.1f}%")
            else:
                st.success(f"✅ Churn risk: {risk_pct:.1f}%")

            # get recommendations
            recs = recommend_for_reader(
                reader,
                artifacts["rec_articles"],
                artifacts["sim_matrix"],
                n=n_recs
            )

            if recs:
                st.markdown("---")
                st.markdown("**Recommended articles:**")

                # show each category in its own column
                cols = st.columns(len(recs))

                for col, (category, titles) in zip(cols, recs.items()):
                    with col:
                        st.markdown(f"**{category}**")
                        for title in titles:
                            st.markdown(f"• {title}")
            else:
                st.info("No recommendations available.")

    # show at-risk readers
    st.markdown("---")
    st.markdown("**At-risk readers to try:**")

    readers_df   = artifacts["readers"]
    sample_ids   = ["reader_0", "reader_1", "reader_2",
                    "reader_3", "reader_4"]

    at_risk_list = []
    for rid in sample_ids:
        match = readers_df[readers_df["reader_id"] == rid]
        if len(match) > 0:
            prob = predict_churn(
                match.iloc[0],
                artifacts["churn_scaler"],
                artifacts["churn_model"]
            )
            if prob > 0.5:
                at_risk_list.append(
                    {"Reader ID": rid, "Churn Risk": f"{prob*100:.1f}%"}
                )

    if at_risk_list:
        st.dataframe(
            pd.DataFrame(at_risk_list),
            hide_index=True
        )