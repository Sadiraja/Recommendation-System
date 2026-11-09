"""
Streamlit Web Application for MovieLens Recommendation System.
Interactive explorer for recommendations, model comparisons, and cold-start simulations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.data_loader import load_movielens_data
from src.split import temporal_per_user_split
from src.models.popularity import PopularityRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.cold_start import ColdStartHandler

st.set_page_config(
    page_title="MovieLens Recommender System",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_and_train_system():
    dataset = load_movielens_data(data_dir="data")
    split = temporal_per_user_split(dataset.ratings, test_ratio=0.2, min_positive_rating=3.5)

    pop = PopularityRecommender().fit(split.train, dataset.metadata)
    cf = CollaborativeFilteringRecommender(n_factors=30).fit(split.train, dataset.metadata)
    cb = ContentBasedRecommender().fit(split.train, dataset.metadata)
    hybrid = HybridRecommender(cf_weight=0.6, gamma=15.0).fit(split.train, dataset.metadata)

    cold_handler = ColdStartHandler(pop, cb, dataset.metadata)

    models = {
        "Hybrid (CF + Content + Popularity)": hybrid,
        "Collaborative Filtering (SVD)": cf,
        "Content-Based (TF-IDF)": cb,
        "Popularity Baseline": pop,
    }
    return dataset, split, models, cold_handler


def main():
    st.title("🎬 MovieLens Production Recommender System")
    st.caption("A multi-model recommendation engine featuring SVD Matrix Factorization, Content-Based TF-IDF, Popularity Baseline, and Adaptive Hybrid with Cold-Start Handling.")

    with st.spinner("Loading dataset and training models..."):
        dataset, split, models, cold_handler = load_and_train_system()

    tab1, tab2, tab3 = st.tabs(["🎯 User Recommendations", "❄️ Cold-Start Simulator", "📊 Evaluation Benchmarks"])

    # ---------------- TAB 1: User Recommendations ----------------
    with tab1:
        st.subheader("Personalized Movie Recommendations")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])

        user_list = sorted(split.train["userId"].unique())
        with col_ctrl1:
            selected_user = st.selectbox("Select User ID:", user_list, index=0)

        with col_ctrl2:
            selected_model_name = st.selectbox("Select Recommender Model:", list(models.keys()))
            selected_model = models[selected_model_name]

        with col_ctrl3:
            top_n = st.slider("Number of Recommendations (Top-K):", min_value=5, max_value=30, value=10)

        col_hist, col_recs = st.columns([1, 1])

        with col_hist:
            st.markdown("#### 📜 User Interaction History (Train)")
            user_ratings = split.train[split.train["userId"] == selected_user].merge(
                dataset.movies, on="movieId", how="left"
            ).sort_values(by="rating", ascending=False)

            display_cols = ["title", "rating", "genres"]
            st.dataframe(user_ratings[display_cols].head(15), use_container_width=True)
            st.caption(f"Total ratings by User #{selected_user}: {len(user_ratings)}")

        with col_recs:
            st.markdown(f"#### 🌟 Top-{top_n} Recommendations ({selected_model_name})")
            rec_ids = selected_model.recommend(user_id=selected_user, n=top_n, exclude_seen=True)

            meta_indexed = dataset.metadata.set_index("movieId")
            rec_rows = []
            for rank, m_id in enumerate(rec_ids, 1):
                if m_id in meta_indexed.index:
                    row = meta_indexed.loc[m_id]
                    title = row.get("title", f"Movie #{m_id}")
                    genres = row.get("genres", "N/A")
                    year = row.get("year", "N/A")
                else:
                    title = f"Movie #{m_id}"
                    genres = "N/A"
                    year = "N/A"

                pred_score = selected_model.predict(selected_user, m_id)
                rec_rows.append({"Rank": rank, "Title": title, "Year": year, "Genres": genres, "Score": round(pred_score, 2)})

            rec_df = pd.DataFrame(rec_rows)
            st.dataframe(rec_df, use_container_width=True)

    # ---------------- TAB 2: Cold-Start Simulator ----------------
    with tab2:
        st.subheader("❄️ Cold-Start Handling for New Users (Zero History)")
        st.write("Demonstrates how the system handles brand-new users with no prior interaction history.")

        cold_strategy = st.radio(
            "Select Cold-Start Onboarding Strategy:",
            [
                "1. Anonymous / Zero-History User (Bayesian Popularity Baseline)",
                "2. Onboarding Genre Preferences (Content-Based Profile)",
                "3. Interactive Seed Rating Elicitation (Instant Profile Synthesis)",
            ],
        )

        top_k_cold = st.slider("Top-K items to recommend:", 5, 20, 10, key="cold_k")

        if "1. Anonymous" in cold_strategy:
            st.info("💡 **Strategy 1**: When no information is available, the Bayesian Weighted Popularity model is used. The IMDb damping formula ensures recommended items have both high average ratings and significant vote counts.")
            results = cold_handler.handle_anonymous_user(n=top_k_cold)
            st.table(pd.DataFrame(results)[["rank", "title", "year", "genres", "explanation"]])

        elif "2. Onboarding Genre" in cold_strategy:
            st.info("💡 **Strategy 2**: During registration, the user selects 1-3 favorite genres. The system builds an on-the-fly TF-IDF preference vector and computes cosine similarity against catalog items.")
            all_genres = [
                "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]
            chosen_genres = st.multiselect("Pick preferred genres:", all_genres, default=["Sci-Fi", "Action"])
            if chosen_genres:
                results = cold_handler.handle_genre_preferences(preferred_genres=chosen_genres, n=top_k_cold)
                st.table(pd.DataFrame(results)[["rank", "title", "year", "genres", "explanation"]])
            else:
                st.warning("Please select at least one genre.")

        elif "3. Interactive Seed" in cold_strategy:
            st.info("💡 **Strategy 3**: The user rates 2-3 popular seed movies. A weighted content profile vector is instantly synthesized and used to score unseen movies.")
            sample_popular = dataset.movies.head(8)
            seed_ratings = {}
            cols = st.columns(4)
            for idx, (_, row) in enumerate(sample_popular.iterrows()):
                col = cols[idx % 4]
                with col:
                    rate = col.select_slider(
                        f"{row['title'][:25]}...",
                        options=["Skip", 1.0, 2.0, 3.0, 4.0, 5.0],
                        value="Skip",
                        key=f"seed_{row['movieId']}"
                    )
                    if rate != "Skip":
                        seed_ratings[row["movieId"]] = float(rate)

            if seed_ratings:
                results = cold_handler.handle_interactive_elicitation(seed_ratings=seed_ratings, n=top_k_cold)
                st.table(pd.DataFrame(results)[["rank", "title", "year", "genres", "explanation"]])
            else:
                st.info("Rate at least one seed movie above to generate instant recommendations.")

    # ---------------- TAB 3: Evaluation Benchmarks ----------------
    with tab3:
        st.subheader("📊 Empirical Offline Evaluation Benchmark")
        st.write("Evaluating Popularity Baseline, Collaborative Filtering (SVD), Content-Based (TF-IDF), and Hybrid Recommender on temporal holdout test set.")

        results_csv = Path("results/evaluation_results.csv")
        results_png = Path("results/metrics_comparison.png")

        if results_csv.exists():
            df_eval = pd.read_csv(results_csv, index_index=0) if "index_index" in pd.read_csv.__doc__ else pd.read_csv(results_csv, index_col=0)
            st.dataframe(df_eval.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        if results_png.exists():
            st.image(str(results_png), caption="Model Performance Comparison across Precision, Recall, NDCG, and Hit Rate at K=10", use_container_width=True)


if __name__ == "__main__":
    main()
