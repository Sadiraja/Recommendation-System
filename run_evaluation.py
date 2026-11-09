"""
Evaluation script to benchmark Popularity, Collaborative Filtering,
Content-Based, and Hybrid models on MovieLens dataset.
Generates evaluation tables, JSON/CSV exports, cold-start simulations, and visualization charts.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cold_start import ColdStartHandler
from src.pipeline import run_pipeline


def save_and_display_results(results: dict, output_dir: Path):
    """
    Formats, prints, and saves evaluation results as JSON and CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results).T
    # Reorder columns logically
    metrics_order = [
        "Precision@5", "Precision@10", "Precision@20",
        "Recall@5", "Recall@10", "Recall@20",
        "NDCG@5", "NDCG@10", "NDCG@20",
        "MAP@5", "MAP@10", "MAP@20",
        "HitRate@5", "HitRate@10", "HitRate@20",
        "MRR@5", "MRR@10", "MRR@20",
    ]
    existing_cols = [c for c in metrics_order if c in df_results.columns]
    df_results = df_results[existing_cols]

    print("\n" + "=" * 80)
    print("                     EVALUATION RESULTS BENCHMARK TABLE")
    print("=" * 80)
    print(df_results.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)

    # Save CSV and JSON
    csv_path = output_dir / "evaluation_results.csv"
    json_path = output_dir / "evaluation_results.json"

    df_results.to_csv(csv_path, index=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[+] Saved results to {csv_path} and {json_path}")
    return df_results


def plot_metrics_comparison(df_results: pd.DataFrame, output_dir: Path):
    """
    Generates bar chart comparing models across Precision@10, Recall@10, NDCG@10, and HitRate@10.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ("Precision@10", "Precision@10 (Higher is Better)", axes[0, 0], "#1f77b4"),
        ("Recall@10", "Recall@10 (Higher is Better)", axes[0, 1], "#ff7f0e"),
        ("NDCG@10", "NDCG@10 (Higher is Better)", axes[1, 0], "#2ca02c"),
        ("HitRate@10", "HitRate@10 (Higher is Better)", axes[1, 1], "#d62728"),
    ]

    for col, title, ax, color in metrics_to_plot:
        if col in df_results.columns:
            models = df_results.index
            values = df_results[col].values
            bars = ax.barh(models, values, color=color, alpha=0.85, edgecolor="black", height=0.55)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=10, fontweight="bold")

    plt.suptitle("MovieLens Recommender Benchmark: Model Comparison at K=10", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    chart_path = output_dir / "metrics_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison plot to {chart_path}")


def run_cold_start_simulation(models: dict, dataset, output_dir: Path):
    """
    Demonstrates and validates multi-tier cold-start handling for new users.
    """
    print("\n" + "=" * 80)
    print("                    COLD-START SIMULATION FOR NEW USERS")
    print("=" * 80)

    handler = ColdStartHandler(
        popularity_model=models["Popularity Baseline"],
        content_model=models["Content-Based (TF-IDF)"],
        metadata=dataset.metadata,
    )

    # 1. Zero History Anonymous User
    print("\n[Scenario 1] Anonymous User with ZERO history (Bayesian Popularity Fallback):")
    anon_recs = handler.handle_anonymous_user(n=5)
    for rec in anon_recs:
        print(f"  #{rec['rank']} | {rec['title']} ({rec['year']}) | Genres: {rec['genres']}")

    # 2. Onboarding Genre Preferences
    preferred_genres = ["Sci-Fi", "Action", "Adventure"]
    print(f"\n[Scenario 2] New User selects genres on signup: {preferred_genres}:")
    genre_recs = handler.handle_genre_preferences(preferred_genres=preferred_genres, n=5)
    for rec in genre_recs:
        print(f"  #{rec['rank']} | {rec['title']} ({rec['year']}) | Genres: {rec['genres']}")

    # 3. Interactive Elicitation: User rates 2 seed movies
    # Let's find IDs for "Inception" or "Toy Story"
    toy_story_id = dataset.movies[dataset.movies["title"].str.contains("Toy Story", case=False, na=False)]["movieId"].iloc[0]
    star_wars_id = dataset.movies[dataset.movies["title"].str.contains("Star Wars: Episode IV", case=False, na=False)]["movieId"].iloc[0]
    seed_ratings = {toy_story_id: 5.0, star_wars_id: 5.0}

    print(f"\n[Scenario 3] New User rates 2 seed movies (Toy Story=5.0, Star Wars=5.0):")
    seed_recs = handler.handle_interactive_elicitation(seed_ratings=seed_ratings, n=5)
    for rec in seed_recs:
        print(f"  #{rec['rank']} | {rec['title']} ({rec['year']}) | Genres: {rec['genres']}")

    # Save cold-start simulation notes
    sim_path = output_dir / "cold_start_simulation.json"
    sim_data = {
        "scenario_1_anonymous": anon_recs,
        "scenario_2_genre_preferences": genre_recs,
        "scenario_3_seed_ratings": seed_recs,
    }
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_data, f, indent=4)
    print(f"\n[+] Saved cold-start simulation examples to {sim_path}")


def main():
    output_dir = Path("results")
    results, split_data, dataset, models = run_pipeline(
        data_dir="data",
        test_ratio=0.2,
        min_positive_rating=3.5,
        k_list=(5, 10, 20),
    )

    df_results = save_and_display_results(results, output_dir)
    plot_metrics_comparison(df_results, output_dir)
    run_cold_start_simulation(models, dataset, output_dir)

    print("\n[SUCCESS] Pipeline and evaluation completed successfully!")


if __name__ == "__main__":
    main()
