"""
Data loading and preprocessing module for MovieLens dataset.
Downloads ml-latest-small from GroupLens if not already cached.
"""

import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass
class MovieLensDataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    tags: pd.DataFrame
    metadata: pd.DataFrame


def download_and_extract_movielens(data_dir: str = "data") -> Path:
    """
    Downloads MovieLens ml-latest-small dataset and extracts it to data_dir.
    Returns path to the extracted directory.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_path / "ml-latest-small"

    if extracted_dir.exists() and (extracted_dir / "ratings.csv").exists():
        return extracted_dir

    zip_path = data_path / "ml-latest-small.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens dataset from {MOVIELENS_URL}...")
        try:
            import requests
            resp = requests.get(MOVIELENS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(resp.content)
        except Exception:
            # Fallback to urllib with unverified context if SSL certs missing on Windows
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(MOVIELENS_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, "wb") as out_file:
                out_file.write(response.read())
        print(f"Downloaded dataset to {zip_path}")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)
    print(f"Dataset extracted to {extracted_dir}")

    return extracted_dir


def extract_year_from_title(title: str) -> Tuple[str, Optional[int]]:
    """
    Extracts the release year from movie title if present:
    'Toy Story (1995)' -> ('Toy Story', 1995)
    """
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    if match:
        year = int(match.group(1))
        clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title.strip())
        return clean_title, year
    return title.strip(), None


def load_movielens_data(data_dir: str = "data", min_user_ratings: int = 5) -> MovieLensDataset:
    """
    Loads, cleans, and merges MovieLens dataset.

    Args:
        data_dir: Path to directory where ml-latest-small is or should be saved.
        min_user_ratings: Filter out users with fewer than this many interactions.

    Returns:
        MovieLensDataset container with ratings, movies, tags, and enriched metadata.
    """
    extracted_dir = download_and_extract_movielens(data_dir)

    ratings = pd.read_csv(extracted_dir / "ratings.csv")
    movies = pd.read_csv(extracted_dir / "movies.csv")
    tags_file = extracted_dir / "tags.csv"
    tags = pd.read_csv(tags_file) if tags_file.exists() else pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    # Basic cleaning
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = ratings["timestamp"].astype(int)

    # Filter users with at least min_user_ratings
    user_counts = ratings["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings["userId"].isin(valid_users)].copy()

    # Extract year and clean title
    titles_years = [extract_year_from_title(t) for t in movies["title"]]
    movies["clean_title"] = [t[0] for t in titles_years]
    movies["year"] = [t[1] for t in titles_years]

    # Normalize genres
    movies["genres_list"] = movies["genres"].apply(
        lambda g: [x.strip() for x in g.split("|") if x.strip() and x.strip() != "(no genres listed)"]
    )

    # Aggregate user tags for content representation
    if not tags.empty:
        tags["tag_clean"] = tags["tag"].astype(str).str.lower().str.strip()
        movie_tags = (
            tags.groupby("movieId")["tag_clean"]
            .apply(lambda s: " ".join(set(s)))
            .reset_index()
            .rename(columns={"tag_clean": "tags_text"})
        )
    else:
        movie_tags = pd.DataFrame(columns=["movieId", "tags_text"])

    # Build enriched metadata for content-based matching
    metadata = movies.merge(movie_tags, on="movieId", how="left")
    metadata["tags_text"] = metadata["tags_text"].fillna("")
    metadata["genres_text"] = metadata["genres_list"].apply(lambda lst: " ".join(lst))
    metadata["combined_content"] = (
        metadata["clean_title"]
        + " "
        + metadata["genres_text"]
        + " "
        + metadata["tags_text"]
    ).str.strip().str.lower()

    return MovieLensDataset(
        ratings=ratings,
        movies=movies,
        tags=tags,
        metadata=metadata,
    )
