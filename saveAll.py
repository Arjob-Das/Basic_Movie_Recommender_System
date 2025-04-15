import zipfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import requests
import tarfile
import os

url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
filename = "ml-10m.zip"

# Download the dataset
response = requests.get(url, stream=True)
with open(filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download completed.")

# Extract the ZIP file
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall()

print("Extraction completed.")

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


movies["clean_title"] = movies["title"].apply(clean_title)
vectorizer = TfidfVectorizer(ngram_range=(0, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies.iloc[indices].iloc[::-1]
    return results


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (
        ratings["rating"] >= 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(
        similar_users)) & (ratings["rating"] >= 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(
        similar_user_recs.index)) & (ratings["rating"] >= 4)]
    all_user_recs = all_users["movieId"].value_counts(
    ) / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / \
        rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    rec_percentages = rec_percentages.head(7).merge(
        movies, left_index=True, right_on="movieId")
    rec_percentages["movieId"] = rec_percentages["movieId"].astype(int)
    return rec_percentages[["movieId", "score", "title", "genres"]]


def process_movie(row):
    movie_id = row["movieId"]
    movie_name = row["title"]
    recommendations = find_similar_movies(movie_id)
    recommendations.insert(0, "base_movieId", movie_id)
    recommendations.insert(1, "base_movieName", movie_name)
    return recommendations


# Use multithreading with tqdm progress bar
all_recommendations = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_movie, row)
               for _, row in movies.iterrows()]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing movies"):
        all_recommendations.append(f.result())

# Combine all recommendations into a single DataFrame
final_recommendations = pd.concat(all_recommendations, ignore_index=True)

# Save all recommendations to a single CSV file
final_recommendations.to_csv("all_recommendations.csv", index=False)
print("Saved all recommendations to 'all_recommendations.csv'")
