import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movie_name = input("Enter Movie Name : ")


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


res = search(movie_name)

if res.empty:
    print("No matching movies found.")
    exit()

movie_ids = res['movieId'].head(3).tolist()


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (
        ratings["rating"] > 4)]["userId"].unique()
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
    # Reset index to retain `movieId` explicitly
    rec_percentages = rec_percentages.head(7).merge(
        movies, left_index=True, right_on="movieId")
    rec_percentages["movieId"] = rec_percentages["movieId"].astype(
        int)  # Explicitly ensure it's an integer
    return rec_percentages[["movieId", "score", "title", "genres"]]


# Main logic
recoms = find_similar_movies(movie_ids[0])

for x in movie_ids[1:]:
    # Concatenate and handle duplicates based on 'movieId'
    recoms = pd.concat([recoms, find_similar_movies(x)]).drop_duplicates(
        subset='movieId', keep='first').reset_index(drop=True)

# Display results
print("Top 10 recommended movies for you based on movie similarity : \n")
print(res[["title", "genres"]])

print("Top 10 recommended movies for you based on rating and movie similarity : \n")
print(recoms.sort_values('score', ascending=False).head(10).reset_index(drop=True))
