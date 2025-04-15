# 🎬 Movie Recommendation System (Content-Based + Collaborative Filtering)

This project builds a hybrid movie recommendation system using the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/). It combines **Content-Based Filtering** and **Collaborative Filtering**, enabling real-time movie suggestions and precomputing recommendations for faster performance.

---

## 🔍 Features

- **Content-Based Filtering** using TF-IDF and cosine similarity on movie titles.
- **Collaborative Filtering** based on user ratings and shared preferences.
- **Interactive Search** using `ipywidgets` for real-time movie suggestions.
- **Multithreaded Training** using `concurrent.futures` and `tqdm` to precompute recommendations.
- **Exported Results** to `all_recommendations.csv` for faster lookup.

---

## 📦 Dataset

- MovieLens 25M: https://files.grouplens.org/datasets/movielens/ml-25m.zip
- Contains `.csv` files like `movies.csv`, `ratings.csv`, etc.

---

## 🧪 Installation

Install dependencies:

```bash
pip install pandas numpy scikit-learn tqdm ipywidgets
```
## Enable ipywidgets in Jupyter Notebook if needed:
```bash
jupyter nbextension enable --py widgetsnbextension
```
1. 📥 Download and Extract Dataset
Run the following code snippet to download and extract the dataset into your current directory:
```python
import zipfile, requests, os

url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
filename = "ml-25m.zip"

response = requests.get(url, stream=True)
with open(filename, 'wb') as f:
    for chunk in response.iter_content(8192):
        if chunk:
            f.write(chunk)

with zipfile.ZipFile(filename, 'r') as zip_ref:
    for member in zip_ref.infolist():
        name = os.path.basename(member.filename)
        if name.endswith(".csv"):
            source = zip_ref.open(member)
            with open(name, "wb") as target:
                target.write(source.read())

print("Dataset ready!")
```

2. 🧼 Data Cleaning

```python
import pandas as pd
import re

movies = pd.read_csv("movies.csv")

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["clean_title"] = movies["title"].apply(clean_title)

```

3.  🧠 Content-Based Filtering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    query = clean_title(title)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = similarity.argsort()[-5:][::-1]
    return movies.iloc[indices]
```

4. 👥 Collaborative Filtering

```python
ratings = pd.read_csv("ratings.csv")

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] >= 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] >= 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(7).merge(movies, left_index=True, right_on="movieId")[["movieId", "score", "title", "genres"]]
```

5. ⚡ Precomputing Recommendations

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_movie(row):
    movie_id = row["movieId"]
    movie_name = row["title"]
    recs = find_similar_movies(movie_id)
    recs.insert(0, "base_movieId", movie_id)
    recs.insert(1, "base_movieName", movie_name)
    return recs

all_recommendations = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_movie, row) for _, row in movies.iterrows()]
    for f in tqdm(as_completed(futures), total=len(futures)):
        all_recommendations.append(f.result())

final_df = pd.concat(all_recommendations, ignore_index=True)
final_df.to_csv("all_recommendations.csv", index=False)


```

6. 🧩 Live Widget-Based Search (Jupyter Notebook)

```python
import ipywidgets as widgets
from IPython.display import display

movie_name_input = widgets.Text(value='Toy Story', description='Movie Title:')
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            res = search(title)
            movie_ids = res['movieId'].head(5).tolist()
            df = pd.read_csv("all_recommendations.csv")
            recoms = df[df["base_movieId"] == movie_ids[0]][["movieId", "score", "title", "genres"]]

            for x in movie_ids[1:]:
                more = df[df["base_movieId"] == x][["movieId", "score", "title", "genres"]]
                recoms = pd.concat([recoms, more]).drop_duplicates(subset='movieId')

            display(recoms.reset_index(drop=True))

movie_name_input.observe(on_type, names='value')
display(movie_name_input, recommendation_list)

```

7. 📁 File Structure

.
├── movies.csv
├── ratings.csv
├── all_recommendations.csv
├── ml-25m.zip
└── movie_recommendations.ipynb
└── saveAll.py

## 📜 License
This project is for educational and research purposes only. The MovieLens dataset is licensed under the GroupLens Terms of Use.

## 🙌 Acknowledgements
MovieLens Project by GroupLens

pandas, numpy, scikit-learn, ipywidgets, tqdm

