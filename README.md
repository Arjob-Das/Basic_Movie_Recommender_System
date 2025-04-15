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
