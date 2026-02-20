# 🎬 Netflix Movies and TV Shows Clustering & Analysis

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** and **Unsupervised Machine Learning (Clustering)** on Netflix's content dataset to uncover hidden patterns, understand content distribution, and group similar movies and TV shows into meaningful clusters. The insights can directly support Netflix's recommendation systems, content strategy, and user engagement goals.

- **Project Type:** EDA + Unsupervised Learning (Clustering)
- **Author:** Surya Teja Chakkala
- **Domain:** Streaming / Entertainment Analytics

---

## 📂 Dataset

**File:** `NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv`

The dataset contains metadata for Netflix movies and TV shows with the following features:

| Column | Description |
|---|---|
| `type` | Movie or TV Show |
| `title` | Name of the content |
| `director` | Director(s) of the content |
| `cast` | Actors/Actresses |
| `country` | Country of production |
| `date_added` | Date added to Netflix |
| `release_year` | Year of original release |
| `rating` | Age/content rating (e.g., TV-MA, PG-13) |
| `duration` | Movie length (minutes) or number of seasons |
| `listed_in` | Genre(s) |
| `description` | Brief plot summary |

---

## 🎯 Problem Statement

Netflix hosts a massive and diverse content library, making it difficult to manually analyze content similarities and viewer-relevant patterns. The goal is to analyze Netflix movies and TV shows data and group similar content using unsupervised learning techniques — uncovering hidden patterns, understanding content distribution, and building meaningful clusters that can support recommendation systems and business decisions.

---

## 🔧 Tech Stack & Libraries

- **Python 3**
- **NumPy, Pandas** — Data manipulation
- **Matplotlib, Seaborn** — Data visualization
- **Scikit-learn** — TF-IDF vectorization, PCA, K-Means, Hierarchical Clustering, DBSCAN, Silhouette Score
- **NLTK** — Text preprocessing (stopwords, lemmatization)
- **Joblib** — Model serialization
- **Gradio** — Interactive deployment UI

---

## 🧪 Project Workflow

### 1. Know Your Data
- Loaded the dataset and inspected shape, data types, and a first look at records.
- Identified missing values in `director`, `cast`, and `country` columns.
- Checked for and removed duplicate records.

### 2. Understanding Variables
- Analyzed unique values per column.
- Described statistical properties using `df.describe(include='all')`.

### 3. Data Wrangling
- Filled missing values in `director`, `cast`, and `country` with `'Unknown'`.
- Dropped duplicate rows.
- Cleaned and normalized text fields for clustering.

### 4. Data Visualization (UBM Rule)

Followed a structured **Univariate → Bivariate → Multivariate** (UBM) approach with 15+ charts:

- **Univariate:** Content type distribution, ratings distribution, top genres, country-wise content count, release year trends.
- **Bivariate:** Content type vs. rating, release year vs. duration, genre combinations.
- **Multivariate:** Correlation heatmaps, pair plots, and cluster visualizations.

### 5. Feature Engineering & Text Preprocessing
- Combined `listed_in` (genres) and `description` into a single text feature.
- Applied preprocessing: lowercasing, punctuation removal, stopword removal, and lemmatization using NLTK.
- Transformed text using **TF-IDF Vectorization**.
- Applied **PCA** for dimensionality reduction.

### 6. Machine Learning — Clustering

Three unsupervised models were evaluated:

| Model | Description |
|---|---|
| **K-Means** | Partition-based clustering; optimal clusters found using Elbow Method & Silhouette Score |
| **Hierarchical Clustering** | Agglomerative approach using dendrograms |
| **DBSCAN** | Density-based clustering; handles noise and outliers |

**Best Model: K-Means** (selected based on highest Silhouette Score after hyperparameter tuning)

### 7. Cluster Analysis

The optimal K-Means model identified **5 content clusters**:

| Cluster | Theme |
|---|---|
| 0 | Family & Animated Content |
| 1 | Crime & Thriller |
| 2 | Romance & Drama |
| 3 | Horror & Suspense |
| 4 | Documentary & International |

### 8. Model Deployment
- Saved trained artifacts: `best_kmeans_model.pkl`, `pca_transformer.pkl`, `tfidf_vectorizer.pkl`
- Built an interactive **Gradio UI** for real-time cluster prediction from new content descriptions.
- Sanity-checked the model on unseen Netflix-style descriptions.

---

## 📊 Key Insights

- Movies dominate the Netflix catalog significantly over TV Shows.
- The majority of content is rated **TV-MA**, indicating a strong focus on adult audiences.
- Content additions accelerated sharply after 2015, reflecting Netflix's global expansion.
- The United States, India, and the United Kingdom are the top content-producing countries.
- Cluster analysis reveals distinct content themes that can directly power recommendation engines.

---

## 💼 Business Impact

- **Personalized Recommendations:** Clusters can be used to recommend similar content to users browsing within a theme.
- **Content Strategy:** Identifies underrepresented genres or regions where Netflix can acquire more content.
- **Targeted Marketing:** Enables segment-based promotions aligned with user viewing preferences.
- **User Retention:** Grouping similar content improves discovery, reducing churn.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk joblib gradio
```

### Steps
1. Clone the repository and place the dataset CSV in the working directory.
2. Open and run `Netflix_Movies_and_TV_Shows_Clustering___Analysis.ipynb` end-to-end.
3. The notebook will train the model and save the `.pkl` artifacts.
4. The final cell launches a **Gradio web interface** for live predictions.

### Live Prediction (Gradio)
Once the Gradio UI is launched, simply enter a Netflix-style description and get the predicted content cluster instantly.

---

## 📁 Project Structure

```
├── Netflix_Movies_and_TV_Shows_Clustering___Analysis.ipynb
├── NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv
├── best_kmeans_model.pkl          # Saved K-Means model
├── pca_transformer.pkl            # Saved PCA transformer
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
└── README.md
```

---

## ✅ Conclusion

This project demonstrates a complete end-to-end machine learning workflow — from raw data exploration to a deployment-ready clustering system. By applying TF-IDF, PCA, and K-Means clustering on Netflix content descriptions, the model successfully identifies five meaningful content themes. The trained model and Gradio interface are ready for real-world deployment and real-time user interaction.
