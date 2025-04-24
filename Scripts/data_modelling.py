import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# The dataset has been loaded and preprocessed. It has the following features: 
#  - Numeric audio features (e.g., 'danceability', 'energy', etc.)
#  - Engineered features: 'tempo_bucket' and 'duration_cat'
#  - Other columns like 'track_name', 'artist' or 'artist_name'
# Here we define the numeric feature columns and extract the feature matrix.
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity',
                'energy_acoustic_diff', 'valence_energy_diff']
# df_train should be preprocessed and loaded:
df_train = pd.read_csv('/data/train_processed.csv')
# (plus any cleaning, scaling, and feature engineering)

features = df_train[feature_cols].values  # Numeric feature matrix from df_train


# Cosine Similarity-based Content Filtering 
def recommend_cosine(seed_index, feature_matrix, k=10):
    seed_vector = feature_matrix[seed_index]
    sims = cosine_similarity(seed_vector.reshape(1, -1), feature_matrix).flatten()
    similar_indices = sims.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != seed_index]
    return similar_indices[:k]

# Example recommendation for seed index 0 using cosine similarity
seed_idx = 5  # Adjust the index as needed
cosine_recs = recommend_cosine(seed_idx, features, k=5)
print("Cosine-similarity recommendations (indices):", cosine_recs)
if 'track_name' in df_train.columns:
    print("Recommended songs (Cosine):", df_train.iloc[cosine_recs]['track_name'].tolist())

#  KNN-based Content Filtering -
nn_model = NearestNeighbors(n_neighbors=11, metric='euclidean')
nn_model.fit(features)
def recommend_knn(seed_index, model, k=10):
    distances, indices = model.kneighbors(features[seed_index].reshape(1, -1), n_neighbors=k+1)
    rec_indices = [idx for idx in indices.flatten() if idx != seed_index]
    return rec_indices[:k]

knn_recs = recommend_knn(seed_idx, nn_model, k=5)
print("KNN recommendations (indices):", knn_recs)
if 'track_name' in df_train.columns:
    print("Recommended songs (KNN):", df_train.iloc[knn_recs]['track_name'].tolist())

# Autoencoder-based Deep Learning Model 
input_dim = features.shape[1]
encoding_dim = 16  # Dimension of the latent feature space

# Build the autoencoder architecture
inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(inputs)
encoded = layers.Dense(encoding_dim, activation='relu')(x)
x = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(x)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder (unsupervised training with a validation split)
autoencoder.fit(features, features, epochs=20, batch_size=256, validation_split=0.1, verbose=0)

# Extract the encoder model to get latent features
encoder_model = Model(inputs, encoded)
latent_features = encoder_model.predict(features)
print("Latent feature shape:", latent_features.shape)

def recommend_autoencoder(seed_index, latent_matrix, k=10):
    seed_vec = latent_matrix[seed_index]
    sims = cosine_similarity(seed_vec.reshape(1, -1), latent_matrix).flatten()
    sim_indices = sims.argsort()[::-1]
    sim_indices = sim_indices[sim_indices != seed_index]
    return sim_indices[:k]

ae_recs = recommend_autoencoder(seed_idx, latent_features, k=5)
print("Autoencoder recommendations (indices):", ae_recs)
if 'track_name' in df_train.columns:
    print("Recommended songs (Autoencoder):", df_train.iloc[ae_recs]['track_name'].tolist())

#  K-Means Clustering-based Recommendation 
# Combine numeric features with one-hot encoded engineered features.
# We assume df_train already contains engineered columns: 'tempo_bucket' and 'duration_cat'
eng_features = pd.get_dummies(df_train[['tempo_bucket', 'duration_cat']], drop_first=True)

genre_dummies = pd.get_dummies(df_train['genre_bucket'], prefix='genre')


features_cluster = np.hstack([features, eng_features.values,genre_dummies.values])


# (For demonstration, let's just print the shape of the resulting one-hot encoding.)
print("One-hot encoded genre bucket shape:", genre_dummies.shape)

# Train a K-Means clustering model (e.g., with 10 clusters)
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_cluster)
df_train['cluster'] = clusters  # Save the cluster assignments in the training DataFrame

def recommend_kmeans(seed_index, features_cluster, clusters, k=10):
    # Get the cluster of the seed track
    seed_cluster = clusters[seed_index]
    # Find indices in the same cluster (excluding the seed)
    cluster_indices = np.where(clusters == seed_cluster)[0]
    cluster_indices = cluster_indices[cluster_indices != seed_index]
    # Within the cluster, rank tracks by cosine similarity using the original features
    seed_vector = features[seed_index]
    sims = cosine_similarity(seed_vector.reshape(1, -1), features[cluster_indices]).flatten()
    sorted_within_cluster = cluster_indices[sims.argsort()[::-1]]
    return sorted_within_cluster[:k]

kmeans_recs = recommend_kmeans(seed_idx, features_cluster, clusters, k=5)
print("K-Means recommendations (indices):", kmeans_recs)
if 'track_name' in df_train.columns:
    print("Recommended songs (K-Means):", df_train.iloc[kmeans_recs]['track_name'].tolist())

#  Evaluation Metrics
def precision_at_k(recommended_indices, relevant_set, k):
    if k == 0:
        return 0.0
    hits = sum(1 for idx in recommended_indices[:k] if idx in relevant_set)
    return hits / k

def recall_at_k(recommended_indices, relevant_set, k):
    if len(relevant_set) == 0:
        return 0.0
    hits = sum(1 for idx in recommended_indices[:k] if idx in relevant_set)
    return hits / len(relevant_set)

def average_precision(recommended_indices, relevant_set, k):
    """Compute average precision at k for a single query."""
    hits = 0
    sum_precisions = 0.0
    for i in range(1, k+1):
        if recommended_indices[i-1] in relevant_set:
            hits += 1
            sum_precisions += hits / i
    # If no relevant items in the recommendations, return 0
    return sum_precisions / len(relevant_set) if relevant_set else 0.0


def hit_rate(recommended_indices, relevant_set, k):
    """Returns 1 if any of the top-k recommendations is relevant, else 0."""
    return 1.0 if any(idx in relevant_set for idx in recommended_indices[:k]) else 0.0


if 'genre_bucket' in df_train.columns:
    seed_genre_bucket = df_train.iloc[seed_idx]['genre_bucket']
    relevant_set = set(df_train[df_train['genre_bucket'] == seed_genre_bucket].index)
    relevant_set.discard(seed_idx)
else:
    relevant_set = set()


models_recs = {
    "Cosine": cosine_recs,
    "KNN": knn_recs,
    "Autoencoder": ae_recs,
    "K-Means": kmeans_recs
}

def diversity(recommended_indices, df):
        rec_genres = df.iloc[recommended_indices]['track_genre']
        return len(set(rec_genres)) / len(recommended_indices) if len(recommended_indices) >0 else 0

def novelty(recommended_indices, df):
    if 'popularity' in df.columns:
        return df.iloc[recommended_indices]['popularity'].mean()
    return None

def feature_similarity(seed_index, recommended_indices, feature_matrix):
    seed_vec = feature_matrix[seed_index]
    dists = [np.linalg.norm(seed_vec - feature_matrix[idx]) for idx in recommended_indices]
    return np.mean(dists) if dists else None

k = 5
for model_name, rec_indices in models_recs.items():
    prec = precision_at_k(rec_indices, relevant_set, k)
    rec = recall_at_k(rec_indices, relevant_set, k)
    ap = average_precision(rec_indices, relevant_set, k)
    hit = hit_rate(rec_indices, relevant_set, k)

    # The previously defined diversity, novelty, and feature similarity functions:
    div = diversity(rec_indices, df_train)
    nov = novelty(rec_indices, df_train)
    fsim = feature_similarity(seed_idx, rec_indices, features)

    # Format output, using "N/A" if any metric is None.
    prec_str = f"{prec:.2f}" if prec is not None else "N/A"
    hit_str = f"{hit:.2f}" if hit is not None else "N/A"
    div_str = f"{div:.2f}" if div is not None else "N/A"
    nov_str = f"{nov:.2f}" if nov is not None else "N/A"
    fsim_str = f"{fsim:.2f}" if fsim is not None else "N/A"

    print(f"{model_name} -> Precision@5: {prec_str}, "
          f"Hit Rate: {hit_str}, Diversity: {div_str}, "
          f"Novelty (avg popularity): {nov_str}, Feature similarity (distance): {fsim_str}")

# Notes on Splitting 
# For these unsupervised methods (cosine similarity, KNN, clustering), a train/test split is not strictly required
# because there is no target label. For the autoencoder model, a validation split is used during training.
# In systems with user feedback, a proper train/test or cross-validation approach is recommended.



def ndcg_at_k(recommended_indices, relevant_set, k):
    dcg = 0.0
    for i, idx in enumerate(recommended_indices[:k]):
        if idx in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # position is 1-based, hence i+2
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# Aggregated Metrics Collection
evaluation_metrics = {
    "Model": [],
    "Precision@5": [],
    "Recall@5": [],
    "MAP@5": [],
    "NDCG@5": [],
    "Hit Rate": [],
    "Diversity": [],
    "Novelty": [],
    "Feature Similarity": []
}

for model_name, rec_indices in models_recs.items():
    prec = precision_at_k(rec_indices, relevant_set, k)
    rec = recall_at_k(rec_indices, relevant_set, k)
    ap = average_precision(rec_indices, relevant_set, k)
    ndcg = ndcg_at_k(rec_indices, relevant_set, k)
    hit = hit_rate(rec_indices, relevant_set, k)
    div = diversity(rec_indices, df_train)
    nov = novelty(rec_indices, df_train)
    fsim = feature_similarity(seed_idx, rec_indices, features)

    evaluation_metrics["Model"].append(model_name)
    evaluation_metrics["Precision@5"].append(prec)
    evaluation_metrics["Recall@5"].append(rec)
    evaluation_metrics["MAP@5"].append(ap)
    evaluation_metrics["NDCG@5"].append(ndcg)
    evaluation_metrics["Hit Rate"].append(hit)
    evaluation_metrics["Diversity"].append(div)
    evaluation_metrics["Novelty"].append(nov)
    evaluation_metrics["Feature Similarity"].append(fsim)

# Convert to DataFrame for easier plotting
eval_df = pd.DataFrame(evaluation_metrics)

# Plotting
import matplotlib.pyplot as plt

for metric in eval_df.columns[1:]:
    plt.figure(figsize=(8, 4))
    plt.bar(eval_df["Model"], eval_df[metric])
    plt.title(f"{metric} Comparison Across Models")
    plt.ylabel(metric)
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
