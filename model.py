import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

# Generate distinct cluster data
np.random.seed(42)
X = np.vstack((
    np.random.randn(100, 2) * 0.75 + [10, 10],
    np.random.randn(100, 2) * 0.75 + [0, 0],
    np.random.randn(100, 2) * 0.75 + [-10, 10]
))

# Train KMeans
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

# Save model and data
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump((kmeans, X), f)

print("Model trained and saved as kmeans_model.pkl.")
