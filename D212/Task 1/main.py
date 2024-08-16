import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

# Load the data
df = pd.read_csv('churn_clean.csv')

# Select the relevant columns
selected_columns = ['Age', 'Income', 'Outage_sec_perweek', 'MonthlyCharge']
df = df[selected_columns]

# Normalize the continuous variables
scaler = StandardScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# Display the first few rows of the preprocessed data
print(df.head())

# Determine the optimal number of clusters using the elbow method and silhouette score
inertia = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df, kmeans.labels_))

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Inertia plot
ax[0].plot(K, inertia, 'bo-')
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Inertia')
ax[0].set_title('Figure 1: Inertia as a function of number of clusters')

# Silhouette score plot
ax[1].plot(K, silhouette_scores, 'bo-')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Silhouette Score')
ax[1].set_title('Figure 2: Silhouette Score as a function of number of clusters')

plt.tight_layout()
plt.show()

df.to_csv("data_cleaned.csv")

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=369)
kmeans.fit(df)

centroids = kmeans.cluster_centers_

# Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids, columns=selected_columns)

# Display the centroids
print("\nCentroids:")
print(centroid_df)
