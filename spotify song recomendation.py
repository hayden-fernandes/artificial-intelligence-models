# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
url = "https://drive.google.com/uc?id=1Cv8ByMlZ6d_Qm8NgU6pO2n0jpeXUgb9N"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Perform data preprocessing operations
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Data Analysis and Visualizations
# Plot distribution of playlist genres
plt.figure(figsize=(10,6))
sns.countplot(y='playlist_genre', data=df, order=df['playlist_genre'].value_counts().index)
plt.title('Distribution of Playlist Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Plot distribution of playlist names
plt.figure(figsize=(10,6))
sns.countplot(y='playlist_subgenre', data=df, order=df['playlist_subgenre'].value_counts().index)
plt.title('Distribution of Playlist Names')
plt.xlabel('Count')
plt.ylabel('Playlist Name')
plt.show()

!pip install pandas seaborn

import pandas as pd
import seaborn as sns

df.info()

df['track_name'].value_counts()

df = df[df['track_name'] != "I Don't Care (with Justin Bieber) - Loud Luxury Remix"]

df.info()

df = df[df['track_name'] != 'Memories - Dillon Francis Remix']

df['track_name'].fillna(0, inplace=True)

df['track_name'] = df['track_name'].astype('category')

!pip install pandas

import pandas as pd

df.dtypes

df['track_name'] = pd.to_numeric(df['track_name'], errors='coerce')

df = df.dropna(subset=['track_name'])

df['track_name'] = df['track_name'].astype(float)

df['track_name'].value_counts()

df['track_name'] = df['track_name'].astype('category').cat.codes

df.dtypes

df = df.drop(columns=['track_artist','track_album_id',
'track_album_name',
'track_album_release_date',
'playlist_name',
'playlist_id',
'playlist_genre',
'playlist_subgenre'])

corr_matrix = df.corr()

print(corr_matrix)

# Correlation Matrix
corr_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Clustering
# Selecting features for clustering
X = df[['valence', 'liveness', 'acousticness', 'tempo']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# From the Elbow Method, we see that the optimal number of clusters is 3
# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataframe
df['cluster'] = kmeans.labels_

# Plot clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x='valence', y='tempo', hue='cluster', data=df, palette='viridis')
plt.title('Clusters of Songs based on valance and Tempo')
plt.xlabel('valance')
plt.ylabel('Tempo')
plt.show()

# Building the Model
# For demonstration, let's assume a simple content-based filtering model
# You can use more sophisticated models like collaborative filtering or hybrid models
# For content-based filtering, we can recommend songs similar to a given song

# Let's say we want to recommend songs similar to a song with the following features
song_features = {'valence': 0.6, 'acousticsness': 0.8, 'liveness': 0.7, 'tempo': 120}

# Standardize the input features
scaled_features = scaler.transform(np.array([list(song_features.values())]))

# Predict the cluster of the input song
predicted_cluster = kmeans.predict(scaled_features)

# Get songs from the same cluster as the input song
similar_songs = df[df['cluster'] == predicted_cluster[0]].sample(5)

!pip install pandas

import pandas as pd

print(similar_songs.head())

similar_songs = pd.Series.to_frame(similar_songs)

# Display recommended songs
print("Recommended Songs:")
print(similar_songs[['track_name', 'artist_name']])

similar_songs = pd.DataFrame(si
