#EDA and Feature Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler


# EDA

df1 = pd.read_csv('../Data/train.csv')

print("Shape of df1:", df1.shape)
print("First 5 rows of df1:")
print(df1.head())
print("Info for df1:")
print(df1.info())

def map_genre_to_bucket(genre):
    # Convert to lowercase for case-insensitive matching
    g = str(genre).lower()
    
    # Pop
    if re.search(r'pop', g):
        return 'Pop'
    
    # Rock 
    elif re.search(r'rock|guitar|punk|grunge|emo', g):
        return 'Rock'
    
    # Hip-Hop/R&B
    elif re.search(r'hip[\s-]?hop|r[-\s]?n[-\s]?b|rap', g):
        return 'Hip-Hop/R&B'
    
    # Electronic
    elif re.search(r'electronic|edm|techno|trance|house|dubstep|deep[-\s]?house|progressive[-\s]?house|idm', g):
        return 'Electronic'
    
    # Jazz/Blues
    elif re.search(r'jazz|blues', g):
        return 'Jazz/Blues'
    
    # Classical/Opera/New Age
    elif re.search(r'classical|opera|new[-\s]?age', g):
        return 'Classical'
    
    # Metal
    elif re.search(r'metal|grindcore', g):
        return 'Metal'
    
    # Country/Folk/Bluegrass
    elif re.search(r'country|bluegrass|honky[-\s]?tonk|folk', g):
        return 'Country/Folk'
    
    # Latin/World
    elif re.search(r'latin|salsa|samba|reggaeton|latino|brazil|mpb|tango', g):
        return 'Latin/World'
    
    # Other special categories 
    elif re.search(r'children|comedy|disney|anime', g):
        return 'Family/Comedy'
    
    # Else'Other'
    else:
        return 'Other'

# Data Cleaning

relevant_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity']



df1.dropna(subset=relevant_cols, inplace=True)

df1.drop_duplicates(inplace=True)

df1 = df1[df1['duration_ms'] > 0]

df1 = df1.drop_duplicates(subset=['track_name'], keep='first')


#Feature Engineering 

tempo_bins = [-np.inf, 60, 90, 120, 150, np.inf]
tempo_labels = ['Very Slow', 'Slow', 'Moderate', 'Fast', 'Very Fast']
df1['tempo_bucket'] = pd.cut(df1['tempo'], bins=tempo_bins, labels=tempo_labels)

# Create duration categories based on minutes
df1['duration_min'] = df1['duration_ms'] / 60000.0
duration_bins = [-np.inf, 2, 4, 6, np.inf]
duration_labels = ['Short', 'Medium', 'Long', 'Very Long']
df1['duration_cat'] = pd.cut(df1['duration_min'], bins=duration_bins, labels=duration_labels)

df1['energy_acoustic_diff'] = df1['energy'] - df1['acousticness']
df1['valence_energy_diff'] = df1['valence'] - df1['energy']


df1['genre_bucket'] = df1['track_genre'].apply(map_genre_to_bucket)


genre_bucket_counts = df1['genre_bucket'].value_counts()
print("Genre bucket counts:")
print(genre_bucket_counts)

#Exploratory Visualizations
# Tempo Bucket vs. Popularity (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='tempo_bucket', y='popularity', data=df1, order=tempo_labels)
plt.title("Popularity vs. Tempo Bucket (Random Dataset)")
plt.xlabel("Tempo Bucket")
plt.ylabel("Popularity")
plt.show()


# Energy vs. Popularity (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='energy', y='popularity', data=df1, alpha=0.3)
plt.title("Energy vs. Popularity (Random Dataset)")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.show()

# Explicit Content vs. Popularity (Violin Plot), if column exists
if 'explicit' in df1.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='explicit', y='popularity', data=df1)
    plt.title("Popularity vs. Explicit Content (Random Dataset)")
    plt.xlabel("Explicit (False=0, True=1)")
    plt.ylabel("Popularity")
    plt.show()

# Correlation Heatmap for Numeric Audio Features
numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity',
                'energy_acoustic_diff', 'valence_energy_diff']

plt.figure(figsize=(12, 10))
corr_matrix = df1[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Audio Features (Random Dataset)")
plt.show()

features = df1[numeric_cols].values

#Count of Songs per Tempo Bucket
plt.figure(figsize=(10, 6))
sns.countplot(x='tempo_bucket', data=df1, order=tempo_labels)
plt.title("Count of Songs per Tempo Bucket (Random Dataset)")
plt.xlabel("Tempo Bucket")
plt.ylabel("Count")
plt.show()

# Visualize the distribution of the broader genre buckets
plt.figure(figsize=(12, 6))
sns.countplot(data=df1, x='genre_bucket', order=genre_bucket_counts.index)
plt.title("Distribution of Refined Genre Buckets")
plt.xlabel("Genre Bucket")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


#EDA for Top 10k Dataset


df2 = pd.read_csv('../data/top_10000_1950-now.csv')

# Data Cleaning
audio_cols = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
              'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 
              'Tempo', 'Track Duration (ms)', 'Popularity']
df2.dropna(subset=audio_cols, inplace=True)
df2.drop_duplicates(inplace=True)
df2 = df2[df2['Track Duration (ms)'] > 0]

#Feature Scaling

scaler = StandardScaler()
df1[numeric_cols] = scaler.fit_transform(df1[numeric_cols].values)

#Feature Engineering
# Convert 'Album Release Date' to datetime and extract year
df2['Album Release Date'] = pd.to_datetime(df2['Album Release Date'], errors='coerce')
df2 = df2.dropna(subset=['Album Release Date'])
df2['year'] = df2['Album Release Date'].dt.year

# Create a decade column
df2['decade'] = (df2['year'] // 10) * 10

# Create engineered features: tempo buckets and duration categories
df2['tempo_bucket'] = pd.cut(df2['Tempo'], bins=tempo_bins, labels=tempo_labels)
df2['duration_min'] = df2['Track Duration (ms)'] / 60000.0
df2['duration_cat'] = pd.cut(df2['duration_min'], bins=duration_bins, labels=duration_labels)

# Exploratory Visualizations


# Distribution of Tempo Buckets Across Decades
plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x='decade', hue='tempo_bucket', order=sorted(df2['decade'].unique()))
plt.title('Distribution of Tempo Buckets Across Decades (Top 10k Dataset)')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.legend(title='Tempo Bucket')
plt.show()

# Trend in Danceability Over the Decades
dance_trend = df2.groupby('decade')['Danceability'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='decade', y='Danceability', data=dance_trend, marker='o', color='green')
plt.title('Average Danceability by Decade (Top 10k Dataset)')
plt.xlabel('Decade')
plt.ylabel('Average Danceability')
plt.grid(True)
plt.show()

# Distribution of Duration Categories Across Decades
plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x='decade', hue='duration_cat', order=sorted(df2['decade'].unique()))
plt.title('Distribution of Duration Categories Across Decades (Top 10k Dataset)')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.legend(title='Duration Category')
plt.show()

trend_cols = ['Danceability', 'Loudness', 'Acousticness', 'Energy']
trend_df = df2.groupby('decade')[trend_cols].mean().reset_index()

plt.figure(figsize=(12, 8))
for col in trend_cols:
    plt.plot(trend_df['decade'], trend_df[col], marker='o', label=col)

plt.title("Music Trends Over Decades (Top 10k Dataset)")
plt.xlabel("Decade")
plt.ylabel("Average Value")
plt.legend(title="Feature")
plt.grid(True)
plt.show()

# Summary
print("EDA for both datasets completed.")
print(" - df1 (Random Dataset) is now processed for training the recommendation models.")
print(" - df2 (Top 10k Dataset) is used for analyzing trends in music across decades.")


df1.to_csv('../data/train_processed.csv',index=False)

