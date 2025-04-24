import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler



df1 = pd.read_csv('/data/train.csv')

print("Shape of df1:", df1.shape)
print("First 5 rows of df1:")
print(df1.head())
print("Info for df1:")
print(df1.info())

def map_genre_to_bucket(genre):
    g = str(genre).lower()

    # Pop
    if re.search(r'pop', g):
        return 'Pop'

    # Rock (includes alternative, indie, emo, punk, etc.)
    elif re.search(r'rock|guitar|punk|grunge|emo', g):
        return 'Rock'

    # Hip-Hop/R&B
    elif re.search(r'hip[\s-]?hop|r[-\s]?n[-\s]?b|rap', g):
        return 'Hip-Hop/R&B'

    # Electronic (includes EDM, techno, trance, house, dubstep, etc.)
    elif re.search(r'electronic|edm|techno|trance|house|dubstep|deep[-\s]?house|progressive[-\s]?house|idm', g):
        return 'Electronic'

    # Jazz/Blues
    elif re.search(r'jazz|blues', g):
        return 'Jazz/Blues'

    # Classical/Opera/New Age
    elif re.search(r'classical|opera|new[-\s]?age', g):
        return 'Classical'

    # Metal (includes black metal, death metal, metalcore, etc.)
    elif re.search(r'metal|grindcore', g):
        return 'Metal'

    # Country/Folk/Bluegrass
    elif re.search(r'country|bluegrass|honky[-\s]?tonk|folk', g):
        return 'Country/Folk'

    # Latin/World (includes salsa, samba, reggaeton, latino, forro, mpb, etc.)
    elif re.search(r'latin|salsa|samba|reggaeton|latino|brazil|mpb|tango', g):
        return 'Latin/World'

    # Other special 
    elif re.search(r'children|comedy|disney|anime', g):
        return 'Family/Comedy'

    # If none of the above match, return 'Other'
    else:
        return 'Other'


relevant_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity']


df1.dropna(subset=relevant_cols, inplace=True)
df1.drop_duplicates(inplace=True)
df1 = df1[df1['duration_ms'] > 0]

df1 = df1.drop_duplicates(subset=['track_name'], keep='first')



tempo_bins = [-np.inf, 60, 90, 120, 150, np.inf]
tempo_labels = ['Very Slow', 'Slow', 'Moderate', 'Fast', 'Very Fast']
df1['tempo_bucket'] = pd.cut(df1['tempo'], bins=tempo_bins, labels=tempo_labels)

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


#  Tempo Bucket vs. Popularity
plt.figure(figsize=(10, 6))
sns.boxplot(x='tempo_bucket', y='popularity', data=df1, order=tempo_labels)
plt.title("Popularity vs. Tempo Bucket")
plt.xlabel("Tempo Bucket")
plt.ylabel("Popularity")
plt.show()


# Energy vs. Popularity (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='energy', y='popularity', data=df1, alpha=0.3)
plt.title("Energy vs. Popularity")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.show()

# Explicit Content vs. Popularity 
if 'explicit' in df1.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='explicit', y='popularity', data=df1)
    plt.title("Popularity vs. Explicit Content")
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

#  Count of Songs per Tempo Bucket
plt.figure(figsize=(10, 6))
sns.countplot(x='tempo_bucket', data=df1, order=tempo_labels)
plt.title("Count of Songs per Tempo Bucket (Random Dataset)")
plt.xlabel("Tempo Bucket")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df1, x='genre_bucket', order=genre_bucket_counts.index)
plt.title("Distribution of Refined Genre Buckets")
plt.xlabel("Genre Bucket")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()



df2 = pd.read_csv('/data/top_10000_1950-now.csv')


audio_cols = ['Danceability', 'Energy', 'Loudness', 'Speechiness',
              'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
              'Tempo', 'Track Duration (ms)', 'Popularity']
df2.dropna(subset=audio_cols, inplace=True)
df2.drop_duplicates(inplace=True)
df2 = df2[df2['Track Duration (ms)'] > 0]


scaler = StandardScaler()
df1[numeric_cols] = scaler.fit_transform(df1[numeric_cols].values)


df2['Album Release Date'] = pd.to_datetime(df2['Album Release Date'], errors='coerce')
df2 = df2.dropna(subset=['Album Release Date'])
df2['year'] = df2['Album Release Date'].dt.year

df2['decade'] = (df2['year'] // 10) * 10

df2['tempo_bucket'] = pd.cut(df2['Tempo'], bins=tempo_bins, labels=tempo_labels)
df2['duration_min'] = df2['Track Duration (ms)'] / 60000.0
df2['duration_cat'] = pd.cut(df2['duration_min'], bins=duration_bins, labels=duration_labels)



plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x='decade', hue='tempo_bucket', order=sorted(df2['decade'].unique()))
plt.title('Distribution of Tempo Buckets Across Decades')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.legend(title='Tempo Bucket')
plt.show()

dance_trend = df2.groupby('decade')['Danceability'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='decade', y='Danceability', data=dance_trend, marker='o', color='green')
plt.title('Average Danceability by Decade')
plt.xlabel('Decade')
plt.ylabel('Average Danceability')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x='decade', hue='duration_cat', order=sorted(df2['decade'].unique()))
plt.title('Distribution of Duration Categories Across Decades')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.legend(title='Duration Category')
plt.show()

trend_cols = ['Danceability', 'Loudness', 'Acousticness', 'Energy']
trend_df = df2.groupby('decade')[trend_cols].mean().reset_index()

plt.figure(figsize=(12, 8))
for col in trend_cols:
    plt.plot(trend_df['decade'], trend_df[col], marker='o', label=col)

plt.title("Music Trends Over Decades")
plt.xlabel("Decade")
plt.ylabel("Average Value")
plt.legend(title="Feature")
plt.grid(True)
plt.show()

# Summary
print("EDA for both datasets completed.")
print(" - df1  is now processed for training the recommendation models.")
print(" - df2  is used for analyzing trends in music across decades.")

df1.to_csv('/data/train_processed.csv',index=False)


df = pd.read_csv('/content/train_processed.csv')
numeric_cols = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'energy_acoustic_diff', 'valence_energy_diff'
]

print("Summary statistics of scaled numeric columns:")
print(df[numeric_cols].describe())

null_counts = df.isnull().sum()
print("\nNull values in each column:")
print(null_counts[null_counts > 0])  

duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.histplot(df['energy'], kde=True)
plt.title("Distribution of 'energy' after scaling")
plt.show()


