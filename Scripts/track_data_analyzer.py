import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):

    df = pd.read_csv('../data/train_cleaned.csv')
    
    numeric_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'key',
                   'loudness', 'mode', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['explicit'] = df['explicit'].map({'TRUE': True, 'FALSE': False})
    
    return df

def basic_statistics(df):
    
    numeric_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo']
    numeric_summary = df[numeric_cols].describe()
    
    categorical_summary = {
        'artists': df['artists'].nunique(),
        'genres': df['track_genre'].nunique(),
        'albums': df['album_name'].nunique(),
        'tracks': len(df)
    }
    
    return numeric_summary, categorical_summary

def analyze_correlations(df):
    """Analyze correlations between numerical features"""
    numerical_cols = ['popularity', 'danceability', 'energy', 'loudness',
                     'speechiness', 'acousticness', 'instrumentalness',
                     'liveness', 'valence', 'tempo']
    
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Audio Features')
    plt.tight_layout()
    
    return plt.gcf()

def analyze_audio_features(df):

    features = ['danceability', 'energy', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        sns.histplot(data=df, x=feature, ax=axes[idx])
        axes[idx].set_title(f'{feature.capitalize()} Distribution')
    
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()


def analyze_popularity(df):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(data=df, x='popularity', ax=ax1)
    ax1.set_title('Distribution of Track Popularity')
    

    genre_popularity = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10)
    genre_popularity.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Popularity by Genre (Top 10)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()




def analyze_top_artists(df, top_n=10):

    top_artists = df['artists'].value_counts().head(top_n)
    
    top_genres = df['track_genre'].value_counts().head(top_n)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    
    top_artists.plot(kind='barh', ax=ax1)
    ax1.set_title(f'Top {top_n} Most Common Artists')
    ax1.set_xlabel('Number of Tracks')
    
    
    plt.tight_layout()
    


def main():

    df = pd.read_csv('../data/train_cleaned.csv')
    
    # Generate all analyses
    numeric_summary, categorical_summary = basic_statistics(df)
    analyze_correlations(df)
    analyze_audio_features(df)
    analyze_popularity(df)
  
    analyze_top_artists(df)

    print("Dataset Overview:")
    print(f"Total number of tracks: {categorical_summary['tracks']}")
    print(f"Number of unique artists: {categorical_summary['artists']}")
    print(f"Number of unique genres: {categorical_summary['genres']}")
    
    print("\nBasic Music Characteristics Statistics:")
    print(numeric_summary)


    plt.show()

if __name__ == "__main__":
    main()