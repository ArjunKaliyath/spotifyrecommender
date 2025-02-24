import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PRIMARY_COLOR = '#FF6B6B'
SECONDARY_COLOR = '#4ECDC4'


def load_and_preprocess_data(filepath):
    df = pd.read_csv("../data/Spotify_artist_info_cleaned.csv")

    df['genres_list'] = df['genres'].apply(lambda x: str(x).split(',') if pd.notna(x) and x != '' else [])

    df['release_span'] = df.apply(lambda x:
                                  x['last_release'] - x['first_release'] if x['first_release'] != -1 and x[
                                      'last_release'] != -1
                                  else 0, axis=1)

    return df


def analyze_popularity_metrics(df):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))

    # Popularity Distribution
    sns.histplot(data=df, x='popularity', ax=ax1, color=PRIMARY_COLOR)
    ax1.set_title('Distribution of Spotify Popularity')

    plt.tight_layout()


def analyze_genres(df):
    all_genres = [genre.strip() for genres in df['genres_list'] for genre in genres]
    genre_counts = pd.Series(all_genres).value_counts()

    genre_stats = []
    for genre in genre_counts.head(20).index:
        genre_artists = df[df['genres'].str.contains(genre, na=False)]
        avg_popularity = genre_artists['popularity'].mean()
        avg_listeners = genre_artists['monthly_listeners'].mean()
        genre_stats.append({
            'genre': genre,
            'count': genre_counts[genre],
            'avg_popularity': avg_popularity,
            'avg_monthly_listeners': avg_listeners
        })

    genre_stats_df = pd.DataFrame(genre_stats)

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Top genres by count
    sns.barplot(data=genre_stats_df, x='count', y='genre', ax=axes[0], color=PRIMARY_COLOR)
    axes[0].set_title('Top 20 Genres by Number of Artists')

    # Top genres by popularity
    sns.barplot(data=genre_stats_df, x='avg_popularity', y='genre', ax=axes[1], color=SECONDARY_COLOR)
    axes[1].set_title('Average Artist Popularity by Genre')

    plt.tight_layout()


def analyze_release_patterns(df):
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    # Release span vs Popularity
    sns.scatterplot(data=df[df['release_span'] > 0],
                    x='release_span', y='popularity',
                    alpha=0.5, ax=axes[0], color=PRIMARY_COLOR)
    axes[0].set_title('Career Span vs Popularity')

    # Number of releases vs Monthly listeners
    sns.scatterplot(data=df[(df['num_releases'] != -1) & (df['monthly_listeners'] > 0)],
                    x='num_releases', y='monthly_listeners',
                    alpha=0.5, ax=axes[1], color=SECONDARY_COLOR)
    axes[1].set_yscale('log')
    axes[1].set_title('Number of Releases vs Monthly Listeners')

    plt.tight_layout()


def generate_summary_statistics(df):
    stats = {
        'total_artists': len(df),
        'artists_with_genres': df['genres'].notna().sum(),
        'total_monthly_listeners': df['monthly_listeners'].sum(),
        'median_monthly_listeners': df['monthly_listeners'].median(),
        'avg_popularity': df['popularity'].mean(),
        'median_releases': df[df['num_releases'] != -1]['num_releases'].median(),
        'avg_career_span': df[df['release_span'] > 0]['release_span'].mean(),
        'most_recent_year': df[df['last_release'] != -1]['last_release'].max()
    }

    return stats


def visualize_top_artists(df, n=10):
    top_artists = df.nlargest(n, 'monthly_listeners')

    plt.figure(figsize=(15, 8))

    colors = [PRIMARY_COLOR, SECONDARY_COLOR] * (len(top_artists) // 2 + 1)

    bars = plt.bar(range(len(top_artists)), top_artists['monthly_listeners'] / 1_000_000,
                   color=colors[:len(top_artists)])

    plt.title(f'Top {n} Artists by Monthly Listeners', fontsize=14, pad=20)
    plt.xlabel('Artist', fontsize=12)
    plt.ylabel('Monthly Listeners (Millions)', fontsize=12)

    plt.xticks(range(len(top_artists)), top_artists['names'], rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}M',
                 ha='center', va='bottom')

    plt.tight_layout()

    return plt.gcf()


def main():
    df = load_and_preprocess_data('../data/Spotify_artist_info_cleaned.csv')

    analyze_popularity_metrics(df)
    analyze_genres(df)
    analyze_release_patterns(df)
    stats = generate_summary_statistics(df)

    top_artists_plot = visualize_top_artists(df)

    print("Dataset Summary:")
    print(f"Total number of artists: {stats['total_artists']:,}")
    print(f"Artists with genre information: {stats['artists_with_genres']:,}")
    print(f"Total monthly listeners across all artists: {stats['total_monthly_listeners']:,.0f}")
    print(f"Median monthly listeners per artist: {stats['median_monthly_listeners']:,.0f}")
    print(f"\nAverage artist popularity score: {stats['avg_popularity']:.2f}")
    print(f"Median number of releases: {stats['median_releases']:.1f}")
    print(f"Average career span (years): {stats['avg_career_span']:.1f}")

    plt.show()


if __name__ == "__main__":
    main()