import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math



def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_trends(df, metric_columns):
    avg_metrics = df.groupby('Album Release Year')[metric_columns].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for col in metric_columns:
        plt.plot(avg_metrics['Album Release Year'], avg_metrics[col], label=f'Average {col}')
    plt.xlabel('Year')
    plt.ylabel('Average Value')
    plt.title('Trends of Metrics Over Time')
    plt.legend()
    plt.show()

def plot_time_series(df, column, ylabel, title):
    avg_values = df.groupby('Album Release Year')[column].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_values['Album Release Year'], avg_values[column], marker='o')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def main():
  file_path = '../data/top_10000_1950-now_cleaned.csv'
  df = pd.read_csv(file_path)

  columns_to_drop = ['Track Name', 'Artist Name(s)', 'Album Name', 'Album Artist Name(s)', 
                    'Artist Genres', 'Label', 'Time Signature']
  df = df.drop(columns=columns_to_drop)

  df['Track Duration (min)'] = df['Track Duration (ms)'] / 60000
  df['Track Duration (min)'] = df['Track Duration (min)'].apply(math.ceil)
  df = df.drop('Track Duration (ms)', axis=1)

  df['Album Release Date'] = pd.to_datetime(df['Album Release Date'], errors='coerce')
  df['Album Release Year'] = df['Album Release Date'].dt.year
  df = df.drop('Album Release Date', axis=1)

  numeric_columns = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 
                    'Speechiness', 'Acousticness', 'Instrumentalness', 
                    'Liveness', 'Valence', 'Tempo', 'Track Duration (min)', 
                    'Popularity', 'Album Release Year']

  numeric_df = df[numeric_columns]
  numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

  numeric_df = numeric_df.fillna(numeric_df.mean())


  plot_correlation_heatmap(numeric_df)
  plot_trends(numeric_df, ['Danceability', 'Acousticness', 'Instrumentalness', 
                          'Liveness', 'Speechiness', 'Energy'])
  plot_time_series(numeric_df, 'Track Duration (min)', 'Average Track Duration (Minutes)', 
                  'Average Track Duration per Year')
  plot_time_series(numeric_df, 'Loudness', 'Average Loudness', 'Average Loudness Over Time')
  plot_time_series(numeric_df, 'Tempo', 'Average Tempo', 'Average Tempo Over Time')


if __name__ == "__main__":
  main()