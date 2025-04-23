# Music Recommendation & EDA Dashboard

A unified Streamlit application that:

- **Visualizes** music trends across decades using Spotify datasets.
- Provides **interactive EDA** for both artists and tracks.
- Embeds a **conversational recommender chatbot** to suggest songs based on a seed track.

## Features

### Trends
- **Average Danceability**, **Tempo Buckets**, **Duration Categories**, and **Multi‑Feature Trends** over 1950s–Now.
- Fully interactive filters for decade selection.

### Artist EDA
- Scatter plot of **Career Span vs Popularity**, sized by release count and colored by listener base.
- Bar chart of **Top N Artists** by monthly listeners, with user‑configurable N.
- **Debut Year** distribution histogram with slider control.
- **Popularity vs Monthly Listeners** scatter, colored by primary genre.
- Summary statistics (total artists, median listeners, etc.).

### Track EDA
- Configurable **feature scatter** between any two audio attributes, genre‑filtered.
- **Popularity histogram** and **Top Artists by Track Count** bar.
- **Boxplot** comparing explicit vs non‑explicit track popularity.
- **Average Feature by Genre** bar chart for user‑selected audio metric.
- Dataset overview and descriptive stats table.

### Conversational Recommender
- Simple chat interface powered by `st.chat_input` and `st.chat_message`.
- Keyword‑based intent detection (`greet`, `recommend`, `bye`).
- Cosine‑similarity recommendations showing **Track – Artist** pairs.

## Getting Started

1. **Clone the repo** and navigate into the project:
   ```bash
   git clone <repo-url>
   cd cap5771sp25-project
   ```
2. **Install dependencies** (preferably in a venv):
   ```bash
   pip install streamlit pandas numpy altair seaborn matplotlib scikit-learn
   ```
3. **Run the app**:
   ```bash
   streamlit run scripts/app.py
   ```
4. Open `http://localhost:8501` in your browser.

## Data
- `data/top_10000_1950-now.csv`: Trend analysis across decades.
- `data/Spotify_artist_info_cleaned.csv`: Artist metadata and popularity.
- `data/train_cleaned.csv` & `data/train_processed.csv`: Track features for EDA and recommendations.

---
