import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import math
import spacy
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")


@st.cache_data
def load_top_data(path):
    df = pd.read_csv(path)
    df['Album Release Date'] = pd.to_datetime(
        df['Album Release Date'], errors='coerce', infer_datetime_format=True
    )
    df = df.dropna(subset=['Album Release Date'])
    df['year']   = df['Album Release Date'].dt.year
    df['decade'] = (df['year']//10)*10

    # tempo & duration buckets
    tempo_bins   = [-np.inf,60,90,120,150,np.inf]
    tempo_labels = ['Very Slow','Slow','Moderate','Fast','Very Fast']
    df['tempo_bucket'] = pd.cut(df['Tempo'], bins=tempo_bins, labels=tempo_labels)

    df['duration_min'] = df['Track Duration (ms)']/60000
    dur_bins   = [-np.inf,2,4,6,np.inf]
    dur_labels = ['Short','Medium','Long','Very Long']
    df['duration_cat'] = pd.cut(df['duration_min'], bins=dur_bins, labels=dur_labels)

    return df

@st.cache_data
def load_artist_data(path):
    df = pd.read_csv(path)
    # split genres
    df['genres_list'] = df['genres'].apply(lambda x:
        str(x).split(',') if pd.notna(x) and x!='' else []
    )
    # compute career span
    df['release_span'] = df.apply(
        lambda r: (r['last_release']-r['first_release'])
                  if r['first_release']!=-1 and r['last_release']!=-1 else 0,
        axis=1
    )
    return df

@st.cache_data
def load_track_data(path):
    df = pd.read_csv(path)

    numcols = ['popularity','duration_ms','danceability','energy','loudness',
               'speechiness','acousticness','instrumentalness','liveness',
               'valence','tempo']
    for c in numcols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['explicit'] = df['explicit'].map({'TRUE':True,'FALSE':False})
    return df

@st.cache_data
def load_train_processed(path):
    return pd.read_csv(path)


def plot_artist_popularity(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df['popularity'], ax=ax, color='#FF6B6B')
    ax.set_title("Distribution of Artist Popularity")
    return fig

def plot_artist_genres(df):
    allg = [g.strip() for row in df['genres_list'] for g in row]
    counts = pd.Series(allg).value_counts().head(20)
    stats = []
    for genre in counts.index:
        sub = df[df['genres'].str.contains(genre, na=False)]
        stats.append((genre, counts[genre],
                      sub['popularity'].mean(),
                      sub['monthly_listeners'].mean()))
    gdf = pd.DataFrame(stats, columns=['genre','count','avg_pop','avg_list'])
    fig, axes = plt.subplots(2,1,figsize=(8,6))
    sns.barplot(data=gdf, x='count',y='genre',ax=axes[0], color='#4ECDC4')
    axes[0].set_title("Top Genres by Artist Count")
    sns.barplot(data=gdf, x='avg_pop',y='genre',ax=axes[1], color='#FF6B6B')
    axes[1].set_title("Avg Artist Popularity by Genre")
    plt.tight_layout()
    return fig

def plot_artist_release(df):
    fig, axes = plt.subplots(2,1,figsize=(8,6))
    sns.scatterplot(data=df[df['release_span']>0],
                    x='release_span',y='popularity',
                    alpha=0.5,ax=axes[0], color='#FF6B6B')
    axes[0].set_title("Career Span vs Popularity")
    sns.scatterplot(data=df[(df['num_releases']!=-1)&(df['monthly_listeners']>0)],
                    x='num_releases',y='monthly_listeners',
                    alpha=0.5,ax=axes[1], color='#4ECDC4')
    axes[1].set_yscale('log')
    axes[1].set_title("Releases vs Monthly Listeners")
    plt.tight_layout()
    return fig

def plot_artist_top(df,n=10):
    top = df.nlargest(n,'monthly_listeners')
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(top['names'], top['monthly_listeners']/1e6,
                  color=['#FF6B6B','#4ECDC4']*n)
    ax.set_title(f"Top {n} Artists by Monthly Listeners (M)")
    ax.set_xticklabels(top['names'], rotation=45, ha='right')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height(),f"{b.get_height():.1f}",
                ha='center',va='bottom')
    plt.tight_layout()
    return fig

def plot_track_corr(df):
    numcols = ['popularity','danceability','energy','loudness',
               'speechiness','acousticness','instrumentalness',
               'liveness','valence','tempo']
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numcols].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Track Audio Feature Correlations")
    return fig

def plot_track_features(df):
    feats = ['danceability','energy','speechiness','acousticness',
             'instrumentalness','liveness','valence']
    fig, axes = plt.subplots(3,3,figsize=(8,8))
    axes = axes.ravel()
    for i,f in enumerate(feats):
        sns.histplot(df[f], ax=axes[i])
        axes[i].set_title(f.capitalize())
    for i in range(len(feats),9):
        fig.delaxes(axes[i])
    plt.tight_layout()
    return fig

def plot_track_popularity(df):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    sns.histplot(df['popularity'], ax=ax1)
    ax1.set_title("Track Popularity Dist")
    topg = df.groupby('track_genre')['popularity'].mean().nlargest(10)
    topg.plot.bar(ax=ax2)
    ax2.set_title("Avg Popularity by Genre (Top 10)")
    plt.tight_layout()
    return fig

def plot_track_top_artists(df,n=10):
    top = df['artists'].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(6,4))
    top.plot.barh(ax=ax)
    ax.set_title(f"Top {n} Artists by # Tracks")
    plt.tight_layout()
    return fig


@st.cache_data
def recommend_by_name(song_name: str, df: pd.DataFrame, feature_matrix: np.ndarray, k: int = 5):
    mask = df['track_name'].str.lower() == song_name.lower()
    if not mask.any():
        return None
    idx = df[mask].index[0]
    sims = cosine_similarity(feature_matrix[idx:idx+1], feature_matrix).flatten()
    sims[idx] = -1
    rec_idxs = sims.argsort()[::-1][:k]

    return df.loc[rec_idxs, ['track_name', 'artists']]

@st.cache_data
def build_recommender(df):
    cols = ['danceability','energy','loudness','speechiness','acousticness',
            'instrumentalness','liveness','valence','tempo','duration_ms',
            'popularity','energy_acoustic_diff','valence_energy_diff']
    mat = df[cols].values
    return mat

def extract_intent(text):
    doc = nlp(text)
    low = text.lower()
    if any(tok.lemma_ in ("hi","hello","hey") for tok in doc):
        return "greet"
    if "recommend" in low or "suggest" in low:
        return "recommend"
    if any(tok.lemma_ in ("bye","goodbye","see") for tok in doc):
        return "bye"
    return "fallback"

st.set_page_config(layout="wide", page_title="Music Dashboard + Chatbot")
st.title("🎵 Music Dashboard & Conversational Recommender")

# load data
df_top    = load_top_data("../data/top_10000_1950-now.csv")
df_artist = load_artist_data("../data/Spotify_artist_info_cleaned.csv")
df_track  = load_track_data("../data/train_cleaned.csv")
df_train  = load_train_processed("../data/train_processed.csv")
train_mat = build_recommender(df_train)

tabs = st.tabs(["Trends","Artist EDA","Track EDA","Chatbot"])

# ─ Trends Tab ─
with tabs[0]:
    st.header("Music Trends Across Decades")
    decades = sorted(df_top['decade'].unique())
    sel = st.multiselect("Decades", decades, default=decades)
    filt = df_top[df_top['decade'].isin(sel)]
    # A: danceability
    dance = filt.groupby('decade')['Danceability'].mean().reset_index()
    c1 = alt.Chart(dance).mark_line(point=True).encode(
        x='decade:O', y='Danceability:Q'
    ).properties(height=250, title="Avg Danceability")
    st.altair_chart(c1, use_container_width=True)
    # B: tempo buckets
    tb = filt.groupby(['decade','tempo_bucket']).size().reset_index(name='count')
    c2 = alt.Chart(tb).mark_bar().encode(
        x='decade:O', y='count:Q', color='tempo_bucket:N'
    ).properties(height=250, title="Tempo Buckets")
    st.altair_chart(c2, use_container_width=True)
    # C: duration categories
    dc = filt.groupby(['decade','duration_cat']).size().reset_index(name='count')
    c3 = alt.Chart(dc).mark_bar().encode(
        x='decade:O', y='count:Q', color='duration_cat:N'
    ).properties(height=250, title="Duration Categories")
    st.altair_chart(c3, use_container_width=True)
    # D: multi‐line trends
    trend_cols = ['Danceability','Loudness','Acousticness','Energy']
    tr = filt.groupby('decade')[trend_cols].mean().reset_index().melt('decade')
    c4 = alt.Chart(tr).mark_line(point=True).encode(
        x='decade:O', y='value:Q', color='variable:N'
    ).properties(height=250, title="Feature Trends")
    st.altair_chart(c4, use_container_width=True)

with tabs[1]:
    st.header("Artist Exploratory Analysis")

    # Slider to filter on career span
    span_min, span_max = int(df_artist['release_span'].min()), int(df_artist['release_span'].max())
    sel_span = st.slider("Career span (years)", span_min, span_max, (0, span_max))

    filt_art = df_artist[
        (df_artist['release_span'] >= sel_span[0]) &
        (df_artist['release_span'] <= sel_span[1])
    ]

    #span vs popularity
    scatter = (
        alt.Chart(filt_art)
           .mark_circle()
           .encode(
               x='release_span:Q',
               y='popularity:Q',
               size='num_releases:Q',
               color='monthly_listeners:Q',
               tooltip=['names','release_span','popularity','num_releases','monthly_listeners']
           )
           .properties(height=300, width=600, title="Career Span vs Popularity")
           .interactive()
    )
    st.altair_chart(scatter)

    # top-N artists by monthly listeners
    top_n = st.number_input("Show top N artists by monthly listeners", 5, 20, 10)
    top_art = df_artist.nlargest(top_n, 'monthly_listeners')
    bar = (
        alt.Chart(top_art)
           .mark_bar()
           .encode(
               x=alt.X('monthly_listeners:Q', title="Monthly Listeners"),
               y=alt.Y('names:N', sort='-x', title="Artist"),
               tooltip=['names','monthly_listeners','release_span']
           )
           .properties(height=300, width=600, title=f"Top {top_n} Artists")
    )
    st.altair_chart(bar)

    # Debut Year Distribution
    st.subheader("Artist Debut Year Distribution")
    min_year = int(df_artist['first_release'].min())
    max_year = int(df_artist['first_release'].max())
    year_range = st.slider("Filter debut year range", min_year, max_year, (min_year, max_year))
    df_debut = df_artist[
        (df_artist['first_release'] >= year_range[0]) &
        (df_artist['first_release'] <= year_range[1])
        ]
    hist_year = (
        alt.Chart(df_debut)
        .mark_bar()
        .encode(
            x=alt.X('first_release:Q', bin=alt.Bin(step=1), title="Debut Year"),
            y=alt.Y('count()', title="Number of Artists"),
        )
        .properties(height=300, title="Artists by Debut Year")
        .interactive()
    )
    st.altair_chart(hist_year, use_container_width=True)

    #Popularity vs. Listeners
    st.subheader("Popularity vs Monthly Listeners by Genre")
    df_artist['primary_genre'] = df_artist['genres_list'].apply(lambda lst: lst[0] if lst else "Unknown")
    scatter_genre = (
        alt.Chart(df_artist)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X('monthly_listeners:Q', title="Monthly Listeners"),
            y=alt.Y('popularity:Q', title="Artist Popularity"),
            color=alt.Color('primary_genre:N', title="Genre"),
            tooltip=['names', 'primary_genre', 'monthly_listeners', 'popularity']
        )
        .properties(height=300, title="Popularity vs Listeners by Genre")
        .interactive()
    )
    st.altair_chart(scatter_genre, use_container_width=True)


with tabs[2]:
    st.header("Track Exploratory Analysis")

    feature_options = [
        'danceability','energy','loudness','speechiness',
        'acousticness','instrumentalness','liveness','valence','tempo'
    ]
    x_feat = st.selectbox("X‑axis feature", feature_options, index=0)
    y_feat = st.selectbox("Y‑axis feature", feature_options, index=1)

    # Genre filter
    genres = df_track['track_genre'].unique().tolist()
    sel_genres = st.multiselect("Filter genres", genres, default=genres[:5])

    filt_track = df_track[df_track['track_genre'].isin(sel_genres)]

    scatter2 = (
        alt.Chart(filt_track)
           .mark_circle(opacity=0.6)
           .encode(
               x=alt.X(f'{x_feat}:Q', title=x_feat.capitalize()),
               y=alt.Y(f'{y_feat}:Q', title=y_feat.capitalize()),
               color='track_genre:N',
               tooltip=['track_name','artists',x_feat,y_feat,'track_genre']
           )
           .properties(height=400, width=700, title=f"{y_feat.capitalize()} vs {x_feat.capitalize()}")
           .interactive()
    )
    st.altair_chart(scatter2)

    # Histogram of popularity
    st.subheader("Popularity Distribution")
    hist = (
        alt.Chart(filt_track)
           .mark_bar()
           .encode(
               alt.X('popularity:Q', bin=alt.Bin(maxbins=30)),
               y='count()'
           )
           .properties(height=300, width=700, title="Track Popularity Histogram")
    )
    st.altair_chart(hist)


    st.subheader("Popularity: Explicit vs. Non‑Explicit")
    box_explicit = (
        alt.Chart(df_track)
        .mark_boxplot()
        .encode(
            x=alt.X('explicit:N', title="Explicit"),
            y=alt.Y('popularity:Q', title="Popularity"),
            color='explicit:N'
        )
        .properties(height=300, title="Track Popularity by Explicit Flag")
    )
    st.altair_chart(box_explicit, use_container_width=True)

    st.subheader("Average Feature Score by Genre")
    feature_options = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    feat = st.selectbox("Choose feature", feature_options, index=0)
    top_n = st.number_input("Top N genres to show", 5, 20, 10)
    avg_feat = (
        df_track
        .groupby('track_genre')[feat]
        .mean()
        .reset_index(name='avg_score')
        .nlargest(top_n, 'avg_score')
    )
    bar_genre = (
        alt.Chart(avg_feat)
        .mark_bar()
        .encode(
            x=alt.X('avg_score:Q', title=f"Avg {feat.capitalize()}"),
            y=alt.Y('track_genre:N', sort='-x', title="Genre"),
            tooltip=['track_genre', 'avg_score']
        )
        .properties(height=300, title=f"Top {top_n} Genres by Avg {feat.capitalize()}")
    )
    st.altair_chart(bar_genre, use_container_width=True)

# ─ Chatbot ─
with tabs[3]:
    st.header("🎤 Conversational Recommender")

    #Initializing  history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Read the new user message
    user_msg = st.chat_input("Type a message…")


    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        intent = extract_intent(user_msg)
        if intent == "greet":
            bot_reply = "Hey there! Tell me a song you like, and I'll recommend some tracks."
        elif intent == "bye":
            bot_reply = "Goodbye—enjoy the music! 🎶"
        elif intent == "recommend":
            low = user_msg.lower()
            seed = user_msg.split("like", 1)[1].strip() if "like" in low else user_msg.strip()
            recs = recommend_by_name(seed, df_train, train_mat, k=5)  # returns a DataFrame

            if recs is None or recs.empty:
                bot_reply = f"Sorry, I couldn’t find '{seed}'. Try another song."
            else:
                lines = [
                    f"- **{row.track_name}** by *{row.artists}*"
                    for _, row in recs.iterrows()
                ]
                bot_reply = "Here’s what I found:\n" + "\n".join(lines)
        else:
            bot_reply = "I can recommend songs—say something like 'Recommend songs like Halo.'"

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
