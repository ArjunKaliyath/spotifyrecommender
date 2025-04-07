# Spotify Music Recommendation System

## Overview

This project implements a music recommendation system using Spotify datasets:

- **Data:**
  - **train.csv:** A large dataset of 114,000 random Spotify tracks used for training and modeling.
  - **top_10000_1950-now.csv:** A dataset of top 10,000 Spotify tracks (from the 1950s to now) used for trend analysis and exploratory data analysis.
  - The processed file, **train_processed.csv**, is generated during data preprocessing.

- **Tech Stack:**
  - **Programming Language:** Python
  - **Libraries:** NumPy, Pandas, scikit-learn, TensorFlow (Keras), Matplotlib, Seaborn

- **Project Components:**
  - **Data Preprocessing:** Data cleaning, feature engineering (including creation of new numerical features and bucketization/one-hot encoding of categorical variables).
  - **Model Training & Recommendation:** Implementation and comparison of four unsupervised recommendation techniques:
    - Cosine Similarity-based Content Filtering
    - K-Nearest Neighbors (KNN) Filtering
    - Autoencoder-based Deep Learning Model
    - K-Means Clustering-based Recommendation
  - **Evaluation:** Ranking-based evaluation using metrics such as Precision@K, Hit Rate, Diversity, Novelty, and Feature Similarity.

## Setup

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

## Project Structure

Repo Name/
├── data/
│   ├── train.csv
│   ├── top_10000_1950-now.csv
│   └── train_processed.csv
│
├── scripts/
│   ├── data_preprocessing.py
│   └── model_training.py
│
└── report/
    └── [Report.pdf]



## Running Project Locally 

- Navigate to Scripts folder and run below command to generate train_processed.csv
```bash
python data_preprocessing.py 
```
- Once file is generated run below command to train models
```bash
python model_training.py
```
