import pandas as pd
import sqlite3
import os
from src.utils import clean_text, preprocess_text, rating_to_sentiment

# ── Paths ──────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join("data", "raw", "amazon.csv")
DB_PATH       = os.path.join("data", "processed", "sentiment_data.db")


def load_and_clean(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load CSV, clean and preprocess all reviews."""
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows.")

    # Drop rows with no review text at all
    df.dropna(subset=['review_title', 'review_content'], how='all', inplace=True)
    df.drop_duplicates(inplace=True)

    # Basic cleaning
    df['review_title']   = df['review_title'].str.strip().str.lower()
    df['review_content'] = df['review_content'].str.strip().str.lower()

    # Combine title + content
    df['review_text'] = df['review_title'] + " " + df['review_content']

    # Advanced cleaning
    df['review_text'] = df['review_text'].apply(clean_text)

    # NLP preprocessing
    print("Preprocessing text (this may take a moment)...")
    df['processed_review_text'] = df['review_text'].apply(preprocess_text)

    # Sentiment labels
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True)
    df['sentiment'] = df['rating'].apply(rating_to_sentiment)

    print(f"Preprocessing complete. Final shape: {df.shape}")
    print(df['sentiment'].value_counts())
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH):
    """Save reviews and predictions to SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS reviews;")
    cursor.execute("DROP TABLE IF EXISTS sentiment_predictions;")

    cursor.execute("""
        CREATE TABLE reviews (
            review_id               INTEGER PRIMARY KEY,
            product_id              TEXT,
            original_review_text    TEXT,
            cleaned_review_text     TEXT,
            processed_review_text   TEXT,
            rating                  INTEGER,
            timestamp               TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE sentiment_predictions (
            prediction_id               INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id                   INTEGER,
            predicted_sentiment_label   TEXT,
            predicted_sentiment_score   REAL,
            FOREIGN KEY (review_id) REFERENCES reviews (review_id)
        );
    """)

    # ── NEW: Insert reviews into reviews table ──────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, row in df.iterrows():
        cursor.execute("""
            INSERT INTO reviews (
                review_id,
                product_id,
                original_review_text,
                cleaned_review_text,
                processed_review_text,
                rating,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            i,
            row.get('product_id', None),
            row.get('review_content', None),
            row.get('review_text', None),
            row.get('processed_review_text', None),
            row.get('rating', None),
            timestamp
        ))

    # ── NEW: Insert predictions into sentiment_predictions table ─
    for i, row in df.iterrows():
        cursor.execute("""
            INSERT INTO sentiment_predictions (
                review_id,
                predicted_sentiment_label,
                predicted_sentiment_score
            ) VALUES (?, ?, ?)
        """, (
            i,
            row.get('predicted_sentiment', None),  # ← from model.py
            row.get('sentiment_score', None)        # ← from model.py
        ))

    conn.commit()
    conn.close()
    print(f" Database saved to {db_path}")