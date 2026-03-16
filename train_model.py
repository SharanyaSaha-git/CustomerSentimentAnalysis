from src.data_processing import load_and_clean, save_to_sqlite
from src.model import train

# Step 1 — Load and clean
df = load_and_clean()

# Step 2 — Train model
model, vectorizer = train(df)

# ── NEW: Add predictions and scores back to df ──────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
all_reviews_tfidf = vectorizer.transform(df['processed_review_text'])
df['predicted_sentiment'] = model.predict(all_reviews_tfidf)
df['sentiment_score']     = model.predict_proba(all_reviews_tfidf).max(axis=1)
print(" Predictions and scores added to dataframe")

# Step 3 — Save to database
save_to_sqlite(df)