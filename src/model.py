import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Paths ──────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("data", "processed", "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join("data", "processed", "tfidf_vectorizer.joblib")


def train(df: pd.DataFrame):
    """Train the model and save .joblib files."""
    X = df['processed_review_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    predictions = model.predict(X_test_tfidf)
    print("\n── Model Evaluation ──")
    print(classification_report(y_test, predictions))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to      {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return model, vectorizer


def load_model():
    """Load saved model and vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found. Run train_model.py first."
        )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer