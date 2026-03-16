from flask import Flask, request, jsonify
from src.model import load_model
from src.utils import full_pipeline

app = Flask(__name__)

# Load once at startup
model, vectorizer = load_model()


@app.route('/')
def home():
    return jsonify({"message": "Sentiment Analysis API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    """Single review → sentiment + confidence."""
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({"error": "Provide a 'review' field in JSON body"}), 400

    processed  = full_pipeline(data['review'])
    vectorized = vectorizer.transform([processed])
    sentiment  = model.predict(vectorized)[0]
    confidence = round(float(model.predict_proba(vectorized).max()), 4)

    return jsonify({
        "review":     data['review'],
        "sentiment":  sentiment,
        "confidence": confidence
    })


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """List of reviews → list of results."""
    data = request.get_json()
    if not data or 'reviews' not in data:
        return jsonify({"error": "Provide a 'reviews' list in JSON body"}), 400

    results = []
    for review in data['reviews']:
        processed  = full_pipeline(review)
        vectorized = vectorizer.transform([processed])
        sentiment  = model.predict(vectorized)[0]
        confidence = round(float(model.predict_proba(vectorized).max()), 4)
        results.append({
            "review":     review,
            "sentiment":  sentiment,
            "confidence": confidence
        })

    return jsonify({"results": results})