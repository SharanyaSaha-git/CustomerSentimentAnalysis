import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Remove HTML, punctuation, numbers, special characters."""
    text = str(text).lower()
    text = re.sub(r'<.*?>',        '', text)  # HTML tags
    text = re.sub(r'[^\w\s]',      '', text)  # Punctuation
    text = re.sub(r'\d+',          '', text)  # Numbers
    text = re.sub(r'[^a-zA-Z\s]',  '', text)  # Special chars
    return text.strip()

def preprocess_text(text: str) -> str:
    """Tokenize, remove stopwords, lemmatize."""
    tokens = word_tokenize(text)
    return ' '.join(
        lemmatizer.lemmatize(w)
        for w in tokens if w not in stop_words
    )

def full_pipeline(text: str) -> str:
    """Clean + preprocess in one call."""
    return preprocess_text(clean_text(text))

def rating_to_sentiment(rating) -> str:
    """Convert numeric rating to sentiment label."""
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"