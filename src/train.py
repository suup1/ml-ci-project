import os
import joblib
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords (first run only)
nltk.download('stopwords')

DATA_PATH = "data/financial_news.csv"

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
        print("Loaded using UTF-8 encoding")
    except UnicodeDecodeError:
        print("UTF-8 failed. Trying latin1 encoding...")
        df = pd.read_csv(path, encoding="latin1")

    print("Columns in dataset:", df.columns)
    return df

def preprocess_columns(df):
    """
    Adjust this function if your column names differ.
    """
    # Try common names automatically
    possible_text_cols = ["text", "sentence", "news", "headline"]
    possible_label_cols = ["label", "sentiment", "category"]

    text_col = None
    label_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if text_col is None or label_col is None:
        raise ValueError("Could not automatically detect text and label columns.")

    print(f"Using text column: {text_col}")
    print(f"Using label column: {label_col}")

    return df[text_col], df[label_col]

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])

def train():
    print("Loading dataset...")
    df = load_data()

    X, y = preprocess_columns(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building pipeline...")
    model = build_pipeline()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.pkl")

    print("Model saved at models/sentiment_model.pkl")

if __name__ == "__main__":
    train()