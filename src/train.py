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
        df = pd.read_csv(
            path,
            encoding="latin1",
            header=None,
            names=["label", "text"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    print("Dataset loaded successfully.")
    print(df.head())
    return df

def preprocess_columns(df):
    return df["text"], df["label"]

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