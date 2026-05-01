import pandas as pd
import re
import nltk
import pickle
import os
import json

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 🔹 Download stopwords
nltk.download('stopwords')

# 🔹 Paths (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# 🔹 Load stopwords once
stop_words = set(stopwords.words('english'))


# 🔹 Clean text
def clean_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


# 🔹 Train model
def train_model():
    print("🚀 Training model...")

    # Check dataset
    if not os.path.exists(DATA_PATH):
        raise Exception(f"❌ Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Check required columns
    if 'Message' not in df.columns or 'Label' not in df.columns:
        raise Exception(f"❌ Columns found: {df.columns}")

    # Encode labels
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

    # Drop missing
    df.dropna(inplace=True)

    # Clean text
    df['Message'] = df['Message'].apply(clean_text)

    X = df['Message']
    y = df['Label']

    # Vectorize
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # Train
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.4f}")
    print("📊 Confusion Matrix:\n", cm)

    # Save model
    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

    # 🔥 Save metrics (for heatmap)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist()
        }, f)

    print("✅ Model + Metrics saved successfully!")


# 🔹 Load model (AUTO TRAIN IF NOT FOUND)
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("⚠ Model not found. Training now...")
        train_model()

    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

    return model, vectorizer


# 🔹 Predict
def predict(text):
    model, vectorizer = load_model()
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]

    return "Spam" if result == 1 else "Not Spam"


# 🔹 Run training manually
if __name__ == "__main__":
    train_model()