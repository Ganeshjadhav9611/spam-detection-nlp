# 📩 Spam Detection with NLP

An AI-powered spam message classifier with a Streamlit web app. Enter a message and get an instant Spam / Not Spam prediction with a confidence score.

## ✨ Features

- Text cleaning, stopword removal, and TF-IDF vectorisation
- Multinomial Naive Bayes classifier
- Live prediction with confidence score
- Model performance view: accuracy and confusion matrix (from `metrics.json`)

## 🛠 Tech Stack

Python · scikit-learn · NLTK · Streamlit · pandas · NumPy · Matplotlib · Seaborn

## 📊 Model

- **Model:** TF-IDF + Multinomial Naive Bayes
- **Accuracy:** 81% on the held-out test set (see `metrics.json`)
- **Training:** `python train.py` — trains the model, saves `model.pkl` / `vectorizer.pkl`, and writes `metrics.json`

## 🚀 Run Locally

```bash
git clone https://github.com/Ganeshjadhav9611/spam-detection-nlp.git
cd spam-detection-nlp
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL in your browser, enter a message, and click **Check Message**.

Try these examples:
- Spam: `Congratulations! You won ₹5000. Click now!`
- Not spam: `Hey bro, let's meet today`

## 📁 Project Structure

```
app.py            # Streamlit UI
train.py          # Text cleaning, training, evaluation
requirements.txt  # Dependencies
data/             # Dataset (dataset.csv: Message, Label)
metrics.json      # Accuracy + confusion matrix from training
model.pkl         # Trained model
vectorizer.pkl    # Trained TF-IDF vectorizer
```

## 👤 Author

Ganesh Jadhav — Python Full-Stack Developer, BE (AI & ML), Impact College of Engineering and Applied Sciences, Bangalore.

LinkedIn: https://www.linkedin.com/in/ganesh-jadhav-104156429/
