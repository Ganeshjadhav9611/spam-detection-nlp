import streamlit as st
import json
import seaborn as sns
import matplotlib.pyplot as plt

from train import predict, load_model, clean_text

# Load model
model, vectorizer = load_model()

st.set_page_config(page_title="Spam Detection", page_icon="📩")

# 🔹 Title
st.title("📩 Spam Message Detector")
st.write("Check whether a message is Spam or Not using NLP")

# 🔹 Session state
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# 🔹 Input
user_input = st.text_area("✍ Enter your message:", value=st.session_state["input_text"])

# 🔹 Examples
st.subheader("📋 Example Messages")

spam_example = "Congratulations! You won ₹5000. Click now!"
ham_example = "Hey bro, let's meet today"

st.code(spam_example)
st.code(ham_example)

col1, col2 = st.columns(2)

with col1:
    if st.button("Use Spam Example"):
        st.session_state["input_text"] = spam_example

with col2:
    if st.button("Use Normal Example"):
        st.session_state["input_text"] = ham_example

# 🔹 Prediction
if st.button("🚀 Check Message"):
    if user_input.strip() == "":
        st.warning("Enter a message")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])

        result = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        spam_prob = prob[1]
        ham_prob = prob[0]

        if result == 1:
            st.error(f"🚫 Spam (Confidence: {spam_prob:.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {ham_prob:.2f})")

        st.progress(float(max(spam_prob, ham_prob)))

# 🔥 HEATMAP SECTION
st.markdown("---")
st.subheader("📊 Model Performance")

try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    acc = metrics["accuracy"]
    cm = metrics["confusion_matrix"]

    st.write(f"Accuracy: {acc:.2f}")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Spam", "Spam"],
                yticklabels=["Not Spam", "Spam"],
                ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

except:
    st.warning("⚠ Train model first to see heatmap")

# 🔹 Tips
st.markdown("### 💡 Tips")
st.info("Spam messages often contain: win, free, prize, click, offer")