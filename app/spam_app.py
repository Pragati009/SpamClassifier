import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Add project root to Python path so we can import spamclassifier package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utility functions
from spamclassifier.utils import clean_text, predict_message

# -----------------------
# Load model and vectorizers
# -----------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    model_path = os.path.join(base_dir, "models", "spam_classifier_model.pkl")
    word_vec_path = os.path.join(base_dir, "models", "word_vectorizer.pkl")
    char_vec_path = os.path.join(base_dir, "models", "char_vectorizer.pkl")

    model = joblib.load(model_path)
    word_vec = joblib.load(word_vec_path)
    char_vec = joblib.load(char_vec_path)
    return model, word_vec, char_vec
model, word_vectorizer, char_vectorizer = load_model()

# -----------------------
# Load dataset for visualization
# -----------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'body'})
df = df.dropna().drop_duplicates()
df['label_bin'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------
# Streamlit UI
# -----------------------
tab1, tab2 = st.tabs(["Prediction", "Visualization"])

# ----- Tab 1: Prediction -----
with tab1:
    st.title("📩 Spam Classifier App")
    st.write("Paste a message below and find out if it is **Spam or Ham**")

    with st.form("prediction_form"):
        user_input = st.text_area("Enter your message:")
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter a message first.")
        else:
            result = predict_message(user_input, model, word_vectorizer, char_vectorizer)
            st.success(f"Prediction: **{result}**")

# ----- Tab 2: Visualization -----
with tab2:
    st.title("📊 Dataset & Model Visualization")

    # Spam vs Ham distribution
    st.subheader("Message Distribution")
    counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, palette=['skyblue', 'salmon'], ax=ax)
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Confusion matrix & classification metrics
    st.subheader("Confusion Matrix & Metrics")
    X_train, X_test, y_train, y_test = train_test_split(
        df['body'], df['label_bin'], test_size=0.2, random_state=42, stratify=df['label_bin']
    )

    X_word_test = word_vectorizer.transform(X_test.apply(clean_text))
    X_char_test = char_vectorizer.transform(X_test.apply(clean_text))
    X_test_combined = hstack([X_word_test, X_char_test])

    y_pred = model.predict(X_test_combined)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    st.dataframe(metrics_df[["precision", "recall", "f1-score"]].round(2))

    # Optional: Download predictions CSV
    st.subheader("Download Predictions")
    download_df = pd.DataFrame({
        "Message": X_test,
        "Actual": y_test.map({0: "Ham", 1: "Spam"}),
        "Predicted": pd.Series(y_pred).map({0: "Ham", 1: "Spam"})
    })
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
