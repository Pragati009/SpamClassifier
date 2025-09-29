import os
import subprocess
import joblib

# File paths
MODEL_FILE = "spam_classifier_model.pkl"
WORD_VEC_FILE = "word_vectorizer.pkl"
CHAR_VEC_FILE = "char_vectorizer.pkl"
APP_FILE = "spam_app.py"

# =========================
# 1. Train model if missing
# =========================
if not (os.path.exists(MODEL_FILE) and 
        os.path.exists(WORD_VEC_FILE) and 
        os.path.exists(CHAR_VEC_FILE)):
    print("Model or vectorizers not found. Training model...")
    subprocess.run(["python", "spam_classifier.py"])
else:
    print("Model and vectorizers found. Skipping training.")

# =========================
# 2. Launch Streamlit app
# =========================
print("Launching Streamlit app...")
subprocess.run(["streamlit", "run", APP_FILE])
