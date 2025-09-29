import re
from scipy.sparse import hstack

# =========================
# Text Cleaning
# =========================
def clean_text(text: str) -> str:
    """
    Lowercase the text, remove non-alphabetic characters, and extra spaces.
    
    Args:
        text (str): Raw input text.
    
    Returns:
        str: Cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation, numbers, special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# Prediction Helper
# =========================
def predict_message(message: str, model, word_vectorizer, char_vectorizer) -> str:
    """
    Predict whether a message is Spam or Ham.
    
    Args:
        message (str): Raw text message.
        model: Trained classifier.
        word_vectorizer: Fitted word-level vectorizer.
        char_vectorizer: Fitted char-level vectorizer.
    
    Returns:
        str: 'Spam' or 'Ham'
    """
    cleaned = clean_text(message)
    X_word = word_vectorizer.transform([cleaned])
    X_char = char_vectorizer.transform([cleaned])
    X = hstack([X_word, X_char])
    prediction = model.predict(X)[0]
    return "Spam" if prediction == 1 else "Ham"
