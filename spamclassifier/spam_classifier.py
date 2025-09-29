import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import hstack
import joblib

# Import clean_text from utils
from utils import clean_text

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'body'})
df = df.dropna().drop_duplicates()
df = df[df['body'].str.strip() != '']
df['label_bin'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
df['clean_body'] = df['body'].apply(clean_text)

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000, stop_words='english')
char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)

X_word = word_vectorizer.fit_transform(df['clean_body'])
X_char = char_vectorizer.fit_transform(df['clean_body'])
X = hstack([X_word, X_char])
y = df['label_bin']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance training data only
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Train model
model = MultinomialNB()
model.fit(X_train_res, y_train_res)

# Evaluate model on test set
y_pred = model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Save model and vectorizers
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(word_vectorizer, "word_vectorizer.pkl")
joblib.dump(char_vectorizer, "char_vectorizer.pkl")

print("Model and vectorizers saved successfully!")
