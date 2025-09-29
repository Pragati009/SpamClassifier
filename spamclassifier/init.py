# spamclassifier/__init__.py

# Expose core functions and classes for easy import
from .utils import clean_text, predict_message
from .spam_classifier import MultinomialNB  # optional, if you want to expose the model class
