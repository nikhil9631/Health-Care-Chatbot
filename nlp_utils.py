"""Shared NLP preprocessing utilities for training and inference.

Both training_py.py and chatbot_py.py import from this module to ensure
identical tokenization and lemmatization.
"""

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

IGNORE_CHARS = ['?', '!', '.', ',']


def tokenize_and_lemmatize(sentence):
    """Tokenize a sentence and lemmatize each word to lowercase.

    Used identically during training (vocabulary building + bag-of-words)
    and inference (input preprocessing).
    """
    words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in words if w not in IGNORE_CHARS]


def lemmatizing_tokenizer(text):
    """Custom tokenizer for sklearn's TfidfVectorizer.

    Must be a named, importable function (not a lambda) so that the
    fitted vectorizer can be pickled and loaded during inference.
    """
    return tokenize_and_lemmatize(text)
