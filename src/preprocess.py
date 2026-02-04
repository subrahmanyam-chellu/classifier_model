import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download("punkt")
nltk.download("punkt_tab") 
nltk.download("stopwords")
nltk.download("wordnet")


EN_STOP = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()



def clean_text(text: str) -> str:
    """Lowercase + remove URLs/emails/HTML + strip punctuation."""
    text = str(text).lower()                            # 1. lowercasing
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # 2. remove urls
    text = re.sub(r"\S+@\S+", " ", text)                # 2. remove emails
    text = re.sub(r"<.*?>", " ", text)                  # 2. remove html
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # 3. remove punct/symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Word tokenization."""
    return word_tokenize(text)                          # 5. tokenization


def remove_stopwords(tokens):
    """Drop common stopwords."""
    return [t for t in tokens if t not in EN_STOP]      # 4. stopwords


def lemmatize_tokens(tokens):
    """Lemmatization."""
    return [lemmatizer.lemmatize(t) for t in tokens]    # 6. lemmatization


# Very simple typo dictionary – extend with your domain errors
COMMON_FIXES = {
    "passwrod": "password",
    "servr": "server",
    "accnt": "account",
}

def fix_typos(tokens):
    """Correct a few common typos."""
    return [COMMON_FIXES.get(t, t) for t in tokens]     # 8. spelling


def preprocess_text(text: str) -> str:
    """
    Apply all preprocessing steps:
    1. lowercase
    2. remove unnecessary chars (urls, emails, html)
    3. remove punctuation
    4. remove stopwords
    5. tokenization
    6. lemmatization
    7. (optional) remove duplicate tokens
    8. handle spelling mistakes
    9. remove very short tokens
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = fix_typos(tokens)

    # 7. remove duplicate tokens (optional)
    seen = set()
    uniq_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq_tokens.append(t)

    
    return uniq_tokens


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='cp1252', low_memory=False)
    return df


# def add_priority_signals(df, text_col='Text', priority_col='Priority', random_state=42):
#     """
#     Add priority-specific signals to text column.
    
#     Args:
#         df: pandas DataFrame with text and priority columns
#         text_col: name of text column (default: 'Text')
#         priority_col: name of priority column (default: 'Priority')
#         random_state: for reproducibility
    
#     Returns:
#         Modified DataFrame with enhanced text
#     """
#     # Work on copy to avoid side effects
#     df = df.copy()
    
#     # FIXED signals (non-empty for all priorities)
#     SIGNALS = {
#         'Low': ['minor', 'low impact', 'non-urgent', 'routine'],
#         'Medium': ['slow', 'timeout', 'moderate', 'team reported'],
#         'High': ['urgent', 'production', 'critical business'],
#         'Critical': ['outage', 'corruption', 'system down']
#     }
    
#     rng = np.random.RandomState(random_state)
    
#     def add_signals(text, priority):
#         """Bulletproof - handles empty cases"""
#         if pd.isna(text) or not isinstance(text, str):
#             return text
#         text_lower = text.lower()
        
#         # Safe signal selection
#         available_signals = SIGNALS.get(priority, [])
#         if not available_signals:
#             return text
        
#         # Pick 1-2 signals safely
#         signals = []
#         candidates = [s for s in available_signals if s not in text_lower]
        
#         if candidates:
#             signals.append(rng.choice(candidates))
        
#         if rng.random() < 0.5 and len(candidates) > 1:
#             remaining = [s for s in candidates if s not in signals]
#             if remaining:
#                 signals.append(rng.choice(remaining))
        
#         if signals:
#             return f"{text} - {' - '.join(signals)}"
        
#         return text
    
#     print("✅ Adding signals (bulletproof)...")
#     df[text_col] = df.apply(lambda row: add_signals(row[text_col], row[priority_col]), axis=1)
    
#     print(f"✅ {len(df)} rows enhanced")
#     return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Convert text to string
    - Drop duplicate tickets
    - Apply full text preprocessing
    - Drop rows that became empty
    """
    df = df.copy()
    #df = add_priority_signals(df)
    df["Text"] = df["Text"].astype(str)
    df = df.drop_duplicates(subset=["Text"])
    
    # Apply full text preprocessing
    df["clean_text"] = df["Text"].apply(preprocess_text)
    return df


df=load_data(r"C:\Storage\CodeBuckets\ai-model & server\.venv\data\IT_Support_Tickets_with_errors_and_irrelevant.csv")
#df=load_data(r"C:\Storage\infosys_project\.venv\data\IT_Support_Tickets_with_errors_and_irrelevant.csv")


