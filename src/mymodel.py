from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from preprocess import load_data, preprocess_dataframe  # Assuming this exists
import chardet
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    data_pathp = r"C:\Storage\CodeBuckets\ai-model & server\.venv\data\final_dataset.csv"
    data_pathc = r"C:\Storage\CodeBuckets\ai-model & server\.venv\data\IT_Support_Tickets_with_errors_and_irrelevant.csv"
    
    def detect_encoding(data_path):
        with open(data_path, 'rb') as f:
            raw_data = f.read()
        return chardet.detect(raw_data)['encoding']

    encodingp = detect_encoding(data_pathp)
    encodingc = detect_encoding(data_pathc)
    dfp = pd.read_csv(data_pathp, encoding=encodingp)
    dfc = pd.read_csv(data_pathc, encoding=encodingc)

    dfp = preprocess_dataframe(dfp)
    dfc = preprocess_dataframe(dfc)

    # Targets
    Xc = dfc["Text"]
    y_cat = dfc["Category"]
    Xp = dfp["Text"]
    y_pri = dfp["Priority"]

    # Fixed splits: correct y and stratify=y (Series)
    X_trainc, X_testc, y_cat_train, y_cat_test = train_test_split(
        Xc, y_cat, test_size=0.2, random_state=42, stratify=y_cat  # Use y_cat!
    )
    X_trainp, X_testp, y_pri_train, y_pri_test = train_test_split(
        Xp, y_pri, test_size=0.2, random_state=42, stratify=y_pri  # Use y_pri!
    )

    # Separate vectorizers per task (standard practice for different datasets)
    vectorizer_cat = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), min_df=2)
    X_train_tfidfc = vectorizer_cat.fit_transform(X_trainc)
    X_test_tfidfc = vectorizer_cat.transform(X_testc)  # transform only!

    vectorizer_pri = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), min_df=2)
    X_train_tfidfp = vectorizer_pri.fit_transform(X_trainp)
    X_test_tfidfp = vectorizer_pri.transform(X_testp)  # transform only!

    # Models (LinearSVC good for text; LogisticRegression fine too)[web:16][web:18]
    cat_model = LogisticRegression()
    pri_model = LinearSVC()

    cat_model.fit(X_train_tfidfc, y_cat_train)
    pri_model.fit(X_train_tfidfp, y_pri_train)

    # Evaluation with matching targets
    y_cat_pred = cat_model.predict(X_test_tfidfc)
    y_pri_pred = pri_model.predict(X_test_tfidfp)

    print("\nClassification report for Category:")
    print(classification_report(y_cat_test, y_cat_pred))

    print("\nClassification report for Priority:")
    print(classification_report(y_pri_test, y_pri_pred))

    # Save separate for each task (or combined if sharing vectorizer)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump({
        "category_vectorizer": vectorizer_cat,
        "category_model": cat_model,
    }, models_dir / "category_classifier.pkl")
    
    joblib.dump({
        "priority_vectorizer": vectorizer_pri,
        "priority_model": pri_model,
    }, models_dir / "priority_classifier.pkl")

    print("Models saved separately.")

if __name__ == "__main__":
    main()
