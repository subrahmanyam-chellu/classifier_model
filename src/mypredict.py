from pathlib import Path
import joblib
from preprocess import clean_text  # Assuming this matches preprocess_dataframe
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def predict_ticket(text: str):
    # Use project-relative paths
    cat_model_path = PROJECT_ROOT / "models" / "category_classifier.pkl"
    pri_model_path = PROJECT_ROOT / "models" / "priority_classifier.pkl"
    
    # Load separate models (matches training)
    cat_data = joblib.load(cat_model_path)
    pri_data = joblib.load(pri_model_path)
    
    cat_vectorizer = cat_data["category_vectorizer"]
    cat_model = cat_data["category_model"]
    pri_vectorizer = pri_data["priority_vectorizer"]
    pri_model = pri_data["priority_model"]

    cleaned = clean_text(text) 
    # Fixed: Match vectorizer to model
    X_vec_cat = cat_vectorizer.transform([cleaned])
    X_vec_pri = pri_vectorizer.transform([cleaned])
    
    pred_cat = cat_model.predict(X_vec_cat)[0]
    pred_pri = pri_model.predict(X_vec_pri)[0]
    
    return {"category": pred_cat, "priority": pred_pri}  # Dict for clarity

if __name__ == "__main__":
    start_ts = time.time()
    sample = "Printer is not working well stuck at middle we need resolve it"
    result = predict_ticket(sample)
    print(result)
    end_ts = time.time()
    print(f"Prediction time [s]: {end_ts - start_ts:.6f}")
