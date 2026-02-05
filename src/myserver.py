from datetime import datetime
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from contextlib import asynccontextmanager
from preprocess import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Global models & counter (same as before)
cat_model = None
cat_vectorizer = None
pri_model = None
pri_vectorizer = None
ticket_counter = 0

MODEL_PATH_C = PROJECT_ROOT / "models" / "category_classifier.pkl"
MODEL_PATH_P = PROJECT_ROOT / "models" / "priority_classifier.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cat_model, cat_vectorizer, pri_model, pri_vectorizer, ticket_counter
    try:
        data_c = joblib.load(MODEL_PATH_C)
        data_p = joblib.load(MODEL_PATH_P)
        cat_vectorizer = data_c["category_vectorizer"]
        cat_model = data_c["category_model"]
        pri_vectorizer = data_p["priority_vectorizer"]
        pri_model = data_p["priority_model"]
        try:
            with open(PROJECT_ROOT / "models" / "ticket_counter.txt", "r") as f:
                ticket_counter = int(f.read().strip())
        except FileNotFoundError:
            ticket_counter = 0
        print("✅ Models & counter loaded")
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        raise
    yield
    with open(PROJECT_ROOT / "models" / "ticket_counter.txt", "w") as f:
        f.write(str(ticket_counter))
    print("🔄 Saved counter")

app = FastAPI(lifespan=lifespan)

class TicketIn(BaseModel):
    issue: str

class TicketOut(BaseModel):
    ticket_id: str
    tStamp: str  # Changed to string for custom format
    predicted_category: str
    predicted_priority: str
    confidence: float | None = None

@app.get("/")
def read_root():
    return {"message": "IT Ticket Classifier API - Ready!"}

@app.post("/predict")
def predict_ticket(ticket: TicketIn):
    global ticket_counter
    if cat_model is None or pri_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    cleaned_issue = [clean_text(ticket.issue)]  

    X_cat = cat_vectorizer.transform(cleaned_issue)
    X_pri = pri_vectorizer.transform(cleaned_issue)
    
    pred_cat = cat_model.predict(X_cat)[0]
    pred_pri = pri_model.predict(X_pri)[0]
    
    ticket_counter += 1
    ticket_id = f"TKT-{ticket_counter:06d}"
    
    try:
        cat_dec = cat_model.decision_function(X_cat)[0]
        confidence = float(abs(max(cat_dec)))
    except:
        confidence = None
    
    # Custom timestamp format: 04/02/26-12:07:25 (DD/MM/YY-HH:MM:SS)
    timestamp = datetime.now().strftime("%d/%m/%y-%H:%M:%S")
    
    return {
        "ticket_id": ticket_id,
        "tStamp": timestamp,  # Now "04/02/26-12:07:25"
        "predicted_category": pred_cat,
        "predicted_priority": pred_pri,
        "confidence": confidence
    }
