from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# 1) Initialize FastAPI
app = FastAPI()

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2) Load tokenizer & model from disk (no training here)
MODEL_PATH = "mental-health-roberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # set to inference mode

# 3) Define request schema
class UserInput(BaseModel):
    user_id: str
    message: str

# 4) Your label map
LABELS = [
    "Depression",
    "Suicidal",
    "Bipolar",
    "Stress",
    "Anxiety",
    "Personality disorder",
    "Normal",
]

# 5) Health check endpoint
@app.get("/health")
def health_check():
    logger.info("Health check passed.")
    return {"status": "healthy"}

# 6) Inference endpoint
@app.post("/analyze")
def analyze_sentiment(user_input: UserInput):
    # Tokenize the input message
    try:
        logger.info(f"Received message for sentiment analysis: {user_input.message}")
        inputs = tokenizer(
            user_input.message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        raise HTTPException(status_code=500, detail="Error during tokenization")

    # Predict sentiment using the model
    try:
        with torch.no_grad():
            output = model(**inputs)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        pred_idx = torch.argmax(scores, dim=-1).item()  # Get the class with the highest score
        sentiment = LABELS[pred_idx]
        logger.info(f"Predicted sentiment: {sentiment} (Class: {pred_idx}) with scores: {scores}")
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        raise HTTPException(status_code=500, detail="Error during inference")

    # Return the predicted sentiment
    return {"sentiment": sentiment}
