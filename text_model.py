import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch.nn.functional as F
import os

MODEL_DIR = r"C://Users//syeda//Downloads//user//OneDrive//Desktop//python//AI-VS-HUMAN-CONTENT-DETECTION//backend//main models//ai-vs-human-content-detection-pytorch-default-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
config = AutoConfig.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
model.to(DEVICE)
model.eval()

def predict_text(text: str):
    """Predict whether text is AI-generated or human-written."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        prob_ai = probs[0][1].item()  # probability of AI class

    label = "AI-generated" if pred == 1 else "Human-written"
    confidence = round(prob_ai * 100, 2) if pred == 1 else round((1 - prob_ai) * 100, 2)
    return {
        "label": label,
        "confidence": confidence,
        "prob_ai": prob_ai,
        "prob_human": 1.0 - prob_ai
    }
