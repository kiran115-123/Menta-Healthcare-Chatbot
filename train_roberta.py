import os
import re
import nltk
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from nltk.corpus import stopwords
from torch.utils.data import Dataset

# Download stopwords if not present
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# === 1. Preprocessing Function ===
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        text = " ".join([word for word in text.split() if word not in stop_words])
    else:
        text = ""
    return text

# === 2. Load and Preprocess Dataset ===
df = pd.read_csv("sentiment.csv")
df = df.dropna(subset=["statement", "status"])

label_map = {
    "Depression": 0,
    "Suicidal": 1,
    "Bipolar": 2,
    "Stress": 3,
    "Anxiety": 4,
    "Personality disorder": 5,
    "Normal": 6
}

df["label"] = df["status"].map(label_map)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

sample_size = min(df["label"].value_counts().min(), 1000)

df_balanced = pd.concat([df[df["label"] == label].sample(n=sample_size, random_state=42) for label in df["label"].unique()])

print(df['status'].value_counts())

# === 3. Tokenization ===
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=256)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.labels)

X_train, X_val, y_train, y_val = train_test_split(df_balanced["statement"], df_balanced["label"], test_size=0.2, random_state=42)
train_dataset = MentalHealthDataset(X_train, y_train)
val_dataset = MentalHealthDataset(X_val, y_val)

# === 4. Compute Class Weights ===
unique_classes = df_balanced["label"].unique()
class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=df_balanced["label"])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"📊 Class Weights: {class_weights}")

# === 5. Load Model ===
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", 
    num_labels=7  # Ensure this is 7 to match your current label setup
)

# === 6. Custom Trainer with Weighted Loss ===
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# === 7. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="steps",
    save_total_limit=5,
    load_best_model_at_end=True,
    warmup_steps=500,                # Added warmup steps to stabilize training
    lr_scheduler_type="linear",      # Linear learning rate scheduler
)

# === 8. Check for Last Checkpoint ===
last_checkpoint = None
if os.path.isdir("results"):
    checkpoints = [f for f in os.listdir("results") if f.startswith("checkpoint-")]
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        last_checkpoint = "results/checkpoint-1050"

# === 9. Trainer ===
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Added Early Stopping
)
# trainer.train(resume_from_checkpoint=False)


# === 10. Train the Model (Resume if Checkpoint Found) ===
if last_checkpoint:
    print(f"🔁 Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint="results/checkpoint-1050")
else:
    trainer.train()

metrics = trainer.evaluate()
print(metrics)


# === 11. Save Final Model ===
import shutil
save_path = "mental-health-roberta"
if os.path.exists(save_path):
    shutil.rmtree(save_path)

model.save_pretrained(save_path, safe_serialization=False)
tokenizer.save_pretrained(save_path)
print(f"✅ Training complete. Model saved to '{save_path}/'")

# === 12. Inference Function ===
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_finetuned_model():
    model_path = "mental-health-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


label_map_inference = {
    0: "Depression",
    1: "Suicidal",
    2: "Bipolar",
    3: "Stress",
    4: "Anxiety",
    5: "Personality disorder",
    6: "Normal"
}

def get_sentiment(text):
    cleaned_text = preprocess_text(text)
    text_lower = text.lower()
    if "happy" in text_lower or "joy" in text_lower or "glad" in text_lower:
        return "Normal"
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        output = model(**inputs)
    scores = torch.nn.functional.softmax(output.logits, dim=-1)
    predicted_label = scores.argmax().item()
    return label_map_inference[predicted_label]

# === 13. CLI Test ===
if __name__ == "__main__":
    while True:
        user_input = input("Say something: ")
        if not user_input:
            break
        prediction = get_sentiment(user_input)
        print(f"→ Sentiment: {prediction}")
