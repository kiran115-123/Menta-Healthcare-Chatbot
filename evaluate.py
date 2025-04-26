import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

print("[INFO] Loading model and tokenizer...")
model_path = "mental-health-roberta"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[INFO] Using device: {device}")

label_map = {
    "Depression": 0,
    "Suicidal": 1,
    "Bipolar": 2,
    "Stress": 3,
    "Anxiety": 4,
    "Personality disorder": 5,
    "Normal": 6
}

class MentalHealthDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        assert isinstance(dataframe, pd.DataFrame), "Dataframe is not a valid pandas.DataFrame"
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        statement = self.dataframe.loc[index, 'statement']
        if not isinstance(statement, str):
            print(f"[WARN] Invalid statement at index {index}: {statement}")
            statement = ""

        label = self.dataframe.loc[index, 'status']
        label = label_map[label]

        inputs = self.tokenizer(statement, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("[INFO] Reading and balancing dataset...")
df = pd.read_csv("sentiment.csv")

df = df.dropna(subset=["statement", "status"])
df = df[df['statement'].apply(lambda x: isinstance(x, str))]
df["label"] = df["status"].map(label_map).astype(int)

# Create balanced dataset
sample_size = min(df["label"].value_counts().min(), 1000)
df_balanced = pd.concat([
    df[df["label"] == label].sample(n=sample_size, random_state=42)
    for label in df["label"].unique()
])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"[INFO] Balanced dataset samples: {len(df_balanced)}")

print("[INFO] Splitting into train and test from balanced dataset...")
train_df, test_df = train_test_split(df_balanced, test_size=0.2, stratify=df_balanced["label"], random_state=42)


print("[INFO] Preparing datasets and dataloaders...")
train_dataset = MentalHealthDataset(train_df, tokenizer, max_len=256)
test_dataset = MentalHealthDataset(test_df, tokenizer, max_len=256)
test_loader = DataLoader(test_dataset, batch_size=16)

def evaluate_model(model, tokenizer, test_loader, label_map):
    model.eval()
    all_preds = []
    all_labels = []

    print("[INFO] Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"[INFO] Processed {i + 1}/{len(test_loader)} batches")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm, all_preds, all_labels

print("[INFO] Running final evaluation...")
accuracy, precision, recall, f1, cm, pred_labels, true_labels = evaluate_model(
    model, tokenizer, test_loader, label_map
)

print("\n[RESULTS]")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
