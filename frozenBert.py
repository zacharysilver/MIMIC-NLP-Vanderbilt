import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback

def compute_embeddings(texts, tokenizer, model, device, batch_size=64):

    model.eval()
    model.to(device)

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )

    loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []

    with torch.no_grad():
        i=1
        for batch in loader:
            print(f"Processing batch {i} / {len(loader)}")
            i+=1
            input_ids, attention_mask = [x.to(device) for x in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # CLS token
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings)

    return torch.cat(all_embeddings)
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "features": self.embeddings[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)
def load_and_precompute(path, tokenizer, model, device):
    df = pd.read_csv(path)  # Limit to 1000 samples for faster testings

    texts = df["clean_text"].astype(str).tolist()
    labels = df["30_day_readmission"].astype(int).values

    print(f"Loaded {len(texts)} samples from {path}")

    embeddings = compute_embeddings(texts, tokenizer, model, device)

    return embeddings, torch.tensor(labels)


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")


model = AutoModel.from_pretrained(
    "medicalai/ClinicalBERT",
    num_labels=2
)

# Freeze BERT
for param in model.parameters():
    param.requires_grad = False
recompute=True
# Precompute
if recompute:
    train_embeds, train_labels = load_and_precompute("data/train.csv", tokenizer, model, device)
    torch.save(train_embeds, "data/train_embeds.pt")
    torch.save(train_labels, "data/train_labels.pt")
    val_embeds, val_labels = load_and_precompute("data/val.csv", tokenizer, model, device)
    torch.save(val_embeds, "data/val_embeds.pt")
    torch.save(val_labels, "data/val_labels.pt")
else:
    train_embeds = torch.load("data/train_embeds.pt")
    train_labels = torch.load("data/train_labels.pt")
    print(sum(train_labels) / len(train_labels))

    val_embeds = torch.load("data/val_embeds.pt")
    val_labels = torch.load("data/val_labels.pt")








# import torch.nn as nn

# class Classifier(nn.Module):
#     def __init__(self, input_dim=768):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 2)
#         )

#     def forward(self, features=None, labels=None):
#         logits = self.net(features)

#         loss = None
#         if labels is not None:
#             loss = nn.CrossEntropyLoss()(logits, labels)

#         return {"loss": loss, "logits": logits}


# train_dataset = EmbeddingDataset(train_embeds, train_labels)
# val_dataset = EmbeddingDataset(val_embeds, val_labels)

# classifier = Classifier()

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds),
#     }

# training_args = TrainingArguments(
#     output_dir="./results/frozenBERTresults",
#     per_device_train_batch_size=128,
#     per_device_eval_batch_size=128,
#     num_train_epochs=1000,
#     logging_strategy="epoch",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     learning_rate=1e-3,

# )

# trainer = Trainer(
#     model=classifier,
#     args=training_args,
#     train_dataset=train_dataset,
#     #callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics
# )

# trainer.train()
# trainer.save_model("./results/frozenBert/final_model")
