from transformers import AutoTokenizer
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset(path, tokenizer):
    df = pd.read_csv(path)
    texts = df["clean_text"].astype(str).tolist()
    labels = df["30_day_readmission"].astype(int).tolist()
    print(f"Loaded {len(texts)} samples from {path}")
    print("Label distribution:", pd.Series(labels).value_counts(normalize=True))
    return TextDataset(texts, labels, tokenizer)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    train_dataset = load_dataset(args.train_csv, tokenizer)
    val_dataset = load_dataset(args.val_csv, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=2
    )
    
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        eval_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./results/final_model")
    tokenizer.save_pretrained("./results/final_model")
    test_dataset = load_dataset(args.test_csv, tokenizer)
    results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test accuracy:", results["eval_accuracy"])
    print("Test F1:", results["eval_f1"])

if __name__ == "__main__":
    main()