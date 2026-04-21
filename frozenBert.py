import torch
import pandas as pd
import joblib
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
BASIC_HEADER_PREFIX = "name: unit no: admission date: discharge date: date of birth:"


def _remove_basic_header(text):
    text = str(text)
    if text.lower().startswith(BASIC_HEADER_PREFIX):
        return text[len(BASIC_HEADER_PREFIX):].lstrip()
    return text


def _chunk_text(text, tokenizer, max_chunk_tokens):
    text = _remove_basic_header(text)
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    if not token_ids:
        token_ids = [tokenizer.unk_token_id or tokenizer.pad_token_id or 0]

    return [
        token_ids[i:i + max_chunk_tokens]
        for i in range(0, len(token_ids), max_chunk_tokens)
    ]


def _mean_pool(last_hidden_state, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = last_hidden_state * expanded_mask
    token_totals = masked_hidden.sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1.0)
    return token_totals / token_counts


def compute_embeddings_for_layers(texts, tokenizer, model, device, batch_size=128):

    model.eval()
    model.to(device)

    max_chunk_tokens = model.config.max_position_embeddings - tokenizer.num_special_tokens_to_add(pair=False)
    all_chunks = []
    chunk_to_document = []

    for text_idx, text in enumerate(tqdm(texts, desc="Preparing BERT chunks", unit="doc"), start=1):
        chunk_token_ids = _chunk_text(text, tokenizer, max_chunk_tokens)
        all_chunks.extend(chunk_token_ids)
        chunk_to_document.extend([text_idx - 1] * len(chunk_token_ids))

    layer_document_sums = {
        "last": None,
        "second_to_last": None,
        "second": None,
    }
    document_chunk_counts = torch.zeros(len(texts), dtype=torch.float32)
    print(f"Total chunks to process: {len(all_chunks)}")
    with torch.no_grad():
        num_batches = (len(all_chunks) + batch_size - 1) // batch_size

        chunk_starts = range(0, len(all_chunks), batch_size)

        for chunk_start in tqdm(chunk_starts, total=num_batches, desc="Computing BERT embeddings", unit="batch"):
            chunk_batch = all_chunks[chunk_start:chunk_start + batch_size]
            document_indices = chunk_to_document[chunk_start:chunk_start + batch_size]

            encoded_batch = tokenizer.pad(
                [
                    tokenizer.prepare_for_model(
                        chunk,
                        add_special_tokens=True,
                        truncation=False
                    )
                    for chunk in chunk_batch
                ],
                padding=True,
                return_tensors="pt"
            )

            input_ids = encoded_batch["input_ids"].to(device)
            attention_mask = encoded_batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            pooled_by_layer = {
                "last": _mean_pool(outputs.hidden_states[-1], attention_mask).cpu(),
                "second_to_last": _mean_pool(outputs.hidden_states[-2], attention_mask).cpu(),
                "second": _mean_pool(outputs.hidden_states[1], attention_mask).cpu(),
            }

            for layer_name, pooled_chunks in pooled_by_layer.items():
                if layer_document_sums[layer_name] is None:
                    embedding_dim = pooled_chunks.shape[1]
                    layer_document_sums[layer_name] = torch.zeros(len(texts), embedding_dim, dtype=pooled_chunks.dtype)

                for chunk_embedding, document_idx in zip(pooled_chunks, document_indices):
                    layer_document_sums[layer_name][document_idx] += chunk_embedding

            for document_idx in document_indices:
                document_chunk_counts[document_idx] += 1

    return {
        layer_name: document_sums / document_chunk_counts.unsqueeze(1).clamp(min=1.0)
        for layer_name, document_sums in layer_document_sums.items()
    }


def load_and_precompute(path, tokenizer, model, device, rows = 1000):
    
    df = pd.read_csv(path)

    texts = df["clean_text"].astype(str).tolist()
    labels = df["readmit"].astype(int).values

    print(f"Loaded {len(texts)} samples from {path}")

    embeddings_by_layer = compute_embeddings_for_layers(texts, tokenizer, model, device)
    return embeddings_by_layer, torch.tensor(labels)


def split_embeddings(embeddings, labels, random_state=41, test_size=0.2, val_size=0.1):
    indices = list(range(len(labels)))
    labels_list = labels.tolist()

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_list
    )

    train_labels_for_split = [labels_list[i] for i in train_indices]
    val_fraction_of_train = val_size / (1.0 - test_size)

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=val_fraction_of_train,
        random_state=random_state,
        stratify=train_labels_for_split
    )

    train_indices = torch.tensor(train_indices, dtype=torch.long)
    val_indices = torch.tensor(val_indices, dtype=torch.long)
    test_indices = torch.tensor(test_indices, dtype=torch.long)

    return (
        embeddings[train_indices],
        labels[train_indices],
        embeddings[val_indices],
        labels[val_indices],
        embeddings[test_indices],
        labels[test_indices],
    )

    
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")


model = AutoModel.from_pretrained(
    "medicalai/ClinicalBERT",
    num_labels=2
)

# Freeze BERT
for param in model.parameters():
    param.requires_grad = False
recompute=False
# Precompute
if recompute:
    embeddings_by_layer, labels = load_and_precompute("data/balanced_readmission.csv", tokenizer, model, device)
    torch.save(embeddings_by_layer["last"], "data/embeddings_last.pt")
    torch.save(embeddings_by_layer["second_to_last"], "data/embeddings_second_to_last.pt")
    torch.save(embeddings_by_layer["second"], "data/embeddings_second.pt")
    torch.save(labels, "data/labels.pt")
    embeddings = embeddings_by_layer["second_to_last"]
else:
    embeddings1, labels = torch.load("data/embeddings_second_to_last.pt"), torch.load("data/labels.pt")
    embeddings2 = torch.load("data/embeddings_second.pt")
    embeddings = torch.cat([embeddings1, embeddings2], dim=1)

    print(embeddings.shape, labels.shape)



train_embeds, train_labels, val_embeds, val_labels, test_embeds, test_labels = split_embeddings(embeddings, labels)





def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def evaluate_classifier(classifier, embeds, labels, split_name, save_prefix=None):
    X = to_numpy(embeds)
    y = to_numpy(labels)
    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)[:, 1]

    print(f"\n{split_name} metrics")
    print(f"Accuracy : {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, preds, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y, preds, zero_division=0):.4f}")
    print(f"AUROC    : {roc_auc_score(y, probs):.4f}")
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({split_name})")
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_{split_name}_roc.png")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_prefix:
        plt.savefig(f"{save_prefix}_{split_name}_cm.png")
    plt.show()



classifier = LogisticRegression(
    max_iter=1000,
    random_state=41,
)

classifier.fit(to_numpy(train_embeds), to_numpy(train_labels))

#evaluate_classifier(classifier, train_embeds, train_labels, "Train")
#evaluate_classifier(classifier, val_embeds, val_labels, "Validation")
evaluate_classifier(classifier, test_embeds, test_labels, "Test", save_prefix="frozenBert")

os.makedirs("results/frozenBert", exist_ok=True)
joblib.dump(classifier, "results/frozenBert/logistic_regression.joblib")
