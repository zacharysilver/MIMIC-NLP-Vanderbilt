# dataloader.py

import pandas as pd
import re

def load_dataset(csv_path, text_col="clean_text", label_col="30_day_readmission"):
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    tokenized_texts = [simple_tokenize(t) for t in texts]
    
    return df, texts, tokenized_texts, labels

def simple_tokenize(text: str):
    text = str(text).lower()
    # keep words/numbers, remove most punctuation as separators
    tokens = re.findall(r"[a-zA-Z0-9\-\+\/\.]+", text)
    return tokens
