import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# your selected words (from model)
QUERY_WORDS = [
    "transplant", "cyclosporine", "pancreatic", "lactulose",
    "gout", "angiogram", "kidney", "marrow", "cholangitis",
    "tacrolimus", "lisinopril", "atorvastatin", "thyroid",
    "furosemide", "valve", "potassium", "lasix",
    "pantoprazole", "ulcer", "palpitations"
]


def simple_tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-Z]+", text)


def count_words_by_class(df):
    readm_counter = Counter()
    noreadm_counter = Counter()

    for _, row in df.iterrows():
        tokens = set(simple_tokenize(row["clean_text"]))  # document-level presence

        if row["30_day_readmission"] == 1:
            for w in QUERY_WORDS:
                if w in tokens:
                    readm_counter[w] += 1
        else:
            for w in QUERY_WORDS:
                if w in tokens:
                    noreadm_counter[w] += 1

    return readm_counter, noreadm_counter


def plot_counts(readm_counter, noreadm_counter):
    words = QUERY_WORDS
    readm_vals = [readm_counter[w] for w in words]
    noreadm_vals = [noreadm_counter[w] for w in words]

    x = range(len(words))
    width = 0.4

    plt.figure(figsize=(14, 6))

    plt.bar([i - width/2 for i in x], readm_vals, width=width, label="Readmission")
    plt.bar([i + width/2 for i in x], noreadm_vals, width=width, label="No Readmission")

    plt.xticks(list(x), words, rotation=45, ha="right")
    plt.ylabel("Number of Notes Containing Word")
    plt.title("Distribution of Clinically Relevant Words by Class")
    plt.legend()

    plt.tight_layout()
    plt.savefig("word_frequency_by_class.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")  # you can also use test.csv

    readm_counter, noreadm_counter = count_words_by_class(df)
    plot_counts(readm_counter, noreadm_counter)