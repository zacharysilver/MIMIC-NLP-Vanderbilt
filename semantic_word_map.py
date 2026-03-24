import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from collections import Counter
import re
import numpy as np

# Words you want to visualize
WORDS = [
    "transplant", "cyclosporine", "pancreatic", "lactulose",
    "gout", "angiogram", "kidney", "marrow", "cholangitis",
    "tacrolimus", "lisinopril", "atorvastatin", "thyroid",
    "furosemide", "valve", "potassium", "lasix",
    "pantoprazole", "ulcer", "palpitations"
]

def simple_tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-Z]+", text)

def get_word_frequencies(csv_path, words, text_col="clean_text"):
    df = pd.read_csv(csv_path)
    counter = Counter()

    for text in df[text_col].astype(str):
        tokens = simple_tokenize(text)
        counter.update(tokens)

    freqs = {w: counter[w] for w in words}
    return freqs

def normalize_sizes(freqs, min_size=10, max_size=28):
    values = np.array(list(freqs.values()), dtype=float)
    if values.max() == values.min():
        return {k: (min_size + max_size) / 2 for k in freqs}
    norm = (values - values.min()) / (values.max() - values.min())
    sizes = min_size + norm * (max_size - min_size)
    return {k: s for k, s in zip(freqs.keys(), sizes)}

def normalize_colors(freqs, cmap_name="Blues"):
    values = np.array(list(freqs.values()), dtype=float)
    cmap = cm.get_cmap(cmap_name)

    if values.max() == values.min():
        return {k: cmap(0.7) for k in freqs}

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    return {k: cmap(norm(v)) for k, v in freqs.items()}

def main():
    # Load trained word2vec vectorizer
    vectorizer = joblib.load("readmission_w2v_vectorizer.joblib")

    # Get frequencies from your data
    freqs = get_word_frequencies("data/train.csv", WORDS)

    # Keep only words found in embedding vocab
    valid_words = [w for w in WORDS if w in vectorizer.w2v.wv.key_to_index]
    if len(valid_words) < 2:
        raise ValueError("Not enough words found in the Word2Vec vocabulary.")

    X = np.array([vectorizer.w2v.wv[w] for w in valid_words])

    # Reduce to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    # Sizes/colors by frequency
    valid_freqs = {w: freqs[w] for w in valid_words}
    word_sizes = normalize_sizes(valid_freqs, min_size=10, max_size=28)
    # word_colors = normalize_colors(valid_freqs, cmap_name="Blues")

    plt.figure(figsize=(12, 9))

    # Stronger colormap (darker range)
    def normalize_colors(freqs):
        values = np.array(list(freqs.values()), dtype=float)

        if values.max() == values.min():
            return {k: (0.1, 0.2, 0.7) for k in freqs}  # solid dark blue

        norm = (values - values.min()) / (values.max() - values.min())

        colors = {}
        for k, v in zip(freqs.keys(), norm):
            # darker blue range only
            colors[k] = (0.0, 0.3 + 0.6 * v, 0.6)
        return colors

    word_colors = normalize_colors(valid_freqs)

    # Plot invisible points for spacing
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.0)

    for i, word in enumerate(valid_words):
        plt.text(
            coords[i, 0],
            coords[i, 1],
            word,
            fontsize=word_sizes[word],
            color=word_colors[word],
            ha="center",
            va="center",
            fontweight="bold"
        )

    # Clean look
    plt.title("Semantic Clusters of Clinical Terms", fontsize=14)

    plt.axis("off") 
    plt.tight_layout()

    plt.savefig("semantic_word_map_clean.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nWord frequencies:")
    for w in sorted(valid_words, key=lambda x: valid_freqs[x], reverse=True):
        print(f"{w:15s} {valid_freqs[w]}")

if __name__ == "__main__":
    main()