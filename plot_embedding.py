import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

QUERY_WORDS = [
    "transplant", "cyclosporine", "pancreatic", "lactulose",
    "gout", "angiogram", "kidney", "marrow", "cholangitis",
    "tacrolimus", "lisinopril", "atorvastatin", "thyroid",
    "furosemide", "valve", "potassium", "lasix",
    "pantoprazole", "ulcer", "palpitations"
]

vectorizer = joblib.load("readmission_w2v_vectorizer.joblib")

valid_words = [w for w in QUERY_WORDS if w in vectorizer.w2v.wv.key_to_index]
X = [vectorizer.w2v.wv[w] for w in valid_words]

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1])

for i, word in enumerate(valid_words):
    plt.text(coords[i, 0], coords[i, 1], word)

plt.title("PCA of Word2Vec Embeddings for Selected Clinical Terms")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("embedding_pca_words.png", dpi=300)
plt.show()