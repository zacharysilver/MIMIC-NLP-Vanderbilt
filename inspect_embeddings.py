import joblib

vectorizer = joblib.load("readmission_w2v_vectorizer.joblib")

query_words = [
    "transplant", "cyclosporine", "pancreatic", "lactulose",
    "gout", "angiogram", "kidney", "marrow", "cholangitis",
    "tacrolimus", "lisinopril", "atorvastatin", "thyroid",
    "furosemide", "valve", "potassium", "lasix",
    "pantoprazole", "ulcer", "palpitations"
]

for word in query_words:
    print(f"\n=== Most similar to '{word}' ===")
    if word in vectorizer.w2v.wv.key_to_index:
        for neighbor, score in vectorizer.w2v.wv.most_similar(word, topn=10):
            print(f"{neighbor:20s} {score:.4f}")
    else:
        print(f"'{word}' not found in vocabulary")