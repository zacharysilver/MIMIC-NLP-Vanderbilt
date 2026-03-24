# model.py

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LSAVectorizer:
    def __init__(self, max_features=10000, n_components=200):
        custom_stopwords = [
            "mg", "bid", "sig", "po", "iv", "qd", "tid",
            "day", "daily", "tablet", "tab",
            "patient", "pt", "he", "she", "on", "and"
        ]
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=5,
            max_df=0.9
        )    
        self.tfidf.set_params(
            stop_words=list(self.tfidf.get_stop_words()) + custom_stopwords
        )    
        self.svd = TruncatedSVD(n_components=n_components)

    def fit(self, texts):
        X_tfidf = self.tfidf.fit_transform(texts)
        self.svd.fit(X_tfidf)
        return self

    def transform(self, texts):
        X_tfidf = self.tfidf.transform(texts)
        return self.svd.transform(X_tfidf)

    def fit_transform(self, texts):
        X_tfidf = self.tfidf.fit_transform(texts)
        return self.svd.fit_transform(X_tfidf)
    
class Word2VecVectorizer:
    def __init__(self, vector_size=200, window=5, min_count=2, workers=4, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.w2v = None

    def fit(self, tokenized_texts):
        self.w2v = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,   # 1 = skip-gram, 0 = CBOW
        )
        return self

    def transform(self, tokenized_texts):
        vectors = []
        for tokens in tokenized_texts:
            word_vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
            if len(word_vecs) == 0:
                vec = np.zeros(self.vector_size, dtype=np.float32)
            else:
                vec = np.mean(word_vecs, axis=0)
            vectors.append(vec)
        return np.array(vectors)

    def fit_transform(self, tokenized_texts):
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)


class ReadmissionClassifier:
    def __init__(self, random_state=42, max_iter=1000):
        self.clf = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)