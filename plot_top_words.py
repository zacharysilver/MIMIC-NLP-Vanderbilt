import matplotlib.pyplot as plt
from model import LSAVectorizer
import joblib
import numpy as np


def get_lsa_top_words(vectorizer, classifier, top_n=15):
    feature_names = vectorizer.tfidf.get_feature_names_out()
    lsa_coefs = classifier.clf.coef_[0]
    word_space_coefs = vectorizer.svd.components_.T @ lsa_coefs

    top_pos_idx = word_space_coefs.argsort()[-top_n:][::-1]
    top_neg_idx = word_space_coefs.argsort()[:top_n]

    pos_words = [feature_names[i] for i in top_pos_idx]
    pos_vals = [word_space_coefs[i] for i in top_pos_idx]

    neg_words = [feature_names[i] for i in top_neg_idx]
    neg_vals = [word_space_coefs[i] for i in top_neg_idx]

    return pos_words, pos_vals, neg_words, neg_vals


def plot_top_words(pos_words, pos_vals, neg_words, neg_vals, save_path="top_predictive_words.png"):
    words = neg_words[::-1] + pos_words
    vals = neg_vals[::-1] + pos_vals

    plt.figure(figsize=(10, 8))
    plt.barh(words, vals)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Coefficient")
    plt.title("Top Predictive Words: Readmission vs No Readmission")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    lsa_vectorizer = joblib.load("LSA_vectorizer.joblib")
    lsa_classifier = joblib.load("LSA_classifier.joblib")

    pos_words, pos_vals, neg_words, neg_vals = get_lsa_top_words(
        lsa_vectorizer, lsa_classifier, top_n=15
    )
    plot_top_words(pos_words, pos_vals, neg_words, neg_vals)