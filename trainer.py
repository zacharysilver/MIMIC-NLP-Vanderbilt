# trainer.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

class Trainer:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, train_tokens, train_labels):
        X_train = self.vectorizer.fit_transform(train_tokens)
        self.classifier.fit(X_train, train_labels)

    def evaluate(self, tokens, labels, texts=None, split_name="Val", save_prefix=None):
        X = self.vectorizer.transform(tokens if tokens is not None else texts)
        preds = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)[:, 1]

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        print(f"\n{split_name} Metrics")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")

        # ROC Curve
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

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
  
    
def show_top_features(vectorizer, classifier, top_n=20):
    if not hasattr(vectorizer, "tfidf") or not hasattr(vectorizer, "svd"):
        print("Top words only available for LSA-style vectorizers.")
        return

    feature_names = vectorizer.tfidf.get_feature_names_out()

    # classifier weights in latent LSA space
    lsa_coefs = classifier.clf.coef_[0]   # shape: (n_components,)

    # project weights back to original word space
    word_space_coefs = vectorizer.svd.components_.T @ lsa_coefs
    # shape: (vocab_size,)

    top_pos_idx = word_space_coefs.argsort()[-top_n:][::-1]
    top_neg_idx = word_space_coefs.argsort()[:top_n]

    print("\nTop words predicting READMISSION:")
    for i in top_pos_idx:
        print(feature_names[i], float(word_space_coefs[i]))

    print("\nTop words predicting NO READMISSION:")
    for i in top_neg_idx:
        print(feature_names[i], float(word_space_coefs[i]))