# main.py
import argparse
import joblib
import dataloader

from model import Word2VecVectorizer, LSAVectorizer, ReadmissionClassifier
from trainer import Trainer, show_top_features
from dataloader import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--vector_size", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--save_prefix", type=str, default="readmission_w2v")
    return parser.parse_args()

def run_experiment(name, vectorizer, train_texts, train_tokens, train_labels,
                   val_texts, val_tokens, val_labels,
                   test_texts, test_tokens, test_labels):

    print(f"\n===== Running {name} =====")

    if name == "LSA":
        X_train = vectorizer.fit_transform(train_texts)
    else:
        X_train = vectorizer.fit_transform(train_tokens)

    classifier = ReadmissionClassifier()
    classifier.fit(X_train, train_labels)

    trainer = Trainer(vectorizer, classifier)

    trainer.evaluate(val_tokens if name != "LSA" else None,
                     val_labels,
                     texts=val_texts,
                     split_name="Validation",
                     save_prefix=name)

    trainer.evaluate(test_tokens if name != "LSA" else None,
                     test_labels,
                     texts=test_texts,
                     split_name="Test",
                     save_prefix=name)

    if name == "LSA":
        show_top_features(vectorizer, classifier)


def main():
    args = parse_args()

    print("Using dataloader from:", dataloader.__file__)
    print("load_dataset object:", load_dataset)

    out = load_dataset(args.train_csv)
    print("Number of returned values from load_dataset:", len(out))

    _, train_texts, train_tokens, train_labels = out
    _, val_texts, val_tokens, val_labels = load_dataset(args.val_csv)
    _, test_texts, test_tokens, test_labels = load_dataset(args.test_csv)
    # WORD2VEC
    w2v = Word2VecVectorizer(vector_size=200)
    run_experiment("Word2Vec", w2v,
                   train_texts, train_tokens, train_labels,
                   val_texts, val_tokens, val_labels,
                   test_texts, test_tokens, test_labels)

    # LSA
    lsa = LSAVectorizer(n_components=200)
    run_experiment("LSA", lsa,
                   train_texts, train_tokens, train_labels,
                   val_texts, val_tokens, val_labels,
                   test_texts, test_tokens, test_labels)
    

if __name__ == "__main__":
    main()