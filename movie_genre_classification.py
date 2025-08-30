"""
Movie Genre Classification
- Input CSV with at least two columns: text column (e.g., "plot") and target column (e.g., "genre").
- Uses TF-IDF + linear classifiers (Naive Bayes, Logistic Regression, Linear SVM).
- Picks the best model by macro F1 on validation set.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def build_pipelines():
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_features=100_000
    )
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=None, class_weight=None),
        "LinearSVC": LinearSVC()
    }
    pipelines = {name: Pipeline([("tfidf", tfidf), ("clf", model)]) for name, model in models.items()}
    return pipelines

def main(args):
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.target_col not in df.columns:
        raise ValueError(f"Columns not found. Available: {list(df.columns)}")
    X = df[args.text_col].astype(str).fillna("")
    y = df[args.target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name = None
    best_score = -1.0
    best_pipe = None
    cv_results = {}

    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        print(f"[CV] {name}: F1_macro={scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_pipe = pipe

    # Fit best pipeline on full train
    best_pipe.fit(X_train, y_train)
    preds = best_pipe.predict(X_test)

    print("\n=== Test Results ===")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save model and metadata
    out_path = args.model_out
    meta = {
        "best_model": best_name,
        "cv_results": {k: {"mean_f1_macro": v[0], "std": v[1]} for k, v in cv_results.items()},
        "text_col": args.text_col,
        "target_col": args.target_col,
        "test_size": args.test_size,
    }
    joblib.dump({"pipeline": best_pipe, "meta": meta}, out_path)
    print(f"\nSaved model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with text and target columns.")
    parser.add_argument("--text_col", default="plot")
    parser.add_argument("--target_col", default="genre")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model_out", default="movie_genre_model.joblib")
    args = parser.parse_args()
    main(args)
