"""
Train the UPI Scam SMS Detector model.
Run: python train.py
Output: models/upi_scam_detector.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from preprocess import preprocess_text


def load_data(path: str = "data/dataset.csv") -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run  python generate_dataset.py  first."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["processed"] = df["text"].apply(preprocess_text)
    X = df["processed"].values
    y = df["label"].values
    print(f"Loaded {len(df)} samples  |  Scam: {y.sum()}  Legitimate: {(y == 0).sum()}")
    return X, y


def build_pipeline() -> Pipeline:
    """
    FeatureUnion of:
      • Word TF-IDF  (unigrams + bigrams) — captures keyword patterns
      • Char-wb TF-IDF (2-4 grams)        — captures Indian script morphology & subwords
    → Logistic Regression
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=15_000,
        sublinear_tf=True,
        min_df=1,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=20_000,
        sublinear_tf=True,
        min_df=1,
    )
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    pipeline = Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def evaluate(pipeline: Pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Scam"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print("\nTraining model ...")
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test)

    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1", n_jobs=1)
    print(f"\n5-Fold CV  F1 : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Save
    os.makedirs("models", exist_ok=True)
    model_path = "models/upi_scam_detector.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved -> {model_path}")
    return pipeline


if __name__ == "__main__":
    train()
