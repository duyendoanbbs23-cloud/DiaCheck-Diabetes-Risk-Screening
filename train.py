"""Rebuild the DiaCheck educational screening model from the original BRFSS CSV.

Run from the project root with: python train.py
This script intentionally keeps the existing 10 model inputs. Adding survey
questions to the interface does not make them model predictors unless they are
added here and the model is retrained and evaluated.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
FEATURES = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "GenHlth", "Age", "Sex",
]
TARGET = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE = 0.20
COST_FN, COST_FP = 5.0, 1.0


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Place diabetes_binary_health_indicators_BRFSS2015.csv in the data/ folder."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {META_PATH}. Keep models/model_meta.json in the repository."
        )

    df = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"The source CSV is missing required columns: {missing}")
    df = df[required].dropna().copy()
    if df.empty:
        raise ValueError("No complete observations remain after removing missing values.")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError(f"Expected a binary target coded 0/1; found {sorted(y.unique())}.")
    if y.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])
    pipeline.fit(X_train, y_train)
    scores = pipeline.predict_proba(X_test)[:, 1]

    # Illustrative decision-analysis assumptions, not clinical guidance.
    threshold = COST_FP / (COST_FP + COST_FN)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    metrics = {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "brier_score": float(brier_score_loss(y_test, scores)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_prevalence_test": float(y_test.mean()),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    metadata["features"] = FEATURES
    metadata["metrics"] = metrics
    metadata["threshold"] = threshold
    metadata["cost_assumptions"] = {
        "false_negative": COST_FN,
        "false_positive": COST_FP,
        "note": "Illustrative relative costs for a classroom decision simulator; not clinical or monetary estimates.",
    }
    metadata["training"] = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split": "stratified train/test split",
        "rows_used": int(len(df)),
        "dropped_missing_rows": int(pd.read_csv(DATA, usecols=required).isna().any(axis=1).sum()),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DiaCheck model training completed.")
    print(f"Rows used: {len(df):,} | Train: {len(y_train):,} | Test: {len(y_test):,}")
    print(json.dumps(metrics, indent=2))
    fairness_path = ROOT / "data" / "fairness_metrics.csv"
    if not fairness_path.exists():
        warnings.warn(
            "data/fairness_metrics.csv is missing. The Fairness page in app.py needs this file.",
            RuntimeWarning,
        )


if __name__ == "__main__":
    main()
