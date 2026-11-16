"""Train the full-questionnaire DiaCheck model from the original BRFSS 2015 CSV.

Run from the repository root: python train.py
The app collects height and weight, calculates BMI automatically, then maps
that BMI to the integer BMI field used in the original BRFSS dataset.
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
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
TARGET = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE = 0.20
COST_FN, COST_FP = 5.0, 1.0


def build_metadata(metrics, rows_used, dropped_missing):
    return {
        "project": "DiaCheck",
        "model": "StandardScaler + LogisticRegression",
        "version": "2.0-full-questionnaire",
        "target": TARGET,
        "target_description": "Self-reported diabetes status in CDC BRFSS 2015; not a laboratory-confirmed diagnosis.",
        "data_source": "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 Diabetes Health Indicators dataset; original imbalanced file.",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "input_mode": "all_questions",
        "questionnaire": {
            "scope": "all 21 available BRFSS predictors",
            "mental_health_question": "MentHlth: number of days (0-30) mental health was not good",
            "physical_health_question": "PhysHlth: number of days (0-30) physical health was not good",
            "bmi": {"mode": "auto", "formula": "weight_kg / (height_cm / 100)^2", "model_value": "rounded to nearest integer to match BRFSS BMI coding", "height_unit": "cm", "weight_unit": "kg"},
            "result_display": "model_score_and_threshold_guidance",
        },
        "valid_input_ranges": {
            "HighBP": [0, 1], "HighChol": [0, 1], "CholCheck": [0, 1],
            "BMI": [12, 98], "Smoker": [0, 1], "Stroke": [0, 1],
            "HeartDiseaseorAttack": [0, 1], "PhysActivity": [0, 1],
            "Fruits": [0, 1], "Veggies": [0, 1], "HvyAlcoholConsump": [0, 1],
            "AnyHealthcare": [0, 1], "NoDocbcCost": [0, 1], "GenHlth": [1, 5],
            "MentHlth": [0, 30], "PhysHlth": [0, 30], "DiffWalk": [0, 1],
            "Sex": [0, 1], "Age": [1, 13], "Education": [1, 6], "Income": [1, 8],
        },
        "threshold": COST_FP / (COST_FP + COST_FN),
        "cost_assumptions": {
            "false_negative": COST_FN,
            "false_positive": COST_FP,
            "note": "Illustrative relative costs for a classroom decision simulator; not clinical or monetary estimates.",
        },
        "metrics": metrics,
        "training": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "split": "stratified train/test split",
            "rows_used": rows_used,
            "dropped_missing_rows": dropped_missing,
        },
        "limitations": [
            "BRFSS 2015 is a historical US telephone survey and may not generalize to current populations or other countries.",
            "Inputs are self-reported survey indicators, not verified clinical measurements.",
            "The model predicts a survey label, not a diagnosis confirmed by laboratory testing.",
            "Education and income are included as predictors in this full-questionnaire version; subgroup performance and potential equity implications should be reviewed.",
            "The score is not clinically calibrated and must not be used to diagnose, rule out, or treat diabetes.",
        ],
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Place diabetes_binary_health_indicators_BRFSS2015.csv in the data/ folder."
        )

    raw = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"The source CSV is missing required columns: {missing}")
    complete = raw[required].dropna().copy()
    if complete.empty:
        raise ValueError("No complete observations remain after removing missing values.")

    X = complete[FEATURES].astype(float)
    y = complete[TARGET].astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError(f"Expected a binary target coded 0/1; found {sorted(y.unique())}.")
    if y.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
    ])
    pipeline.fit(X_train, y_train)
    scores = pipeline.predict_proba(X_test)[:, 1]

    threshold = COST_FP / (COST_FP + COST_FN)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    metrics = {
        "train_size": int(len(y_train)), "test_size": int(len(y_test)),
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
    metadata = build_metadata(
        metrics=metrics,
        rows_used=int(len(complete)),
        dropped_missing=int(raw[required].isna().any(axis=1).sum()),
    )
    META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Rebuild subgroup metrics from this exact held-out test set so the Fairness
    # page cannot accidentally display results from the previous 10-feature model.
    fairness_rows = []
    group_specs = {
        "Age group": ("Age", {i: f"Age code {i}" for i in range(1, 14)}),
        "Sex group": ("Sex", {0: "Female", 1: "Male"}),
        "Income group": ("Income", {i: f"Income code {i}" for i in range(1, 9)}),
        "Education group": ("Education", {i: f"Education code {i}" for i in range(1, 7)}),
    }
    test_frame = X_test.copy()
    test_frame["_actual"] = y_test.to_numpy()
    test_frame["_predicted"] = predicted
    for dimension, (column, labels) in group_specs.items():
        for code, group_label in labels.items():
            subset = test_frame[test_frame[column] == code]
            if subset.empty:
                continue
            actual = subset["_actual"].to_numpy(dtype=int)
            pred = subset["_predicted"].to_numpy(dtype=int)
            tp_g = int(((actual == 1) & (pred == 1)).sum())
            tn_g = int(((actual == 0) & (pred == 0)).sum())
            fp_g = int(((actual == 0) & (pred == 1)).sum())
            fn_g = int(((actual == 1) & (pred == 0)).sum())
            fairness_rows.append({
                "dimension": dimension, "group": group_label, "n": int(len(subset)),
                "sensitivity": tp_g / (tp_g + fn_g) if (tp_g + fn_g) else np.nan,
                "specificity": tn_g / (tn_g + fp_g) if (tn_g + fp_g) else np.nan,
                "precision": tp_g / (tp_g + fp_g) if (tp_g + fp_g) else np.nan,
                "positive_rate": float(actual.mean()),
                "predicted_positive_rate": float(pred.mean()),
            })
    pd.DataFrame(fairness_rows).to_csv(ROOT / "data" / "fairness_metrics.csv", index=False)

    print("DiaCheck full-questionnaire model training completed.")
    print(f"Features: {len(FEATURES)} | Rows used: {len(complete):,} | Train: {len(y_train):,} | Test: {len(y_test):,}")
    print(json.dumps(metrics, indent=2))
    fairness_path = ROOT / "data" / "fairness_metrics.csv"
    if not fairness_path.exists():
        warnings.warn("data/fairness_metrics.csv is missing. The Fairness page in app.py needs this file.", RuntimeWarning)


if __name__ == "__main__":
    main()
