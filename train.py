"""Train the full-questionnaire DiaCheck model from the original BRFSS 2015 CSV.

Run from the repository root: python train.py

The app collects height and weight, calculates BMI automatically, then rounds
that BMI to the integer BMI field used in the original BRFSS dataset.

Outputs
-------
models/pipeline.joblib        fitted StandardScaler + LogisticRegression
models/model_meta.json        metadata + held-out test metrics
data/fairness_metrics.csv     subgroup metrics for the Fairness page

The feature list written to model_meta.json always matches the fitted
pipeline's own `feature_names_in_`, so app.py can never send a DataFrame
with the wrong columns.
"""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
FAIR_PATH = ROOT / "data" / "fairness_metrics.csv"

# Canonical feature order. The fitted scaler will record this exact list in
# `feature_names_in_`, and that list is what we write to model_meta.json.
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
THRESHOLD = COST_FP / (COST_FP + COST_FN)


def build_metadata(metrics, features, rows_used, dropped_missing):
    return {
        "project": "DiaCheck",
        "model": "StandardScaler + LogisticRegression",
        "version": "2.0-full-questionnaire",
        "target": TARGET,
        "target_description": (
            "Self-reported diabetes status in CDC BRFSS 2015; "
            "not a laboratory-confirmed diagnosis."
        ),
        "data_source": (
            "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 "
            "Diabetes Health Indicators dataset; original imbalanced file."
        ),
        "features": list(features),
        "feature_count": len(features),
        "input_mode": "all_questions",
        "questionnaire": {
            "scope": f"all {len(features)} available BRFSS predictors",
            "mental_health_question": (
                "MentHlth: number of days (0-30) mental health was not good"
            ),
            "physical_health_question": (
                "PhysHlth: number of days (0-30) physical health was not good"
            ),
            "bmi": {
                "mode": "auto",
                "formula": "weight_kg / (height_cm / 100)^2",
                "model_value": (
                    "rounded to nearest integer to match BRFSS BMI coding"
                ),
                "height_unit": "cm",
                "weight_unit": "kg",
            },
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
        "threshold": THRESHOLD,
        "cost_assumptions": {
            "false_negative": COST_FN,
            "false_positive": COST_FP,
            "note": (
                "Illustrative relative costs for a classroom decision "
                "simulator; not clinical or monetary estimates."
            ),
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
            "BRFSS 2015 is a historical US telephone survey and may not "
            "generalize to current populations or other countries.",
            "Inputs are self-reported survey indicators, not verified "
            "clinical measurements.",
            "The model predicts a survey label, not a diagnosis confirmed "
            "by laboratory testing.",
            "Education and income are included as predictors in this "
            "full-questionnaire version; subgroup performance and potential "
            "equity implications should be reviewed.",
            "The score is not clinically calibrated and must not be used to "
            "diagnose, rule out, or treat diabetes.",
        ],
    }


def _subgroup_metrics(df, y_true, y_pred, y_score, group_col, dimension_name):
    """Compute per-group classification metrics for one demographic axis."""
    rows = []
    for group_value, idx in df.groupby(group_col).groups.items():
        yt = y_true.loc[idx]
        yp = y_pred.loc[idx]
        ys = y_score.loc[idx]
        # Skip groups with only one class present — sensitivity/specificity
        # are undefined there and would produce misleading rows.
        if yt.nunique() < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "dimension": dimension_name,
            "group": str(group_value),
            "n": int(len(yt)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "precision": float(tp / (tp + fp)) if (tp + fp) else None,
            "positive_rate": float(yt.mean()),
            "predicted_positive_rate": float(yp.mean()),
            "roc_auc": float(roc_auc_score(yt, ys)),
        })
    return rows


def _check_columns(raw: pd.DataFrame) -> None:
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(
            f"The source CSV is missing required columns: {missing}\n"
            f"Expected columns: {required}"
        )


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Place diabetes_binary_health_indicators_BRFSS2015.csv in "
            "the data/ folder. Download link: "
            "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"
        )

    print(f"Loading {DATA} ...")
    raw = pd.read_csv(DATA)
    _check_columns(raw)

    required = FEATURES + [TARGET]
    complete = raw[required].dropna().copy()
    if complete.empty:
        raise ValueError("No complete observations remain after removing missing values.")

    X = complete[FEATURES].astype(float)
    y = complete[TARGET].astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError(
            f"Expected a binary target coded 0/1; found {sorted(y.unique())}."
        )
    if y.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    print(f"Rows used: {len(complete):,}  (dropped {len(raw) - len(complete):,} with missing values)")
    print(f"Positive prevalence: {y.mean():.3%}")
    print(f"Train/test split: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, random_state={RANDOM_STATE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
    ])
    pipeline.fit(X_train, y_train)

    # Always use the pipeline's own feature names — this guarantees the
    # JSON metadata and the joblib artifact can never disagree.
    fitted_features = list(pipeline.named_steps["scaler"].feature_names_in_)
    if fitted_features != FEATURES:
        raise RuntimeError(
            "The fitted scaler's feature_names_in_ does not match FEATURES. "
            f"Got: {fitted_features}"
        )

    scores = pipeline.predict_proba(X_test)[:, 1]
    predicted = (scores >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()

    metrics = {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "brier_score": float(brier_score_loss(y_test, scores)),
        "threshold": float(THRESHOLD),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "positive_prevalence_test": float(y_test.mean()),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    metadata = build_metadata(
        metrics=metrics,
        features=fitted_features,
        rows_used=int(len(complete)),
        dropped_missing=int(raw[required].isna().any(axis=1).sum()),
    )
    META_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Fairness audit on the exact held-out test set ---------------------
    y_true_series = pd.Series(y_test.values, index=X_test.index)
    y_pred_series = pd.Series(predicted, index=X_test.index)
    y_score_series = pd.Series(scores, index=X_test.index)
    test_frame = X_test.copy()

    group_specs = {
        "Age group": "Age",
        "Sex group": "Sex",
        "Income group": "Income",
        "Education group": "Education",
    }
    fair_rows = []
    for dimension_name, column in group_specs.items():
        fair_rows.extend(
            _subgroup_metrics(
                test_frame, y_true_series, y_pred_series, y_score_series,
                group_col=column,
                dimension_name=dimension_name,
            )
        )

    fair_df = pd.DataFrame(fair_rows)
    FAIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    fair_df.to_csv(FAIR_PATH, index=False)

    # ---- Console summary ---------------------------------------------------
    line = "=" * 72
    print()
    print(line)
    print("DiaCheck training complete")
    print(line)
    print(f"Rows used         : {len(complete):,}")
    print(f"Model saved       : {MODEL_PATH}")
    print(f"Metadata saved    : {META_PATH}")
    print(f"Fairness saved    : {FAIR_PATH}  ({len(fair_df)} subgroup rows)")
    print(f"Features ({len(fitted_features):>2})       : {fitted_features}")
    print(f"Decision threshold: {THRESHOLD:.3f}  (FN={COST_FN}, FP={COST_FP})")
    print(f"ROC-AUC           : {metrics['roc_auc']:.3f}")
    print(f"Average precision : {metrics['average_precision']:.3f}")
    print(f"Brier score       : {metrics['brier_score']:.3f}")
    print(f"Accuracy          : {metrics['accuracy']:.3f}")
    print(f"Precision         : {metrics['precision']:.3f}")
    print(f"Sensitivity       : {metrics['recall_sensitivity']:.3f}")
    print(f"Specificity       : {metrics['specificity']:.3f}")
    print(f"F1                : {metrics['f1']:.3f}")
    print(line)
    print("Next step: streamlit run app.py")
    print(line)


if __name__ == "__main__":
    main()
