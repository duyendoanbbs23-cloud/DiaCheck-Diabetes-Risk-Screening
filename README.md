# DiaCheck — T3 Diabetes Risk Screening

A multilingual Streamlit healthcare decision-support prototype for the T3 assignment.

> **User-decision statement:** An adult answering questions at home uses the app to decide whether to get a blood sugar test.

## Features
- Single-case screening form with a plain-language recommendation.
- Decision threshold and false-positive / false-negative cost simulator.
- Case-specific explanation using logistic-regression feature contributions.
- About page and model card with test metrics, limitations and medical disclaimer.
- Fairness audit by age, sex, income and education groups.
- Six UI languages: English, Vietnamese, Chinese, Japanese, Korean and Spanish.

## Dataset
CDC BRFSS 2015 Diabetes Health Indicators, Kaggle. This project uses the **original imbalanced binary dataset** (`diabetes_binary_health_indicators_BRFSS2015.csv`), not the 50/50 balanced file and not the three-class target. The dataset has 253,680 responses and 21 predictors plus the target.

## Model
- Pipeline: `StandardScaler` + `LogisticRegression`.
- Ten selected predictors: `HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, GenHlth, Age, Sex`.
- Stratified 80/20 train/test split, random state 42.
- No oversampling or class weighting.
- Default decision threshold: 0.167, from illustrative relative costs FN=5 and FP=1 using `C_FP / (C_FP + C_FN)`.
- Income and education are excluded from model inputs and used only for subgroup auditing.

### Held-out test metrics
| Metric | Value |
|---|---:|
| ROC-AUC | 0.815 |
| Average precision | 0.388 |
| Brier score | 0.101 |
| Accuracy | 0.758 |
| Precision | 0.328 |
| Sensitivity / recall | 0.698 |
| Specificity | 0.768 |
| F1 | 0.446 |

These are test-set estimates from one split, not external validation. Threshold-dependent metrics depend on the stated cost assumptions.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild model
The raw CSV is included in `data/`. Rebuild the fitted pipeline and metadata:
```bash
python train.py
```

## Deploy on Streamlit Community Cloud
1. Push this repository to a **public GitHub repository**.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose this repository, branch `main`, and main file `app.py`.
4. Deploy. The repository includes `requirements.txt` and `runtime.txt`.

## Important limitations and safety
- The source is a 2015 US telephone survey. It may not represent current populations or people outside the US.
- Indicators are self-reported, not verified clinical measurements.
- The target is a survey label, not a laboratory-confirmed diagnosis.
- This app uses a short feature subset; a smaller questionnaire improves usability but may sacrifice predictive information.
- Subgroup metrics are descriptive and do not prove fairness. Sample sizes and uncertainty matter.
- The score should not be interpreted as a clinically calibrated individual probability.

**Medical disclaimer:** DiaCheck is an educational student project, not a medical device and not for clinical use. It cannot diagnose, rule out, or treat diabetes. Users should not delay care or make health decisions solely from this result. A qualified healthcare professional and appropriate blood glucose testing are needed.

## Decision log (initial)
| Decision | Choice | Rationale / rejected alternative |
|---|---|---|
| Dataset | Original imbalanced binary BRFSS file | Matches assignment instructions; rejected 50/50 variant because it changes prevalence. |
| Target | `Diabetes_binary` | Binary diabetes/no-diabetes classification. |
| Inputs | Ten short-form features | Balances a manageable public-facing questionnaire with predictive signal. |
| Split | Stratified random 80/20 | Preserves outcome prevalence in train and test; one respondent per row in this dataset. |
| Model | StandardScaler + logistic regression | Transparent linear model supports case-level contribution explanations. |
| Threshold | FN cost 5, FP cost 1; threshold 0.167 | Illustrative assumption prioritizes avoiding missed positive survey labels; users can change costs in-app. |
| Fairness | Age, sex, income, education groups | Income and education are audit-only variables, not predictors. |

## AI use appendix (template to complete)
AI assistance was used to scaffold app structure, translation dictionaries, model training code, and documentation. The team must review, test, and understand all generated code, document any changes, and verify that no leakage or misleading clinical claims remain before submission.
