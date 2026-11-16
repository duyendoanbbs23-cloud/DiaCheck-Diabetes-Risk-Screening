# Team decision log — DiaCheck T3

| Date | Decision | Chosen option | Reason | Rejected option |
|---|---|---|---|---|
| 2026-11-16 | Dataset version | Original imbalanced binary dataset | Matches the assignment's explicit file recommendation and preserves natural prevalence | 50/50 balanced sample; three-class target |
| 2026-11-16 | Predictors | Ten-question short form | Designed for a member of the public; income and education reserved for fairness audit | All 21 predictors, which would make the form longer |
| 2026-11-16 | Model | StandardScaler + LogisticRegression | Transparent and supports additive per-case contributions | Opaque model requiring SHAP dependency |
| 2026-11-16 | Evaluation | Stratified 80/20 split, random_state=42 | Reproducible, preserves class share | Unstratified split |
| 2026-11-16 | Threshold | FN cost 5, FP cost 1 (threshold 0.167) | Explicitly makes missed positive labels more costly in this educational scenario | Default 0.5 without cost justification |
| 2026-11-16 | Fairness audit | Age, sex, income, education | Includes socioeconomic group checks; income and education not model inputs | Omitting subgroup review |
