"""Rebuild the DiaCheck model and evaluation artifacts from the raw CSV."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss)

ROOT=Path(__file__).parent
DATA=ROOT/"data"/"diabetes_binary_health_indicators_BRFSS2015.csv"
FEATURES=['HighBP','HighChol','BMI','Smoker','Stroke','HeartDiseaseorAttack',
          'PhysActivity','GenHlth','Age','Sex']
TARGET='Diabetes_binary'
df=pd.read_csv(DATA)
X=df[FEATURES]
y=df[TARGET].astype(int)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.20,random_state=42,stratify=y)
pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(max_iter=2000,solver="lbfgs"))
])
pipeline.fit(X_train,y_train)
probs=pipeline.predict_proba(X_test)[:,1]
cost_fn,cost_fp=5.0,1.0
threshold=cost_fp/(cost_fp+cost_fn)
pred=(probs>=threshold).astype(int)
tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
metrics={
 "test_size":int(len(y_test)),"roc_auc":float(roc_auc_score(y_test,probs)),
 "average_precision":float(average_precision_score(y_test,probs)),
 "brier_score":float(brier_score_loss(y_test,probs)),"threshold":float(threshold),
 "accuracy":float(accuracy_score(y_test,pred)),
 "precision":float(precision_score(y_test,pred,zero_division=0)),
 "recall_sensitivity":float(recall_score(y_test,pred,zero_division=0)),
 "specificity":float(tn/(tn+fp)),"f1":float(f1_score(y_test,pred,zero_division=0)),
 "confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
 "positive_prevalence_test":float(y_test.mean())
}
(ROOT/"models").mkdir(exist_ok=True)
joblib.dump(pipeline,ROOT/"models"/"pipeline.joblib",compress=3)
meta=json.loads((ROOT/"models"/"model_meta.json").read_text())
meta["metrics"]=metrics
meta["threshold"]=threshold
meta["cost_assumptions"]={"false_negative":cost_fn,"false_positive":cost_fp}
(ROOT/"models"/"model_meta.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(metrics,indent=2))
