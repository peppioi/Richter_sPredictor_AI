import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import joblib

# === Caricamento dati preprocessati ===
X_train = np.load("X_train_processed.npy")
X_test = np.load("X_test_processed.npy")
y_train = pd.read_csv("train_labels.csv")["damage_grade"] - 1  # [1,2,3] → [0,1,2]
test_ids = pd.read_csv("test_values.csv")["building_id"]
submission_format = pd.read_csv("submission_format.csv")

# === Classificatore ordinale personalizzato ===
class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator if base_estimator else LogisticRegression()

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for i in range(len(self.classes_) - 1):
            y_bin = (y > self.classes_[i]).astype(int)
            model = clone(self.base_estimator)
            model.fit(X, y_bin)
            self.models_.append(model)
        return self

    def predict(self, X):
        preds = np.zeros((X.shape[0], len(self.models_)))
        for i, model in enumerate(self.models_):
            if hasattr(model, "predict_proba"):
                preds[:, i] = model.predict_proba(X)[:, 1]
            else:
                preds[:, i] = model.predict(X)
        return (preds > 0.5).sum(axis=1)

# === Definizione modelli per ensemble ===
xgb = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42)
cat = CatBoostClassifier(loss_function="MultiClass", verbose=0, random_state=42)
ordlog = OrdinalClassifier(base_estimator=LogisticRegression(max_iter=1000, random_state=42))

# === Voting ensemble ===
ensemble = VotingClassifier(
    estimators=[
        ("xgb", xgb),
        ("catboost", cat),
        ("ordinal", ordlog)
    ],
    voting="hard"
)

# === Addestramento e predizione ===
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
y_pred = y_pred + 1  # ritorniamo da [0,1,2] a [1,2,3]

# === Generazione submission ===
submission = submission_format.copy()
submission["damage_grade"] = y_pred
submission.to_csv("submission_ensemble.csv", index=False)
print("✅ Submission salvata in 'submission_ensemble.csv'")

# === Salvataggio ensemble ===
joblib.dump(ensemble, "ensemble_model.joblib")
print("✅ Ensemble salvato in 'ensemble_model.joblib'")