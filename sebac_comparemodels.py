import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
import warnings

# === Dati ===
X = np.load("X_train_processed.npy")
y = pd.read_csv("train_labels.csv")["damage_grade"] - 1  # [1,2,3] -> [0,1,2]

warnings.filterwarnings("ignore", category=UserWarning)

# === Classificatore Ordinale ===
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

# === Modelli ===
models = {
    "XGBoost": XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42),
    #"LightGBM": LGBMClassifier(objective="multiclass", num_class=3, random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(loss_function="MultiClass", verbose=0, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    #"SVM": SVC(kernel="rbf", decision_function_shape="ovo", random_state=42),
    #"KNN": KNeighborsClassifier(n_neighbors=5),
    "Ordinal Logistic": OrdinalClassifier(base_estimator=LogisticRegression(max_iter=1000, random_state=42))
}

# === Valutazione ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("📊 Confronto modelli (5-fold CV):")
for name, model in models.items():
    f1_list = []
    acc_list = []
    prec_list = []
    recall_list = []
    conf_matrices = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1_list.append(f1_score(y_val, y_pred, average="micro"))
        acc_list.append(accuracy_score(y_val, y_pred))
        prec_list.append(precision_score(y_val, y_pred, average="macro"))
        recall_list.append(recall_score(y_val, y_pred, average="macro"))
        conf_matrices.append(confusion_matrix(y_val, y_pred, normalize='true'))

    print(f"{name:16s}:")
    print(f"  Confusion matrix media (normalizzata):\n{np.mean(conf_matrices, axis=0)}\n")
    print(f"  Accuracy : {np.mean(acc_list):.4f}")
    print(f"  Precision: {np.mean(prec_list):.4f}")
    print(f"  Recall   : {np.mean(recall_list):.4f}")
    print(f"  F1 micro : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

# === Importanza delle feature con XGBoost ===
print("\n📈 Feature importance da XGBoost (top 15):")
xgb = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42)
xgb.fit(X, y)

importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f"f{i}" for i in range(X.shape[1])]  # se non hai ancora i nomi reali
sorted_names = [feature_names[i] for i in indices]

for i in range(min(15, len(importances))):
    print(f"{sorted_names[i]:<10s}: {importances[indices[i]]:.4f}")