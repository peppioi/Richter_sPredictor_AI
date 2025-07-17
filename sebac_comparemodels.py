import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# === Dati ===
X = np.load("X_train_processed.npy")
y = pd.read_csv("train_labels.csv")["damage_grade"] - 1  # [1,2,3] -> [0,1,2]

f1_micro = make_scorer(f1_score, average="micro")

# === Modelli ===
models = {
    "XGBoost": XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42),
    "LightGBM": LGBMClassifier(objective="multiclass", num_class=3, random_state=42),
    "CatBoost": CatBoostClassifier(loss_function="MultiClass", verbose=0, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# === Valutazione ===
print("📊 Confronto modelli (F1 micro, 5-fold CV):")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring=f1_micro)
    print(f"{name:12s}: {scores.mean():.4f} ± {scores.std():.4f}")
