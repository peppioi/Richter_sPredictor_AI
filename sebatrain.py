import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import joblib

# === Caricamento oggetti preprocessati ===
X_train = np.load("X_train_processed.npy")
X_test = np.load("X_test_processed.npy")
y_train = pd.read_csv("train_labels.csv")["damage_grade"] - 1  # [1,2,3] → [0,1,2]
test_ids = pd.read_csv("test_values.csv")["building_id"]
submission_format = pd.read_csv("submission_format.csv")

classifier = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)


# === Valutazione con cross-validation ===
f1_micro = make_scorer(f1_score, average='micro')
scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring=f1_micro)
print(f"F1 micro (CV): {scores.mean():.4f} ± {scores.std():.4f}")

# === Addestramento finale e predizione ===
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = y_pred + 1  # riportiamo le classi a [1,2,3]

# === Generazione submission ===
submission = submission_format.copy()
submission["damage_grade"] = y_pred
submission.to_csv("submission_pipeline.csv", index=False)
print("✅ Submission salvata in 'submission_pipeline.csv'")

# === Salvataggio modello ===
joblib.dump(classifier, "xgb_model.joblib")
print("✅ Modello salvato in 'xgb_model.joblib'")
