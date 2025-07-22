from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import mode
import numpy as np
import joblib

# === Caricamento dati ===
X_train_bal, y_train_bal, X_val_split, y_val_split, _ = joblib.load("data/processed_unprocessed/preprocessed_unbalanced.pkl")
best_xgb = joblib.load("models/best_xgb_model.pkl")
best_rf = joblib.load("models/best_rf_model.pkl")
best_cat = joblib.load("models/best_cat_model.pkl")
cv_bagging_clf = joblib.load("models/cv_bagging_model.pkl")  # Carica modello già addestrato

# === Bagging di VotingClassifier ===
print("\nBagging VotingClassifier (5 seeds)...")

predictions = []

for seed in [0, 1, 2, 3, 4]:
    # Usiamo direttamente i modelli ottimizzati caricati da file
    # Non usiamo versioni randomizzate per rf e cat, per coerenza
    voting_bag = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('cat', best_cat),
            ('xgb', best_xgb),
            ('cvb', cv_bagging_clf),
            # ('lgbm', LGBMClassifier(random_state=seed))  # puoi riattivarlo se vuoi
        ],
        voting='soft',
        weights=(1.5, 1, 3, 2.5)
    )

    voting_bag.fit(X_train_bal, y_train_bal)
    preds = voting_bag.predict(X_val_split)
    predictions.append(preds)

    if seed == 0:
        final_model = voting_bag  # Salva il primo ensemble


joblib.dump(final_model, "models/ensemble_model.pkl")
print("\nModello ensemble salvato in 'models/ensemble_model.pkl'")


# === Maggioranza delle predizioni ===
y_pred_bagged = mode(np.array(predictions), axis=0).mode.squeeze()

print("\nBagged VotingClassifier Performance (modalità su 5 modelli):")
print("Accuracy:", accuracy_score(y_val_split, y_pred_bagged))
print("F1 micro:", f1_score(y_val_split, y_pred_bagged, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_bagged, average='macro'))
print(classification_report(y_val_split, y_pred_bagged))
