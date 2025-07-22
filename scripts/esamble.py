from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import mode
import numpy as np
import joblib

# === Caricamento dati ===
X_train_bal, y_train_bal, X_val_split, y_val_split, _ = joblib.load("data/processed_unprocessed/preprocessed_unbalanced.pkl")
best_xgb = joblib.load("models/best_xgb_model.pkl")
cv_bagging_clf = joblib.load("models/cv_bagging_model.pkl")  # Carica modello già addestrato

# === Bagging di VotingClassifier ===
print("\nBagging VotingClassifier (5 seeds)...")

predictions = []

for seed in [0, 1, 2, 3, 4]:
    rf_bag = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed)
    cat_bag = CatBoostClassifier(iterations=100, verbose=0, random_state=seed)
    #lgbm_bag = LGBMClassifier(random_state=seed)

    voting_bag = VotingClassifier(
        estimators=[
            ('rf', rf_bag),
            ('cat', cat_bag),
            ('xgb', best_xgb),
            ('cvb', cv_bagging_clf),
            #('lgbm', lgbm_bag)
        ],
        voting='soft',
        weights=(1, 1, 2, 1.5)#, 0.3)
    )

    voting_bag.fit(X_train_bal, y_train_bal)
    preds = voting_bag.predict(X_val_split)
    predictions.append(preds)

# === Maggioranza delle predizioni ===
y_pred_bagged = mode(np.array(predictions), axis=0).mode.squeeze()

print("\nBagged VotingClassifier Performance (modalità su 5 modelli):")
print("Accuracy:", accuracy_score(y_val_split, y_pred_bagged))
print("F1 micro:", f1_score(y_val_split, y_pred_bagged, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_bagged, average='macro'))
print(classification_report(y_val_split, y_pred_bagged))
