import joblib
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tune_models import tune_xgboost, tune_catboost, tune_randomforest
import os
from lightgbm import LGBMClassifier


# === Carica i dati preprocessati ===
X_train_bal, y_train_bal, X_val_split, y_val_split, X_test_final = joblib.load("data/processed_unprocessed/preprocessed_unbalanced.pkl")

# === Dizionario per salvare i risultati ===
results = {}

# === Random Forest (ottimizzato) ===
if os.path.exists("models/best_rf_model.pkl"):
    best_rf = joblib.load("models/best_rf_model.pkl")
    print("\nModello Random Forest ottimizzato caricato da file.")
else:
    print("\nEsecuzione tuning Random Forest (prima volta)...")
    best_rf = tune_randomforest(X_train_bal, y_train_bal)
    joblib.dump(best_rf, "models/best_rf_model.pkl")
    print("Modello Random Forest ottimizzato salvato in 'best_rf_model.pkl'")

# Addestramento
best_rf.fit(X_train_bal, y_train_bal)
y_pred_rf = best_rf.predict(X_val_split)

print("\nRandom Forest (ottimizzato)")
print("Accuracy:", accuracy_score(y_val_split, y_pred_rf))
print("F1 micro:", f1_score(y_val_split, y_pred_rf, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_rf, average='macro'))
print(classification_report(y_val_split, y_pred_rf))
results['Random Forest'] = f1_score(y_val_split, y_pred_rf, average='macro')

# === Modello 2: XGBoost (ottimizzato) ===
# Se esiste già il file del modello tunato, lo carica; altrimenti lo crea e salva
if os.path.exists("models/best_xgb_model.pkl"):
    best_xgb = joblib.load("models/best_xgb_model.pkl")
    print("\nModello XGBoost ottimizzato caricato da file.")
else:
    print("\nEsecuzione tuning XGBoost (prima volta)...")
    best_xgb = tune_xgboost(X_train_bal, y_train_bal)
    joblib.dump(best_xgb, "models/best_xgb_model.pkl")
    print("Modello XGBoost ottimizzato salvato in 'best_xgb_model.pkl'")

# Addestramento: rimappa le classi per XGBoost
best_xgb.fit(X_train_bal, y_train_bal - 1)

# Predizione: rimappa indietro per confronto corretto
y_pred_best_xgb = best_xgb.predict(X_val_split) + 1

print("\nXGBoost (ottimizzato)")
print("Accuracy:", accuracy_score(y_val_split, y_pred_best_xgb))
print("F1 micro:", f1_score(y_val_split, y_pred_best_xgb, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_best_xgb, average='macro'))
print(classification_report(y_val_split, y_pred_best_xgb))
results['XGBoost'] = f1_score(y_val_split, y_pred_best_xgb, average='macro') 

# === CatBoost (ottimizzato) ===
if os.path.exists("models/best_cat_model.pkl"):
    best_cat = joblib.load("models/best_cat_model.pkl")
    print("\nModello CatBoost ottimizzato caricato da file.")
else:
    print("\nEsecuzione tuning CatBoost (prima volta)...")
    best_cat = tune_catboost(X_train_bal, y_train_bal)
    joblib.dump(best_cat, "models/best_cat_model.pkl")
    print("Modello CatBoost ottimizzato salvato in 'best_cat_model.pkl'")

# Addestramento
best_cat.fit(X_train_bal, y_train_bal)
y_pred_cat = best_cat.predict(X_val_split)

print("\nCatBoost (ottimizzato)")
print("Accuracy:", accuracy_score(y_val_split, y_pred_cat))
print("F1 micro:", f1_score(y_val_split, y_pred_cat, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_cat, average='macro'))
print(classification_report(y_val_split, y_pred_cat))
results['CatBoost'] = f1_score(y_val_split, y_pred_cat, average='macro')


# === Modello 4: LightGBM ===
lgbm = LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    num_class=3  # Perché le classi sono 1,2,3
)

lgbm.fit(X_train_bal, y_train_bal)
y_pred_lgbm = lgbm.predict(X_val_split)

print("\nLightGBM")
print("Accuracy:", accuracy_score(y_val_split, y_pred_lgbm))
print("F1 micro:", f1_score(y_val_split, y_pred_lgbm, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_lgbm, average='macro'))
print(classification_report(y_val_split, y_pred_lgbm))
results['LightGBM'] = f1_score(y_val_split, y_pred_lgbm, average='macro')

from sklearn.ensemble import BaggingClassifier

print("\nCross-Validated BaggingClassifier (CVBaggingClassifier)...")

cv_bagging_clf = BaggingClassifier(
    estimator=RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    n_estimators=5,
    random_state=42,
    n_jobs=-1
)

cv_bagging_clf.fit(X_train_bal, y_train_bal)
y_pred_cvbag = cv_bagging_clf.predict(X_val_split)

print("\nCV BaggingClassifier Performance:")
print("Accuracy:", accuracy_score(y_val_split, y_pred_cvbag))
print("F1 micro:", f1_score(y_val_split, y_pred_cvbag, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_cvbag, average='macro'))
print(classification_report(y_val_split, y_pred_cvbag))

# === Salvataggio del modello già allenato ===
joblib.dump(cv_bagging_clf, "models/cv_bagging_model.pkl")

# === Riepilogo finale ===
print("\nRiepilogo F1 macro:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")
