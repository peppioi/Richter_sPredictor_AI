import joblib
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgb_tuning import tune_xgboost
import os
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# === Carica i dati preprocessati ===
X_train_bal, y_train_bal, X_val_split, y_val_split, _ = joblib.load("preprocessed_data.pkl")

# === Dizionario per salvare i risultati ===
results = {}

# === Modello 1: Random Forest con class_weight ===
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_val_split)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_val_split, y_pred_rf))
print("F1 micro:", f1_score(y_val_split, y_pred_rf, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_rf, average='macro'))
print(classification_report(y_val_split, y_pred_rf))
results['Random Forest'] = f1_score(y_val_split, y_pred_rf, average='macro')

# === Modello 2: XGBoost (ottimizzato) ===
# Se esiste già il file del modello tunato, lo carica; altrimenti lo crea e salva
if os.path.exists("best_xgb_model.pkl"):
    best_xgb = joblib.load("best_xgb_model.pkl")
    print("\nModello XGBoost ottimizzato caricato da file.")
else:
    print("\nEsecuzione tuning XGBoost (prima volta)...")
    best_xgb = tune_xgboost(X_train_bal, y_train_bal)
    joblib.dump(best_xgb, "best_xgb_model.pkl")
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

# === Modello 3: CatBoost ===
cat = CatBoostClassifier(iterations=100, verbose=0, random_state=42)
cat.fit(X_train_bal, y_train_bal)
y_pred_cat = cat.predict(X_val_split)

print("\nCatBoost")
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

# === Modello 5: MLPClassifier (NON UTILE PERFORMANCE TROPPO BASSE)===
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
mlp.fit(X_train_bal, y_train_bal)
y_pred_mlp = mlp.predict(X_val_split)

print("\nMLPClassifier")
print("Accuracy:", accuracy_score(y_val_split, y_pred_mlp))
print("F1 micro:", f1_score(y_val_split, y_pred_mlp, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred_mlp, average='macro'))
print(classification_report(y_val_split, y_pred_mlp))
results['MLPClassifier'] = f1_score(y_val_split, y_pred_mlp, average='macro')

# === Riepilogo finale ===
print("\nRiepilogo F1 macro:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")
