from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import joblib

# === Carica i dati preprocessati ===
X_train_bal, y_train_bal, X_val_split, y_val_split, _ = joblib.load("preprocessed_data.pkl")

# === Carica modello XGBoost già ottimizzato ===
best_xgb = joblib.load("best_xgb_model.pkl")

# === Inizializza gli altri modelli ===
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
cat = CatBoostClassifier(iterations=100, verbose=0, random_state=42)
lgbm = LGBMClassifier(random_state=42)

# === Ensemble con pesi personalizzati ===
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('cat', cat),
        ('lgbm', lgbm),
        ('xgb', best_xgb)
    ],
    voting='soft',
    weights=[1, 1, 1, 2]  
)

# === Addestramento ensemble ===
voting_clf.fit(X_train_bal, y_train_bal)

# === Valutazione ===
y_pred = voting_clf.predict(X_val_split)

print("\nEnsemble VotingClassifier")
print("Accuracy:", accuracy_score(y_val_split, y_pred))
print("F1 micro:", f1_score(y_val_split, y_pred, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_pred, average='macro'))
print(classification_report(y_val_split, y_pred))

