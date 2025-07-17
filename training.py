from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocessing import X_train, y_train, X_val_split, y_val_split, X_train_bal, y_train_bal

# === Modelli base ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='mlogloss', random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# === Ensemble Voting === (con hard voting per stabilità)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
    voting='hard'  # evita problemi con predict_proba
)

# === Training su dati bilanciati ===
ensemble.fit(X_train_bal, y_train_bal)

# === Validazione su set non bilanciato ===
y_val_pred = ensemble.predict(X_val_split)

print("\n=== Valutazione VotingClassifier (hard) ===")
print("Accuracy:", accuracy_score(y_val_split, y_val_pred))
print("F1 micro:", f1_score(y_val_split, y_val_pred, average='micro'))
print("F1 macro:", f1_score(y_val_split, y_val_pred, average='macro'))
print("\nReport dettagliato:\n", classification_report(y_val_split, y_val_pred))
