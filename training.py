from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from preprocessing import X_train_balanced, y_train_balanced, X_test_final

# === 1. Training ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# === 2. Valutazione su train (solo esempio, meglio validare con split) ===
y_pred_train = model.predict(X_train_balanced)
print("\n=== Valutazione su dati di training ===")
print("Accuracy:", accuracy_score(y_train_balanced, y_pred_train))
print("F1 (micro):", f1_score(y_train_balanced, y_pred_train, average='micro'))
print("F1 (macro):", f1_score(y_train_balanced, y_pred_train, average='macro'))
print("\nReport dettagliato:\n", classification_report(y_train_balanced, y_pred_train))

# === 3. Previsione sul test set ===
y_test_pred = model.predict(X_test_final)

# === 4. Creazione del file di submission ===
submission = pd.read_csv("submission_format.csv")  # contiene solo building_id
submission["damage_grade"] = y_test_pred
submission.to_csv("submission.csv", index=False)
print("\nFile 'submission.csv' creato con successo!")
