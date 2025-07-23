import pandas as pd
import joblib

# === Caricamento dati preprocessati ===
print("Caricamento dati preprocessati...")
_, _, _, _, X_test_final = joblib.load("data/processed_unprocessed/preprocessed_data.pkl")

# === Caricamento building_id dal file originale ===
test_values = pd.read_csv("data/raw/test_values.csv")
building_ids = test_values["building_id"]

# === Caricamento modello ensemble ===
print("Caricamento modello ensemble...")
ensemble_model = joblib.load("models/ensemble_model.pkl")

# === Predizione sul test set ===
print("Generazione predizioni sul test set...")
y_test_pred = ensemble_model.predict(X_test_final)

# === Creazione file di submission ===
submission = pd.DataFrame({
    "building_id": building_ids,
    "damage_grade": y_test_pred
})

# === Salvataggio submission ===
submission.to_csv("submission.csv", index=False)
print("Submission salvata in 'submission.csv'")
