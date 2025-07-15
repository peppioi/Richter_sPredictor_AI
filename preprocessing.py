import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# === 1. Caricamento dati ===
train_values = pd.read_csv("train_values.csv")
train_labels = pd.read_csv("train_labels.csv")
test_values = pd.read_csv("test_values.csv")

# === 2. Analisi iniziale ===
print("Train values shape:", train_values.shape)
print("Train labels shape:", train_labels.shape)
print("Test values shape:", test_values.shape)

print("\nMissing values in train_values:\n", train_values.isnull().sum().sum())
print("Missing values in train_labels:\n", train_labels.isnull().sum().sum())
print("Missing values in test_values:\n", test_values.isnull().sum().sum())

# === 3. Merge feature e label ===
train_df = pd.merge(train_values, train_labels, on="building_id")

# === 4. Separazione X, y e test ===
X_train = train_df.drop(columns=["building_id", "damage_grade"])
y_train = train_df["damage_grade"]
X_test = test_values.drop(columns=["building_id"])

# === 5. Riconoscimento automatico delle colonne ===
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
numeric_cols = X_train.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric/binary columns:", numeric_cols)

# === 6. Winsorization per contenere gli outlier numerici ===
def winsorize(df, cols, lower_percentile=0.01, upper_percentile=0.99):
    capped_df = df.copy()
    for col in cols:
        lower = df[col].quantile(lower_percentile)
        upper = df[col].quantile(upper_percentile)
        capped_df[col] = np.clip(df[col], lower, upper)
    return capped_df

# Applica solo alle colonne numeriche, e riassegna una per una
X_train_wins = winsorize(X_train[numeric_cols], numeric_cols)
X_test_wins = winsorize(X_test[numeric_cols], numeric_cols)

# Sostituisci le colonne numeriche originali con quelle winsorizzate
X_train[numeric_cols] = X_train_wins
X_test[numeric_cols] = X_test_wins


# === 7. OneHotEncoding sulle categoriche ===
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

# === 8. Ricostruzione dei DataFrame encoded ===
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoded_col_names, index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoded_col_names, index=X_test.index)

# === 9. Scaling delle colonne numeriche ===
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_num_scaled = scaler.transform(X_test[numeric_cols])

X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=numeric_cols, index=X_train.index)
X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=numeric_cols, index=X_test.index)

# === 10. Unione numeriche scalate + categoriche codificate ===
X_train_final = pd.concat([X_train_num_df.reset_index(drop=True), 
                           X_train_cat_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_num_df.reset_index(drop=True), 
                          X_test_cat_df.reset_index(drop=True)], axis=1)

# === 11. Verifica finale ===
print("\nFinal train shape:", X_train_final.shape)
print("Final test shape:", X_test_final.shape)
print("y_train distribution:\n", y_train.value_counts(normalize=True))
print("y_train distribution:\n", y_train.value_counts(normalize=True))

# === 12. Applica SMOTE per bilanciare le classi ===
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)

# === 13. Verifica nuova distribuzione ===
print("\nDistribuzione delle classi dopo SMOTE:")
print(y_train_balanced.value_counts(normalize=True))

