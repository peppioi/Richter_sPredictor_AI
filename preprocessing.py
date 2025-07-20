import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif


# ===Caricamento dati ===
train_values = pd.read_csv("train_values.csv")
train_labels = pd.read_csv("train_labels.csv")
test_values = pd.read_csv("test_values.csv")

# ===Analisi iniziale ===
print("Train values shape:", train_values.shape)
print("Train labels shape:", train_labels.shape)
print("Test values shape:", test_values.shape)

print("\nMissing values in train_values:\n", train_values.isnull().sum().sum())
print("Missing values in train_labels:\n", train_labels.isnull().sum().sum())
print("Missing values in test_values:\n", test_values.isnull().sum().sum())

# ===Merge feature e label ===
train_df = pd.merge(train_values, train_labels, on="building_id")

# === Separazione X, y e test ===
X_train = train_df.drop(columns=["building_id", "damage_grade"])
y_train = train_df["damage_grade"]
X_test = test_values.drop(columns=["building_id"])

# ===Riconoscimento automatico delle colonne ===
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
numeric_cols = X_train.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric/binary columns:", numeric_cols)

""" #=====GESTIONE FEATURE NUMERICHE =====
# ===Winsorization per contenere gli outlier numerici ===
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
<<<<<<< HEAD
X_test[numeric_cols] = X_test_wins """

# === Winsorization corretta: calcolo limiti dal train ===
def compute_winsor_limits(df, cols, lower=0.01, upper=0.99):
    return {col: (df[col].quantile(lower), df[col].quantile(upper)) for col in cols}

def apply_winsor_limits(df, limits):
    df_capped = df.copy()
    for col, (low, high) in limits.items():
        df_capped[col] = np.clip(df[col], low, high)
    return df_capped


# Calcola i limiti sul training set
winsor_limits = compute_winsor_limits(X_train, numeric_cols)

# Applica i limiti sia a train che a test
X_train[numeric_cols] = apply_winsor_limits(X_train[numeric_cols], winsor_limits)
X_test[numeric_cols] = apply_winsor_limits(X_test[numeric_cols], winsor_limits)

print("\n=== Winsorization completata ===")
for col in numeric_cols[:3]:  # Mostra solo le prime 3 per brevità
    low, high = winsor_limits[col]
    print(f"Colonna '{col}': limiti applicati [{low:.2f}, {high:.2f}]")

print("Esempio valori dopo winsorization (train):")
print(X_train[numeric_cols[:3]].describe())

# === Scaling delle colonne numeriche ===
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_num_scaled = scaler.transform(X_test[numeric_cols])

X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=numeric_cols, index=X_train.index)
X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=numeric_cols, index=X_test.index)

print("\n=== Scaling completato ===")
print("Media e std (prime 3 feature numeriche, train):")
print(X_train_num_df[numeric_cols[:3]].agg(['mean', 'std']))

#=====GESTIONE FEATURE CATEGORICHE=====
# ===OneHotEncoding sulle categoriche ===
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])


# ===Ricostruzione dei DataFrame encoded ===
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoded_col_names, index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoded_col_names, index=X_test.index)

print("\n=== One-Hot Encoding completato ===")
print(f"Colonne categoriche originali: {len(categorical_cols)}")
print(f"Nuove colonne generate: {len(encoded_col_names)}")
print("Esempio colonne codificate:", list(encoded_col_names[:5]))


# ===Unione numeriche scalate + categoriche codificate ===
X_train_final = pd.concat([X_train_num_df.reset_index(drop=True), 
                           X_train_cat_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_num_df.reset_index(drop=True), 
                          X_test_cat_df.reset_index(drop=True)], axis=1)

print("\n=== Unione numeriche + categoriche completata ===")
print("Shape finale X_train:", X_train_final.shape)
print("Shape finale X_test:", X_test_final.shape)

# === SPLIT ===
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print("\n=== Train/Validation split completato ===")
print("Shape X_train_split:", X_train_split.shape)
print("Shape X_val_split:", X_val_split.shape)
print("Distribuzione y_train_split:")
print(y_train_split.value_counts(normalize=True))

# === FEATURE SELECTION ===
# Rimuovi le feature con varianza molto bassa (es. dummies inutili)
var_thresh = VarianceThreshold(threshold=0.01)
X_train_split_var = var_thresh.fit_transform(X_train_split)
X_val_split_var = var_thresh.transform(X_val_split)

# Stampa il numero di feature rimaste dopo VarianceThreshold
num_features_var = X_train_split_var.shape[1]
print(f"\n=== VarianceThreshold completato ===")
print(f"Numero di feature dopo rimozione bassa varianza: {num_features_var}")

# Seleziona le migliori k feature secondo ANOVA F-test
k_best = 50  # puoi aumentare o diminuire dopo test
selector = SelectKBest(score_func=f_classif, k=k_best)
X_train_split_sel = selector.fit_transform(X_train_split_var, y_train_split)
X_val_split_sel = selector.transform(X_val_split_var)

# Recupera i nomi delle colonne selezionate
# Serve ricostruire i nomi dopo VarianceThreshold
feature_names_after_var = X_train_split.columns[var_thresh.get_support()]
selected_feature_names = feature_names_after_var[selector.get_support()]

print(f"\n=== SelectKBest completato ===")
print(f"Top {k_best} feature selezionate (ANOVA F-test):")
for i, name in enumerate(selected_feature_names):
    print(f"{i+1}. {name}")

# Sovrascrivi le variabili da passare a SMOTE e ai modelli
X_train_split = X_train_split_sel
X_val_split = X_val_split_sel

# === SMOTE SOLO SUL TRAINING ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_split, y_train_split)

print("\n=== SMOTE applicato ===")
print("Distribuzione y_train_bal (dopo oversampling):")
print(y_train_bal.value_counts(normalize=True))

# Salvataggio dei dati preprocessati
joblib.dump((X_train_bal, y_train_bal, X_val_split, y_val_split, X_test_final), "preprocessed_data.pkl")

print("\n=== Dati preprocessati salvati in 'preprocessed_data.pkl' ===")

