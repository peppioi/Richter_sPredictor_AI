import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

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

# === Feature Engineering: Superstructure Quality ===
def get_superstructure_quality(df):
    df["superstructure_quality"] = -1

    good = ["has_superstructure_bamboo", "has_superstructure_rc_engineered",
            "has_superstructure_rc_non_engineered", "has_superstructure_timber"]
    bad = ["has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
           "has_superstructure_cement_mortar_stone", "has_superstructure_mud_mortar_brick",
           "has_superstructure_cement_mortar_brick", "has_superstructure_stone_flag"]

    has_good = df[good].any(axis=1)
    has_bad = df[bad].any(axis=1)

    df.loc[has_good, "superstructure_quality"] = 1
    df.loc[df["has_superstructure_other"] == 1, "superstructure_quality"] = 0
    df.loc[
        (has_good & (df["has_superstructure_other"] == 1)) |
        (has_good & has_bad) |
        ((df["has_superstructure_other"] == 1) & has_bad),
        "superstructure_quality"
    ] = 0

    return df

X_train = get_superstructure_quality(X_train)
X_test = get_superstructure_quality(X_test)


# === Feature Engineering: Geo Risk based on geo_level_1_id ===
def get_geo_risk(df_raw, y_raw, df_target, geo_level=1):
    joined = df_raw.join(y_raw)
    risk_stats = joined.groupby(f"geo_level_{geo_level}_id")["damage_grade"].value_counts(normalize=True).unstack(fill_value=0)

    for grade in [1, 2, 3]:
        df_target[f"geo_{geo_level}_risk_grade_{grade}"] = df_target[f"geo_level_{geo_level}_id"].map(risk_stats.get(grade, 0))

    return df_target

X_train = get_geo_risk(X_train, y_train, X_train, geo_level=1)
X_test = get_geo_risk(X_train, y_train, X_test, geo_level=1)

# ===Riconoscimento automatico delle colonne ===
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
numeric_cols = X_train.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric/binary columns:", numeric_cols)

print("\n=== Conteggio outlier (percentili 1% e 99%) per feature numerica ===")
outlier_counts = {}

for col in numeric_cols:
    unique_vals = X_train[col].nunique()
    if unique_vals <= 2:
        print(f"{col}: variabile binaria - outlier non applicabile")
        continue

    q_low = X_train[col].quantile(0.01)
    q_high = X_train[col].quantile(0.99)
    count_low = (X_train[col] < q_low).sum()
    count_high = (X_train[col] > q_high).sum()
    total_outliers = count_low + count_high
    outlier_counts[col] = total_outliers
    print(f"{col}: {total_outliers} outlier ({count_low} < {q_low:.2f}, {count_high} > {q_high:.2f})")

print(f"\nTotale outlier sommati su tutte le feature non binarie: {sum(outlier_counts.values())}")

# === Scaling delle colonne numeriche ===
scaler = RobustScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_num_scaled = scaler.transform(X_test[numeric_cols])

X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=numeric_cols, index=X_train.index)
X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=numeric_cols, index=X_test.index)

print("\n=== Scaling completato ===")
print("Media e std (prime 3 feature numeriche, train):")
print(X_train_num_df[numeric_cols[:3]].agg(['mean', 'std']))

#=====GESTIONE FEATURE CATEGORICHE=====
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoded_col_names, index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoded_col_names, index=X_test.index)

print("\n=== One-Hot Encoding completato ===")
print(f"Colonne categoriche originali: {len(categorical_cols)}")
print(f"Nuove colonne generate: {len(encoded_col_names)}")
print("Esempio colonne codificate:", list(encoded_col_names[:5]))

X_train_final = pd.concat([X_train_num_df.reset_index(drop=True), 
                           X_train_cat_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_num_df.reset_index(drop=True), 
                          X_test_cat_df.reset_index(drop=True)], axis=1)

print("\n=== Unione numeriche + categoriche completata ===")
print("Shape finale X_train:", X_train_final.shape)
print("Shape finale X_test:", X_test_final.shape)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print("\n=== Train/Validation split completato ===")
print("Shape X_train_split:", X_train_split.shape)
print("Shape X_val_split:", X_val_split.shape)
print("Distribuzione y_train_split:")
print(y_train_split.value_counts(normalize=True))

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_selector.fit(X_train_split, y_train_split)

sfm = SelectFromModel(rf_selector, threshold="median")
X_train_split_sel = sfm.transform(X_train_split)
X_val_split_sel = sfm.transform(X_val_split)

selected_feature_names = X_train_split.columns[sfm.get_support()]

print(f"\n=== SelectFromModel (Random Forest) completato ===")
print(f"Feature selezionate: {len(selected_feature_names)}")
for i, name in enumerate(selected_feature_names):
    print(f"{i+1}. {name}")

X_train_split = X_train_split_sel
X_val_split = X_val_split_sel

# === Salvataggio dati NON bilanciati ===
joblib.dump((X_train_split, y_train_split, X_val_split, y_val_split, X_test_final), "preprocessed_unbalanced.pkl")
print("\n=== Dati NON bilanciati salvati in 'preprocessed_unbalanced.pkl' ===")

# === SMOTE SOLO SUL TRAINING ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_split, y_train_split)

print("\n=== SMOTE applicato ===")
print("Distribuzione y_train_bal (dopo oversampling):")
print(y_train_bal.value_counts(normalize=True))

joblib.dump((X_train_bal, y_train_bal, X_val_split, y_val_split, X_test_final), "preprocessed_data.pkl")
print("\n=== Dati preprocessati salvati in 'preprocessed_data.pkl' ===")
