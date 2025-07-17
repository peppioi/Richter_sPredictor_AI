import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# === Caricamento dei dati ===
train_values = pd.read_csv("train_values.csv")
train_labels = pd.read_csv("train_labels.csv")
test_values = pd.read_csv("test_values.csv")

# === Merge tra valori e etichette per il train ===
train_df = train_values.merge(train_labels, on="building_id")

# === Separazione X e y ===
X_train = train_df.drop(columns=["building_id", "damage_grade"])
y_train = train_df["damage_grade"]
X_test = test_values.drop(columns=["building_id"])

# === Identificazione delle colonne categoriche e numeriche ===
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# === Costruzione del preprocessing pipeline ===

# Imputer per i numerici: sostituisce eventuali NaN con la mediana
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Imputer + OneHotEncoder per le categoriche
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing completo con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Applica il preprocessing a X_train e X_test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# === Output shape (utile per debug) ===
print(f"Shape X_train processed: {X_train_processed.shape}")
print(f"Shape X_test processed: {X_test_processed.shape}")

np.save("X_train_processed.npy", X_train_processed)
np.save("X_test_processed.npy", X_test_processed)
