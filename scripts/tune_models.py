import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# === Caricamento dati preprocessati ===
X_train, y_train, _, _, _ = joblib.load("data/processed_unprocessed/preprocessed_unbalanced.pkl")

# === Tuning CatBoost ===
print("\nInizio tuning CatBoost...")
param_dist_cat = {
    'iterations': [100, 300, 500],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

cat = CatBoostClassifier(verbose=0, random_state=42)
search_cat = RandomizedSearchCV(cat, param_distributions=param_dist_cat,
                                n_iter=20, cv=3, scoring='f1_micro',
                                n_jobs=-1, random_state=42)
search_cat.fit(X_train, y_train)

best_cat = search_cat.best_estimator_
joblib.dump(best_cat, "models/best_cat_model.pkl")
print(" CatBoost tuning completato. Miglior F1 micro:", search_cat.best_score_)

# === Tuning RandomForest ===
print("\n🔍 Inizio tuning RandomForest...")
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf,
                               n_iter=20, cv=3, scoring='f1_micro',
                               n_jobs=-1, random_state=42)
search_rf.fit(X_train, y_train)

best_rf = search_rf.best_estimator_
joblib.dump(best_rf, "models/best_rf_model.pkl")
print(" RandomForest tuning completato. Miglior F1 micro:", search_rf.best_score_)
