import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier

# === Caricamento dati preprocessati ===
X_train, y_train, _, _, _ = joblib.load("data/processed_unprocessed/preprocessed_unbalanced.pkl")

def tune_catboost(X, y):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer, f1_score

    print("\n=== Tuning CatBoost ===")

    param_dist = {
        'iterations': [100, 300, 500],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    cat = CatBoostClassifier(verbose=0, random_state=42)

    f1_micro = make_scorer(f1_score, average='micro')

    search = RandomizedSearchCV(
        estimator=cat,
        param_distributions=param_dist,
        scoring=f1_micro,
        n_iter=20,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X, y)

    print("Migliori parametri:", search.best_params_)
    print("Miglior F1 micro:", search.best_score_)

    return search.best_estimator_

def tune_randomforest(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer, f1_score

    print("\n=== Tuning Random Forest ===")

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    f1_micro = make_scorer(f1_score, average='micro')

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        scoring=f1_micro,
        n_iter=20,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X, y)

    print("Migliori parametri:", search.best_params_)
    print("Miglior F1 micro:", search.best_score_)

    return search.best_estimator_


def tune_xgboost(X, y):
    # Rimappa le classi da [1, 2, 3] a [0, 1, 2]
    y_mapped = y - 1

    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    f1_macro = make_scorer(f1_score, average='macro')

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=f1_macro,
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X, y_mapped)

    print("\n=== Migliori iperparametri trovati ===")
    print(grid.best_params_)
    print("Miglior F1 macro in cross-validation:", grid.best_score_)

    return grid.best_estimator_

