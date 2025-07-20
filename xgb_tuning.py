from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier

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
