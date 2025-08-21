
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.preprocessing import load_data, preprocess_data

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

def train_and_save_model(data_path, model_path):
    # Load & preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    pipe = Pipeline([
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    # Save model + feature names together
    model_package = {
        "model": grid.best_estimator_,
        "features": list(X_train.columns)
    }

    joblib.dump(model_package, model_path)
    return grid.best_estimator_, X_test, y_test
