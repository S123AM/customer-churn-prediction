# -*- coding: utf-8 -*-
"""
02_model_training.py
--------------------
This script trains multiple machine learning models 
for the Customer Churn Prediction project.
It includes preprocessing (encoding + scaling) and saves the trained models.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -------------------------------------------------------------------
# 1. Load cleaned dataset
# -------------------------------------------------------------------
try:
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "cleaned_telco.csv"
    )
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded cleaned dataset with shape {df.shape}")
except FileNotFoundError:
    print(f"[ERROR] Could not find cleaned dataset at {DATA_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load dataset → {e}")
    sys.exit(1)


# -------------------------------------------------------------------
# 2. Split data into features & target
# -------------------------------------------------------------------
try:
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
except Exception as e:
    print(f"[ERROR] Failed during train-test split → {e}")
    sys.exit(1)


# -------------------------------------------------------------------
# 3. Preprocessing (OneHot for categorical, Scale for numeric)
# -------------------------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"[INFO] Categorical columns: {categorical_cols}")
print(f"[INFO] Numeric columns: {numeric_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)


# -------------------------------------------------------------------
# 4. Define models
# -------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
    ),
}


# -------------------------------------------------------------------
# 5. Train and save models
# -------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

for name, model in models.items():
    try:
        print(f"\n[INFO] Training {name} ...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)

        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"[SUCCESS] Saved {name} → {model_path}")
    except Exception as e:
        print(f"[ERROR] Training {name} failed → {e}")
