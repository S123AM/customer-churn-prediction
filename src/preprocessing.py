# src/preprocessing.py
"""
preprocessing.py
----------------
Utility functions for:
    - Loading dataset
    - Splitting into train/test
    - Scaling features
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Split dataset into train/test and apply scaling.

    Args:
        df (pd.DataFrame): input dataframe
        target_col (str): target column name

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
