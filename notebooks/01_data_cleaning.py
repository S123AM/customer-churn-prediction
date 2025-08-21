# -*- coding: utf-8 -*-
"""
01_data_cleaning.py
-------------------
This script handles:
    - Loading raw Telco Customer Churn dataset
    - Cleaning & preprocessing (fix data types, handle missing values, remove duplicates)
    - Saving a cleaned version for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def main():
    # -------------------- PATH SETUP --------------------
    try:
        HERE = Path(__file__).resolve().parent
        ROOT = HERE.parent
        DATA_DIR = ROOT / "data"
        REPORTS_DIR = ROOT / "reports"
        MODELS_DIR = ROOT / "models"

        for d in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        print(f"[ERROR] Failed to set up directories → {e}")
        sys.exit(1)

    # -------------------- FIND DATASET --------------------
    try:
        candidates = [
            "Telco-Customer-Churn.csv",
            "WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "telco_customer_churn.csv",
        ]
        csv_path = None
        for name in candidates:
            p = DATA_DIR / name
            if p.exists():
                csv_path = p
                break

        # If none of the known names exist → pick first CSV
        if csv_path is None:
            found = list(DATA_DIR.glob("*.csv"))
            if not found:
                raise FileNotFoundError(
                    f"No CSV file found in {DATA_DIR}. Please add dataset file."
                )
            csv_path = found[0]

        print(f"[INFO] Using dataset: {csv_path.name}")

    except Exception as e:
        print(f"[ERROR] Dataset not found → {e}")
        sys.exit(1)

    # -------------------- LOAD DATA --------------------
    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded dataset with shape {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV → {e}")
        sys.exit(1)

    # -------------------- CLEANING STEPS --------------------
    try:
        # 1) Strip whitespace around text columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        # 2) Convert TotalCharges to numeric if exists
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # 3) Drop identifier columns
        for c in ["customerID", "CustomerID", "id", "ID"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        # 4) Encode target Churn → {Yes=1, No=0}
        if "Churn" in df.columns and df["Churn"].dtype == "object":
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("Int64")

        # 5) Handle missing values
        num_cols = df.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        for c in num_cols:
            if df[c].isna().any():
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)

        for c in cat_cols:
            if df[c].isna().any():
                mode_val = df[c].mode().iloc[0]
                df[c] = df[c].fillna(mode_val)

        # 6) Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if after < before:
            print(f"[INFO] Removed {before - after} duplicate rows")

    except Exception as e:
        print(f"[ERROR] Cleaning step failed → {e}")
        sys.exit(1)

    # -------------------- SAVE CLEANED DATA --------------------
    try:
        clean_path = DATA_DIR / "cleaned_telco.csv"
        df.to_csv(clean_path, index=False, encoding="utf-8")
        print(f"[SUCCESS] Cleaned dataset saved → {clean_path}")
        print(f"[INFO] Final shape: {df.shape}")

        if "Churn" in df.columns:
            print("[INFO] Target distribution (Churn):")
            print(df["Churn"].value_counts(dropna=False))

    except Exception as e:
        print(f"[ERROR] Failed to save cleaned dataset → {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
