import joblib
import pandas as pd

def predict_single(model_path, sample_dict):
    # Load model package (model + features)
    package = joblib.load(model_path)
    model = package["model"]
    feature_names = package["features"]

    # Build dataframe with correct feature order
    sample = pd.DataFrame([sample_dict])
    sample = sample.reindex(columns=feature_names, fill_value=0)

    proba = model.predict_proba(sample)[:, 1][0]
    pred = model.predict(sample)[0]

    return {
        "prediction": int(pred),       # 0 = Stay, 1 = Churn
        "probability": float(proba)
    }
