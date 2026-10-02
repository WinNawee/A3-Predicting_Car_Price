"""
Shared inference code for the A3 Car Price Classifier.

Deliberately tiny and dependency-light: inference only needs the fitted
`prep` (sklearn ColumnTransformer) and `W` (a plain numpy weight matrix)
saved by the training notebook -- not the custom `LogisticRegression`
class. That means this module (and the pickle it loads) has no dependency
on notebook-only code, so there's nothing that can go missing when the
bundle is unpickled in the deployed container.

Used by:
  - app/code/pages/predict.py (the Dash page)
  - app/tests/test_model.py (unit tests)
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model" / "car_price_classifier.pkl"

STUDENT_ID = os.environ.get("STUDENT_ID", "st127031")
MODEL_NAME = f"{STUDENT_ID}-a3-model"
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Staging")
AIT_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th")

# The course MLflow server is unstable (TA announcement, 1 Oct 2026), so
# the app serves from the local bundle by default. Set
# USE_MLFLOW_REGISTRY=1 to load from the registry instead. If the
# server hangs, a long MLflow request at startup would exceed gunicorn's
# worker boot timeout and take the whole site down -- so the registry
# path is opt-in and uses a short timeout with no retries.
USE_MLFLOW_REGISTRY = os.environ.get("USE_MLFLOW_REGISTRY", "0") == "1"


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def load_predictor():
    """Serve from the local bundle saved by the training notebook, or,
    when USE_MLFLOW_REGISTRY=1, from the MLflow Model Registry (falling
    back to the local bundle if the registry can't be reached). Returns
    (predict_fn, class_labels, class_price_ranges, source_label).
    """
    local_bundle = load_bundle()

    if not USE_MLFLOW_REGISTRY:
        def predict_fn(raw_df):
            return predict(local_bundle, raw_df)

        return predict_fn, local_bundle["class_labels"], local_bundle["class_price_ranges"], "local model file"

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

    try:
        import mlflow
        import mlflow.pyfunc

        mlflow.set_tracking_uri(AIT_MLFLOW_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)

        def predict_fn(raw_df):
            out = pyfunc_model.predict(raw_df[local_bundle["feature_cols"]])
            preds = out["predicted_class"].to_numpy()
            proba_cols = sorted(c for c in out.columns if c.startswith("proba_class_"))
            probs = out[proba_cols].to_numpy()
            return preds, probs

        print(f"Loaded model from MLflow registry: {model_uri}")
        return predict_fn, local_bundle["class_labels"], local_bundle["class_price_ranges"], f"MLflow registry ({model_uri})"

    except Exception as e:
        print(f"Could not load model from MLflow registry ({type(e).__name__}: {e}).")
        print(f"Falling back to local bundle: {MODEL_PATH}")

    def predict_fn(raw_df):
        return predict(local_bundle, raw_df)

    return predict_fn, local_bundle["class_labels"], local_bundle["class_price_ranges"], "local model file (registry unreachable)"


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def predict(bundle, raw_df: pd.DataFrame):
    """raw_df: DataFrame with columns matching bundle['feature_cols']
    (year, km_driven, mileage, engine, max_power, seats, owner [already
    ordinal-encoded as an int], fuel, transmission, seller_type, brand).

    Returns (predicted_class_array, probability_matrix)."""
    X = bundle["prep"].transform(raw_df[bundle["feature_cols"]])
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    probs = softmax(X @ bundle["W"])
    preds = np.argmax(probs, axis=1)
    return preds, probs
