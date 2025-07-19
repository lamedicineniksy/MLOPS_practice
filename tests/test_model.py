# File: tests/test.py

import joblib
import pandas as pd
from sklearn.datasets import load_iris
import os

def test_model_prediction():
    # Load model
    model_path = "models/model.joblib"
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    
    model = joblib.load(model_path)

    # Load sample data
    iris = load_iris(as_frame=True)
    df = iris.frame
    X_sample = df.drop("target", axis=1).iloc[:5]

    # Predict
    preds = model.predict(X_sample)

    # Assertions
    assert len(preds) == 5, "Model should return 5 predictions"
    assert preds.dtype in [int, float, 'int32', 'int64', 'float64'], f"Unexpected dtype: {preds.dtype}"

if __name__ == "__main__":
    test_model_prediction()
    print("✅ test_model_prediction passed.")

