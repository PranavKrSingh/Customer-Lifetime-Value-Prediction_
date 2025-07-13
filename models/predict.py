
import joblib, pandas as pd, os

MODEL_PATH = os.path.join(os.path.dirname(__file__),'clv_model.pkl')

_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def predict_single(features: dict):
    if _model is None:
        return 'Model not trained.'
    df = pd.DataFrame([features])
    return round(float(_model.predict(df)[0]), 2)
