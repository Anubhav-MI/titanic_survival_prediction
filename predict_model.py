# predict_model.py
import pandas as pd
import joblib

def predict_from_input(input_data: dict):
    model = joblib.load('ml_pipeline.pkl')
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return int(prediction[0])
