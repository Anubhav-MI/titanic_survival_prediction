# train_model.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessor import get_preprocessor
from load_data import load_titanic_data

def train_and_save_model():
    X, y = load_titanic_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', get_preprocessor()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'ml_pipeline.pkl')
    return "Model trained and saved."
